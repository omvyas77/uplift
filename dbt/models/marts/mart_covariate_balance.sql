{{ config(materialized='table') }}

{#
  The randomization check, in SQL. Under a valid randomization every |smd|
  should be tiny. A failure here means the assumption underpinning every
  downstream number in this repo is broken, which is why the Dagster asset
  check on this model is BLOCKING.
#}

with unpivoted as (
    {% for i in range(12) %}
    select 'f{{ i }}' as feature, is_treated, f{{ i }} as value
    from {{ ref('int_units_with_splits') }}
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
),

stats as (
    select
        feature,
        is_treated,
        count(*)        as n,
        avg(value)      as mean_value,
        var_samp(value) as var_value
    from unpivoted
    group by 1, 2
)

select
    feature,
    max(case when is_treated = 1 then mean_value end) as mean_treatment,
    max(case when is_treated = 0 then mean_value end) as mean_control,
    max(case when is_treated = 1 then var_value end)
      / nullif(max(case when is_treated = 0 then var_value end), 0) as var_ratio,
    -- standardized mean difference, pooled-SD version
    (max(case when is_treated = 1 then mean_value end)
     - max(case when is_treated = 0 then mean_value end))
    / nullif(sqrt((
        max(case when is_treated = 1 then var_value end)
        + max(case when is_treated = 0 then var_value end)
      ) / 2.0), 0) as smd
from stats
group by feature
