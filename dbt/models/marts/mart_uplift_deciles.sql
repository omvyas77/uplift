{{ config(materialized='table') }}

{#
  Observed uplift by decile of a covariate proxy. This is NOT a model-score
  decile table - the model scores live in Python/Parquet and the calibration
  table in src/uplift/evaluation/calibration.py is the real one. This model
  exists so the dashboard has a SQL-native uplift-by-group view and so the
  decile machinery is testable in dbt.
#}

with binned as (
    select
        *,
        ntile(10) over (order by cupac_covariate, unit_id) as decile
    from {{ ref('fct_experiment_units') }}
    where cupac_covariate is not null
)

select
    decile,
    count(*)                                                  as n_units,
    sum(case when is_treated = 1 then 1 else 0 end)           as n_treatment,
    sum(case when is_treated = 0 then 1 else 0 end)           as n_control,
    avg(case when is_treated = 1 then visited::double end)    as visit_rate_treatment,
    avg(case when is_treated = 0 then visited::double end)    as visit_rate_control,
    avg(case when is_treated = 1 then visited::double end)
      - avg(case when is_treated = 0 then visited::double end) as observed_uplift
from binned
group by decile
order by decile
