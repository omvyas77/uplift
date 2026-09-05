{{ config(materialized='table') }}

with per_arm as (
    select
        split,
        is_treated,
        count(*)                        as n_units,
        sum(visited)                    as n_visits,
        sum(converted)                  as n_conversions,
        sum(is_exposed)                 as n_exposed,
        avg(visited::double)            as visit_rate,
        avg(converted::double)          as conversion_rate,
        avg(is_exposed::double)         as exposure_rate
    from {{ ref('int_units_with_splits') }}
    group by 1, 2
)

select
    split,
    max(case when is_treated = 1 then n_units end)          as n_treatment,
    max(case when is_treated = 0 then n_units end)          as n_control,
    max(case when is_treated = 1 then visit_rate end)       as visit_rate_treatment,
    max(case when is_treated = 0 then visit_rate end)       as visit_rate_control,
    max(case when is_treated = 1 then visit_rate end)
      - max(case when is_treated = 0 then visit_rate end)   as visit_rate_lift_abs,
    (max(case when is_treated = 1 then visit_rate end)
      / nullif(max(case when is_treated = 0 then visit_rate end), 0)) - 1
                                                            as visit_rate_lift_rel,
    max(case when is_treated = 1 then conversion_rate end)  as conv_rate_treatment,
    max(case when is_treated = 0 then conversion_rate end)  as conv_rate_control,
    -- P(exposed | treated). This is the IV first stage; CACE = ITT / this.
    max(case when is_treated = 1 then exposure_rate end)    as compliance_rate,
    max(case when is_treated = 1 then n_units end)::double
      / nullif(sum(n_units), 0)                             as observed_treatment_ratio
from per_arm
group by split
