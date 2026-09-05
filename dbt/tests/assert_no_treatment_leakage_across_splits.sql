-- The treatment ratio must be stable across splits. Drift here would mean the
-- split is correlated with assignment, which would bias held-out evaluation.
with s as (
    select split, avg(is_treated::double) as ratio
    from {{ ref('int_units_with_splits') }}
    group by 1
)
select * from s
where abs(ratio - 0.85) > {{ var('treatment_ratio_tolerance') }}
