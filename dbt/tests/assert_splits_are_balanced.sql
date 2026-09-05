-- Hash bucketing should give ~60/20/20. Allow the configured drift.
with s as (
    select split, count(*)::double / sum(count(*)) over () as frac
    from {{ ref('int_units_with_splits') }}
    group by 1
)
select * from s
where (split = 'train' and abs(frac - 0.60) > {{ var('split_tolerance') }})
   or (split = 'valid' and abs(frac - 0.20) > {{ var('split_tolerance') }})
   or (split = 'test'  and abs(frac - 0.20) > {{ var('split_tolerance') }})
