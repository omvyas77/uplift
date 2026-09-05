{{ config(materialized='table') }}

{#
  A SQL stand-in for the CUPAC covariate: the control-group visit rate within a
  coarse cell of the two highest-variance features. The real covariate is the
  cross-fitted LightGBM prediction in src/uplift/experiment/cuped.py - this
  model exists so the cell-level control means are available to dbt tests and
  to the dashboard without a Python round-trip.

  Control-only by construction, so it carries no treatment information.
#}

with cells as (
    select
        *,
        ntile(10) over (order by f0) as f0_decile,
        ntile(10) over (order by f1) as f1_decile
    from {{ ref('int_units_with_splits') }}
),

control_means as (
    select
        f0_decile,
        f1_decile,
        count(*)                              as n_control,
        avg(visited::double)                  as control_visit_rate,
        avg(converted::double)                as control_conversion_rate
    from cells
    where is_treated = 0
    group by 1, 2
)

select
    c.unit_id,
    c.f0_decile,
    c.f1_decile,
    c.is_treated,
    c.visited,
    m.n_control,
    m.control_visit_rate as cupac_covariate
from cells c
left join control_means m using (f0_decile, f1_decile)
