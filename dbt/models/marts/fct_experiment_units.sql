{{ config(materialized='table') }}

{#
  The analysis-ready fact table: one row per randomized unit, with the split,
  the bucket, and the SQL CUPAC covariate joined on.

  `is_exposed` is carried here because the IV/CACE module legitimately needs it.
  Every consumer that is NOT the IV module reads the feature columns and
  is_treated only - see src/uplift/data/io.py, where load_split() omits it and
  load_iv_split() is the single function that returns it.
#}

select
    u.unit_id,
    {% for i in range(12) %}
    u.f{{ i }},
    {% endfor %}
    u.is_treated,
    u.is_exposed,
    u.visited,
    u.converted,
    u.bucket,
    u.split,
    c.cupac_covariate
from {{ ref('int_units_with_splits') }} u
left join {{ ref('int_cupac_covariate') }} c using (unit_id)
