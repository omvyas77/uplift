-- Overall allocation must match the published 85/15 design.
select 1 as failure
from {{ ref('int_units_with_splits') }}
having abs(avg(is_treated::double) - 0.85) > {{ var('treatment_ratio_tolerance') }}
