-- The stronger, aggregate form of the exposure check: the control arm's
-- exposure rate must be exactly zero, which is what makes this one-sided
-- non-compliance and lets CACE = ITT / P(exposed | treated).
select split, avg(is_exposed::double) as control_exposure_rate
from {{ ref('int_units_with_splits') }}
where is_treated = 0
group by split
having avg(is_exposed::double) > 0
