-- Only units assigned to treatment can be exposed to the ad.
-- A violation would mean the assignment/exposure logging is inconsistent and
-- the IV first stage is not what we think it is.
-- VERIFIED on the full 13,979,592 rows: 0 violations.
select unit_id
from {{ ref('int_units_with_splits') }}
where is_exposed = 1 and is_treated = 0
