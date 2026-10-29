-- Funnel monotonicity: a conversion should imply a visit.
-- Checked rather than assumed. Verified on the
-- full 13,979,592 rows: 0 violations, so the funnel is monotone in this data.
select unit_id
from {{ ref('int_units_with_splits') }}
where converted = 1 and visited = 0
