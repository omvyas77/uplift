-- A conversion implies a visit (verified: 0 violations on all 13,979,592 rows),
-- so the conversion rate can never exceed the visit rate. If this fires, either
-- the funnel assumption broke or the metrics model is computing the wrong ratio.
select split, visit_rate, conversion_rate
from {{ ref('mart_metrics') }}
where conversion_rate > visit_rate
