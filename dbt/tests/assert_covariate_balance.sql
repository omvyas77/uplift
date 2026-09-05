-- The randomization gate as a singular test. Gates on the matching-literature
-- rule of thumb (|SMD| < 0.10), NOT the tight 0.02, because Criteo v2.1 fails
-- 0.02 on all 12 features and that failure is reported as a finding rather than
-- tuned away. See docs/findings.md and the WARN-level column test in schema.yml.
select feature, smd
from {{ ref('mart_covariate_balance') }}
where abs(smd) > {{ var('smd_blocking_tolerance') }}
