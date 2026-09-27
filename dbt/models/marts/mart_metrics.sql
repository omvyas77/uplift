{{ config(materialized='table') }}

{#
  THE METRICS LAYER.

  Every headline metric defined ONCE, in version control, next to the data it is
  computed from - so "north-star metric" is a file rather than a claim, and two
  dashboards cannot quietly disagree about what a visit rate is.

  WHY THIS IS A MODEL AND NOT A dbt SEMANTIC LAYER. The first attempt defined
  semantic_models + MetricFlow metrics. dbt refused to parse it:

      The semantic layer requires a time spine model with granularity DAY or
      smaller in the project, but none was found.

  MetricFlow is built around time-series aggregation, and this dataset has NO
  time dimension - no dates, no user ids, no sessions (see docs/findings.md).
  Satisfying the parser would have meant adding a calendar table that no metric
  uses, in a project whose stated finding is that there is no time column. That
  is boilerplate pretending to be a capability.

  So this is the fallback the build plan itself names: a metrics model plus
  docs/metric_definitions.md. The mf CLI is lost; the substance - definitions
  single-sourced, version-controlled, testable, and joinable - is kept. Knowing
  the difference between the substance and the tool is the point.
#}

with per_split as (
    select
        split,
        count(*)                                                         as units,
        sum(visited)                                                     as visits,
        sum(converted)                                                   as conversions,
        sum(is_exposed)                                                  as exposures,
        sum(case when is_treated = 1 then 1 else 0 end)                   as treated_units,
        sum(case when is_treated = 1 then is_exposed else 0 end)          as treated_exposures
    from {{ ref('int_units_with_splits') }}
    group by 1
),

overall as (
    select
        'all' as split,
        count(*)                                                         as units,
        sum(visited)                                                     as visits,
        sum(converted)                                                   as conversions,
        sum(is_exposed)                                                  as exposures,
        sum(case when is_treated = 1 then 1 else 0 end)                   as treated_units,
        sum(case when is_treated = 1 then is_exposed else 0 end)          as treated_exposures
    from {{ ref('int_units_with_splits') }}
),

combined as (
    select * from per_split
    union all
    select * from overall
)

select
    split,
    units,
    -- NORTH STAR. Chosen over conversion_rate because a 4.7% base rate gives
    -- usable precision; this design resolves 1.05% relative on visit against
    -- 4.76% on conversion.
    visits::double / nullif(units, 0)                        as visit_rate,
    -- SECONDARY / GUARDRAIL. Revenue lives here, but ~0.29% is too rare to
    -- rank on. Never the optimisation target.
    conversions::double / nullif(units, 0)                   as conversion_rate,
    -- IV FIRST STAGE. CACE = ITT / compliance_rate. ~0.036 here, which is why
    -- the complier effect is roughly 28x the intention-to-treat effect.
    treated_exposures::double / nullif(treated_units, 0)     as compliance_rate,
    exposures::double / nullif(units, 0)                     as exposure_rate
from combined
