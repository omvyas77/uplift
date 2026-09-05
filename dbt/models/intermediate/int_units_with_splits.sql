{{ config(materialized='table') }}

with base as (select * from {{ ref('stg_criteo__units') }})

select
    *,
    {{ deterministic_bucket('unit_id') }} as bucket,
    case
        when {{ deterministic_bucket('unit_id') }} < 60 then 'train'
        when {{ deterministic_bucket('unit_id') }} < 80 then 'valid'
        else 'test'
    end as split
from base
