{{ config(materialized='view') }}

select
    unit_id,
    {% for i in range(12) %}
    cast(f{{ i }} as double) as f{{ i }},
    {% endfor %}
    cast(treatment  as tinyint) as is_treated,
    cast(exposure   as tinyint) as is_exposed,
    cast(visit      as tinyint) as visited,
    cast(conversion as tinyint) as converted
from {{ raw_units() }}
