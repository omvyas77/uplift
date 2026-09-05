{#
  On the `dev` target the raw table is the DuckDB view built by `uplift ingest`.
  On the `ci` target it is the committed 50k-row seed, so the whole dbt project
  builds in Actions without downloading 311 MB.
#}
{% macro raw_units() %}
    {% if target.name == 'ci' %}
        {{ ref('criteo_sample') }}
    {% else %}
        {{ source('raw', 'units') }}
    {% endif %}
{% endmacro %}
