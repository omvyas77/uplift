{% macro deterministic_bucket(id_column, n_buckets=100) %}
    mod(abs(hash({{ id_column }})), {{ n_buckets }})
{% endmacro %}
