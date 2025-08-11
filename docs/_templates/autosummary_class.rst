{{ objname | underline}}

.. currentmodule:: {{ module }}

{{ 'Class' }}
{{ '-----' }}

.. autoclass:: {{ objname }}

{# Aesthetic section: drop if maintenance becomes a future difficulty #}
{% set counter_methods = namespace(count = 0) %}
{% for item in methods %}
    {%- if not item.startswith('_') %}
        {% set counter_methods.count = counter_methods.count + 1 %}
    {%- endif -%}
{%- endfor %}

{% if attributes or counter_methods.count > 0 %}
    {{- 'Class Members: Summary\n' }}
    {{- '----------------------'}}
{% endif %}

{% block attributes %}
{% if attributes %}
    {{- 'Properties\n' }}
    {{- '~~~~~~~~~~\n' }}
    {{- '.. autosummary::' }}
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {% endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if counter_methods.count > 0 %}
    {{- 'Methods\n' }}
    {{- '~~~~~~~\n' }}
    {{- '.. autosummary::' }}
    {{- '    :nosignatures:'}}
    {% for item in methods %}
        {%- if not item.startswith('_') %}
            ~{{ name }}.{{ item }}
        {%- endif -%}
    {% endfor %}
{% endif %}
{% endblock %}

{% if attributes or counter_methods.count > 0 %}
    {{- 'Class Members: Descriptions\n' }}
    {{- '---------------------------' }}
{% endif %}


{% block attributes2 %}
{% if attributes %}
    {{- 'Properties\n' }}
    {{- '~~~~~~~~~~\n' }}
    {% for item in attributes %}
.. autoattribute:: {{ name }}.{{ item }}
    {%- endfor %}
{% endif %}
{% endblock %}

{% block methods2 %}
{% if counter_methods.count > 0 %}
    {{- 'Methods\n' }}
    {{- '~~~~~~~\n' }}
    {% for item in methods %}
        {%- if not item.startswith('_') %}
.. automethod:: {{ name }}.{{ item }}
        {%- endif -%}
    {% endfor %}
{% endif %}
{% endblock %}