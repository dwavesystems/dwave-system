{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{{ 'Functions: Summary' }}
{{ '------------------' }}

{% block functions %}
{# To prevent template error, but template should not be applied to such modules #}
{% if functions %}
   {{- '.. autosummary::' }}
   {%- for item in functions %}
      {%- if not item.startswith('_') %}
      {{ item }}
      {%- endif %}
   {%- endfor %}
{% endif %}
{% endblock %}


{{ 'Functions: Descriptions'}}
{{ '-----------------------' }}

{% block functions2 %}
{# To prevent template error, but template should not be applied to such modules #}
{% if functions %}
   {%- for item in functions %}
      {%- if not item.startswith('_') %}
.. autofunction:: {{ item }}
      {%- endif %}
   {%- endfor %}
{% endif %}
{% endblock %}