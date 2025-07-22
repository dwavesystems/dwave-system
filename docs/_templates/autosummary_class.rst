{{ objname | underline}}

.. currentmodule:: {{ module }}

{{ 'Class' }}
{{ '=====' }}

.. autoclass:: {{ objname }}

{{ 'Class Members: Summary' }}
{{ '======================' }}

{% block attributes %}
{% if attributes %}
{{ 'Properties' }}
{{ '----------' }}
.. autosummary::
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
{{ 'Methods' }}
{{ '-------' }}
.. autosummary::
    :nosignatures:
{% for item in methods %}
    {%- if not item.startswith('_') %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{{ 'Class Members: Descriptions' }}
{{ '===========================' }}

{% block attributes2 %}
{% if attributes %}
{{ 'Properties' }}
{{ '----------' }}
{% for item in attributes %}
.. autoattribute:: {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods2 %}
{% if methods %}
{{ 'Methods' }}
{{ '-------' }}
{% for item in methods %}
{%- if not item.startswith('_') %}
.. automethod:: {{ name }}.{{ item }}
{%- endif -%}
{% endfor %}
{% endif %}
{% endblock %}