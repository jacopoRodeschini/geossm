{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   {%- if objname in ["LRStateSpaceModel", "LRStateSpaceResults"] %}
   :inherited-members:
   {%- endif %}

   {% block methods %}
   {% set own_methods = methods | reject("equalto", "__init__") | list %}
   {% if objname not in ["LRStateSpaceModel", "LRStateSpaceResults"] %}
   {% set own_methods = own_methods | reject("in", inherited_members) | list %}
   {% endif %}
   {% if own_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set own_attributes = attributes %}
   {% if objname not in ["LRStateSpaceModel", "LRStateSpaceResults"] %}
   {% set own_attributes = own_attributes | reject("in", inherited_members) | list %}
   {% endif %}
   {% if own_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in own_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
