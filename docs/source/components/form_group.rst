FormGroup Component
================

The FormGroup component provides a structured way to create and manage forms in the application.

.. automodule:: src.gui.components.composite.form_group
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
------------

.. code-block:: python

   from src.gui.components.composite import FormGroup
   
   # Create a form group
   form = FormGroup()
   
   # Add form fields
   form.add_field("username", "text", label="Username")
   form.add_field("password", "password", label="Password")
   form.add_field("remember", "checkbox", label="Remember me")
   
   # Add validation
   form.set_validation("username", lambda x: len(x) >= 3, "Username must be at least 3 characters")
   
   # Handle submission
   def on_submit(data):
       print(f"Form submitted with data: {data}")
   
   form.on_submit = on_submit 