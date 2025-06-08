Dialog Component
===============

The Dialog component provides a customizable modal dialog with configurable buttons and callbacks.

.. automodule:: src.gui.components.composite.dialog
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
------------

.. code-block:: python

   from src.gui.components.composite import Dialog
   from PyQt6.QtWidgets import QLabel

   # Create a confirmation dialog
   dialog = Dialog(
       title="Confirm Action",
       size=(400, 200)
   )

   # Set content
   content = QLabel("Are you sure you want to proceed?")
   dialog.set_content(content)

   # Add buttons
   dialog.add_button("Cancel", role="default", callback=lambda: dialog.close())
   dialog.add_button("Proceed", role="primary", callback=lambda: handle_confirmation())

   # Show the dialog
   dialog.show()
