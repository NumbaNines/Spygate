Card Component
=============

The Card component provides a themed container with header, content, and footer sections.

.. automodule:: src.gui.components.composite.card
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
------------

.. code-block:: python

   from src.gui.components.composite import Card
   from PyQt6.QtWidgets import QLabel
   
   # Create a card with a title
   card = Card(title="Player Statistics")
   
   # Add content
   content = QLabel("Player stats will be displayed here")
   card.set_content(content)
   
   # Add footer
   footer = QLabel("Last updated: Today")
   card.set_footer(footer)
   
   # Make it collapsible
   card.set_collapsible(True)
   
   # Set elevation
   card.set_elevation(2) 