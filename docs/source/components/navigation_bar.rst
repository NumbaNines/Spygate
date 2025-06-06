NavigationBar Component
===================

The NavigationBar component provides a flexible navigation interface with support for different styles and orientations.

.. automodule:: src.gui.components.composite.navigation_bar
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
------------

.. code-block:: python

   from src.gui.components.composite import NavigationBar
   
   # Create a horizontal navigation bar
   navbar = NavigationBar(orientation="horizontal", style="tabs")
   
   # Add navigation items
   navbar.add_item("home", "Home", icon="home.png")
   navbar.add_item("stats", "Statistics", icon="stats.png", badge="3")
   navbar.add_item("settings", "Settings", icon="settings.png")
   
   # Handle selection
   def on_select(item_id):
       print(f"Selected item: {item_id}")
   
   navbar.on_select = on_select
   
   # Set active item
   navbar.set_active("home") 