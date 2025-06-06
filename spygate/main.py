"""
Spygate - Madden NFL 25 Game Analysis Tool
Main application entry point
"""

import sys
from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from gui.components.main_window import MainWindow


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    apply_stylesheet(app, theme='dark_teal.xml')
    
    window = MainWindow(app)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 