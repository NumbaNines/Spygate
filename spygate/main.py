"""
Spygate - Madden NFL 25 Game Analysis Tool
Main application entry point
"""

import sys

from PyQt6.QtWidgets import QApplication

from spygate.database.config import init_db
from spygate.gui.components.main_window import MainWindow


def main():
    """Application entry point."""
    # Initialize the database
    init_db()

    app = QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
