"""
Simplified entry point for Spygate application
This bypasses import issues and focuses on core functionality for integration testing
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMainWindow(QMainWindow):
    """Simplified main window for testing basic functionality."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate - Integration Testing Mode")
        self.setMinimumSize(800, 600)

        # Create central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Add basic info
        info_label = QLabel("Spygate Integration Testing Mode\n\nApplication started successfully!")
        layout.addWidget(info_label)

        self.setCentralWidget(central_widget)


def test_functionality():
    """Test basic functionality - called by main app."""
    try:
        # Test PyQt6 import
        from PyQt6.QtWidgets import QApplication

        # Test basic file operations
        test_dir = Path("data")
        test_dir.mkdir(exist_ok=True)

        # Test logging
        import logging

        logging.info("Test functionality check passed")

        return True
    except Exception as e:
        print(f"Test functionality failed: {e}")
        return False


def main():
    """Simplified application entry point."""
    logger.info("Starting Spygate in integration testing mode")

    try:
        # Create Qt application
        app = QApplication(sys.argv)

        # Create and show main window
        window = SimpleMainWindow()
        window.show()

        logger.info("Application started successfully")
        return app.exec()

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
