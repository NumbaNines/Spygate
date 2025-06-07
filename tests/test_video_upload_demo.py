"""Demo script for testing video upload with player identification."""

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from src.ui.video import VideoImportWidget


class MainWindow(QMainWindow):
    """Main window for the demo."""

    def __init__(self):
        """Initialize the window."""
        super().__init__()
        self.setWindowTitle("Video Upload Demo")

        # Create and set the video import widget
        self.import_widget = VideoImportWidget()
        self.setCentralWidget(self.import_widget)

        # Connect signals
        self.import_widget.import_started.connect(self.on_import_started)
        self.import_widget.import_progress.connect(self.on_import_progress)
        self.import_widget.import_completed.connect(self.on_import_completed)
        self.import_widget.import_error.connect(self.on_import_error)

        # Set size
        self.resize(800, 600)

    def on_import_started(self):
        """Handle import started signal."""
        print("\nImport started")

    def on_import_progress(self, value):
        """Handle import progress signal.

        Args:
            value: Progress percentage
        """
        print(f"Import progress: {value}%")

    def on_import_completed(self, video_id):
        """Handle import completed signal.

        Args:
            video_id: ID of the imported video
        """
        print(f"\nImport completed! Video ID: {video_id}")

    def on_import_error(self, error):
        """Handle import error signal.

        Args:
            error: Error message
        """
        print(f"\nImport error: {error}")


def main():
    """Run the video upload demo."""
    # Create application
    app = QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
