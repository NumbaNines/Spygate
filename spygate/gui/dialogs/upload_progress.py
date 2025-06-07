"""
Dialog for showing video upload and processing progress.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class UploadProgressDialog(QDialog):
    """Dialog showing progress of video upload and processing."""

    progress_updated = pyqtSignal(int)  # Signal for progress updates (0-100)

    def __init__(self, parent=None):
        """Initialize the dialog.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Uploading Video")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Progress frame
        progress_frame = QFrame()
        progress_frame.setStyleSheet(
            "QFrame {"
            "   background-color: #2A2A2A;"
            "   border-radius: 8px;"
            "   padding: 16px;"
            "}"
        )
        progress_layout = QVBoxLayout(progress_frame)

        # Status label
        self.status_label = QLabel("Processing video...")
        self.status_label.setStyleSheet("color: #D1D5DB;")
        progress_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            "QProgressBar {"
            "   background-color: #1E1E1E;"
            "   border: none;"
            "   border-radius: 4px;"
            "   text-align: center;"
            "}"
            "QProgressBar::chunk {"
            "   background-color: #3B82F6;"
            "   border-radius: 4px;"
            "}"
        )
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(progress_frame)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(
            "QPushButton {"
            "   background-color: #374151;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px 16px;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #4B5563;"
            "}"
        )
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        # Connect progress signal
        self.progress_updated.connect(self._update_progress)

        # Set window properties
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )
        self.setStyleSheet("background-color: #1E1E1E;")

    def _update_progress(self, value: int):
        """Update the progress bar value.

        Args:
            value: Progress value (0-100)
        """
        self.progress_bar.setValue(value)
        if value >= 100:
            self.status_label.setText("Processing complete!")
            self.cancel_button.setText("Close")

    def set_status(self, status: str):
        """Set the status message.

        Args:
            status: Status message to display
        """
        self.status_label.setText(status)
