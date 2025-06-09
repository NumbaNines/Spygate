"""
Dialog for showing video import progress.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

logger = logging.getLogger(__name__)


class UploadProgressDialog(QDialog):
    """Dialog showing progress of video import operations."""

    # Signal emitted when cancel is requested
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the upload progress dialog."""
        super().__init__(parent)
        self.setWindowTitle("Importing Video")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Processing video file...")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #3B82F6;
                border-radius: 4px;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
            }
        """
        )
        layout.addWidget(self.progress_bar)

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._handle_cancel)
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """
        )
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def set_progress(self, value: int, status: Optional[str] = None):
        """
        Update the progress bar and optionally the status text.

        Args:
            value: Progress value (0-100)
            status: Optional status text to display
        """
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)

    def _handle_cancel(self):
        """Handle cancel button click."""
        self.cancelled.emit()
        self.reject()
