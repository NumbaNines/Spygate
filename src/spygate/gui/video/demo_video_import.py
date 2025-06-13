import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from .video_import import VideoImportWidget


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Import Demo")
        self.setMinimumSize(800, 600)

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3daee9;
                border: none;
                border-radius: 3px;
                padding: 8px 15px;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4dbef9;
            }
            QPushButton:pressed {
                background-color: #2d9ed9;
            }
        """
        )

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Add instructions
        instructions = QLabel(
            "Drag and drop video files here to import.\n"
            "Supported formats: MP4, AVI, MOV, MKV\n"
            "Supported codecs: H.264, H.265\n\n"
            "Click the delete button next to a video to remove it."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Add video import widget
        self.video_import = VideoImportWidget()
        layout.addWidget(self.video_import)

        # Add status label
        self.status_label = QLabel("Ready for import")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Connect signals
        self.video_import.videosImported.connect(self._on_videos_imported)
        self.video_import.videoDeleted.connect(self._on_video_deleted)

    def _on_videos_imported(self, imported_videos):
        """Handle imported videos."""
        num_videos = len(imported_videos)
        self.status_label.setText(
            f"Successfully imported {num_videos} video{'s' if num_videos != 1 else ''}"
        )

    def _on_video_deleted(self, video_path):
        """Handle video deletion."""
        self.status_label.setText(f"Deleted video: {video_path}")


def main():
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
