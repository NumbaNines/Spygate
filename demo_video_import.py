import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from spygate.gui.video.video_import import VideoImportWidget


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate Video Import Demo")
        self.setMinimumSize(600, 600)  # Increased height for preview

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
            }
        """
        )

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Add title
        title = QLabel("Video Import Demo")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffffff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Add description
        description = QLabel(
            "Drag and drop video files to import them.\nSupported formats: MP4, AVI, MOV (H.264 and H.265 only)"
        )
        description.setStyleSheet("color: #999999;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        # Add video import widget
        self.video_import = VideoImportWidget()
        layout.addWidget(self.video_import)

        # Add status label
        self.status_label = QLabel("No videos imported")
        self.status_label.setStyleSheet("color: #999999;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Connect signals
        self.video_import.videosImported.connect(self._on_videos_imported)

    def _on_videos_imported(self, videos):
        """Handle imported videos."""
        paths = [path for path, _ in videos]
        self.status_label.setText(f"Imported {len(paths)} video(s):\n" + "\n".join(paths))


def main():
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
