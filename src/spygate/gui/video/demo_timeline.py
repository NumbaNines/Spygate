import sys

from PyQt6.QtCore import QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spygate.video.codec_validator import CodecValidator

from .video_timeline import VideoTimeline


class DemoWindow(QMainWindow):
    """Demo window to showcase video timeline functionality."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Timeline Demo")
        self.setGeometry(100, 100, 800, 200)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create media player and audio output
        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)

        # Create timeline widget
        self.timeline = VideoTimeline(self.media_player)
        layout.addWidget(self.timeline)

        # Add file picker button
        self.open_btn = QPushButton("Open Video File")
        self.open_btn.clicked.connect(self.open_file)
        layout.addWidget(self.open_btn)

        # Load a test video if provided as argument
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            self.media_player.setSource(QUrl.fromLocalFile(video_path))

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if file_path:
            valid, error, metadata = CodecValidator.validate_video(file_path)
            if not valid:
                QMessageBox.critical(
                    self,
                    "Unsupported Video",
                    f"{error}\n\nPlease use H.264 or H.265 encoded MP4/MOV/AVI files.",
                )
                return
            self.media_player.setSource(QUrl.fromLocalFile(file_path))


def main():
    """Run the demo application."""
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
