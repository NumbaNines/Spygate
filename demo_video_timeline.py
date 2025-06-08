import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from src.ui.components.video_timeline import VideoTimeline


class VideoTimelineDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate Video Timeline Demo")
        self.setMinimumSize(800, 600)

        # Set dark theme styles
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 8px;
                background: #2d2d2d;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #1084d8;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #353535;
                border: 1px solid #0078d4;
            }
            QPushButton:pressed {
                background-color: #252525;
            }
        """
        )

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Check for test video
        test_video_path = os.path.join("test_videos", "h264_720p.mp4")
        if not os.path.exists(test_video_path):
            QMessageBox.warning(
                self,
                "Video Not Found",
                f"Test video not found at {test_video_path}. Please ensure the video exists.",
            )
            sys.exit(1)

        # Create video timeline with test video
        self.timeline = VideoTimeline(video_path=test_video_path, player_name="Self", parent=self)

        # Add timeline to main layout
        main_layout.addWidget(self.timeline)


def main():
    app = QApplication(sys.argv)
    window = VideoTimelineDemo()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
