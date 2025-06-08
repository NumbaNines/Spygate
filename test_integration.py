import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.components.annotation_tool import AnnotationTool
from src.ui.components.video_timeline import VideoTimeline
from src.ui.models.annotation import Annotation


class IntegrationDemo(QMainWindow):
    """Integration demo showing all components working together."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate Integration Demo")
        self.setMinimumSize(1024, 768)

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
                margin: 10px;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """
        )

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Instructions label
        instructions = QLabel(
            "Demo Instructions:\n"
            "1. Space: Play/Pause\n"
            "2. Left/Right: Seek 5 seconds\n"
            "3. Up/Down: Seek 30 seconds\n"
            "4. Ctrl+A: Add annotation\n"
            "5. Ctrl+[/]: Change speed\n"
            "All controls are accessible via keyboard and screen reader."
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Check for test video
        test_video_path = os.path.join("test_videos", "h264_720p.mp4")
        if not os.path.exists(test_video_path):
            QMessageBox.warning(
                self,
                "Video Not Found",
                f"Test video not found at {test_video_path}. Please ensure the video exists.",
            )
            sys.exit(1)

        # Create video timeline
        self.timeline = VideoTimeline(video_path=test_video_path, player_name="Test Player")
        layout.addWidget(self.timeline)

        # Add test controls
        controls_layout = QVBoxLayout()

        # Add test annotation button
        add_test_annotation = QPushButton("Add Test Annotation")
        add_test_annotation.clicked.connect(self.add_test_annotation)
        add_test_annotation.setAccessibleName("Add test annotation")
        controls_layout.addWidget(add_test_annotation)

        # Add speed test button
        test_speed = QPushButton("Test Speed Changes")
        test_speed.clicked.connect(self.test_speed_changes)
        test_speed.setAccessibleName("Test playback speeds")
        controls_layout.addWidget(test_speed)

        # Add seek test button
        test_seek = QPushButton("Test Seeking")
        test_seek.clicked.connect(self.test_seeking)
        test_seek.setAccessibleName("Test video seeking")
        controls_layout.addWidget(test_seek)

        layout.addLayout(controls_layout)

        # Status label
        self.status_label = QLabel("Ready for testing")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Set up test timer
        self.test_timer = QTimer(self)
        self.test_timer.timeout.connect(self.update_status)
        self.current_test = None

    def add_test_annotation(self):
        """Add a test annotation at the current position."""
        current_time = self.timeline.video_player.position() / 1000
        annotation = Annotation(
            timestamp=current_time,
            text="Test annotation at " + self.timeline.format_time(current_time * 1000),
            duration=5.0,
            color=QColor(255, 165, 0).name(),  # Orange
            player_name=self.timeline.player_name,
        )
        self.timeline.add_annotation(annotation)
        self.status_label.setText(
            f"Added annotation at {self.timeline.format_time(current_time * 1000)}"
        )

    def test_speed_changes(self):
        """Test different playback speeds."""
        if self.current_test == "speed":
            self.test_timer.stop()
            self.timeline.set_playback_speed("1x")
            self.status_label.setText("Speed test completed")
            self.current_test = None
            return

        self.current_test = "speed"
        self.speeds = ["0.5x", "1x", "1.5x", "2x"]
        self.speed_index = 0
        self.test_timer.start(3000)  # Change speed every 3 seconds
        self.status_label.setText("Testing playback speeds...")
        self.timeline.video_player.play()

    def test_seeking(self):
        """Test seeking through the video."""
        if self.current_test == "seek":
            self.test_timer.stop()
            self.status_label.setText("Seek test completed")
            self.current_test = None
            return

        self.current_test = "seek"
        self.seek_positions = [0, 5000, 10000, 15000]  # Positions in milliseconds
        self.seek_index = 0
        self.test_timer.start(2000)  # Seek every 2 seconds
        self.status_label.setText("Testing video seeking...")

    def update_status(self):
        """Update the current test status."""
        if self.current_test == "speed":
            self.speed_index = (self.speed_index + 1) % len(self.speeds)
            speed = self.speeds[self.speed_index]
            self.timeline.set_playback_speed(speed)
            self.status_label.setText(f"Testing speed: {speed}")

            if self.speed_index == 0:
                self.test_speed_changes()  # Stop test after full cycle

        elif self.current_test == "seek":
            self.seek_index = (self.seek_index + 1) % len(self.seek_positions)
            position = self.seek_positions[self.seek_index]
            self.timeline.seek_to_position(position)
            self.status_label.setText(f"Seeking to: {self.timeline.format_time(position)}")

            if self.seek_index == 0:
                self.test_seeking()  # Stop test after full cycle


def main():
    app = QApplication(sys.argv)
    window = IntegrationDemo()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
