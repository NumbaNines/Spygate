"""
Demo script for formation analysis and visualization.

This script demonstrates the real-time formation analysis capabilities
with visual feedback and progress reporting.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

# Add parent directory to path to import spygate
sys.path.append(str(Path(__file__).parent.parent.parent))

from spygate.core.hardware import HardwareDetector
from spygate.video.formation_analyzer import FormationAnalyzer, FormationConfig
from spygate.visualization.visualization_manager import (
    VisualizationConfig,
    VisualizationManager,
    VisualizationMode,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoWindow(QMainWindow):
    """Main window for the formation analysis demo."""

    def __init__(self):
        """Initialize the demo window."""
        super().__init__()

        # Setup window
        self.setWindowTitle("Formation Analysis Demo")
        self.setGeometry(100, 100, 1300, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create controls
        controls_layout = QHBoxLayout()

        # Visualization mode selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(
            [
                "Tracking Only",
                "Motion Vectors",
                "Heat Map",
                "Formation Analysis",
                "Full Visualization",
            ]
        )
        self.mode_selector.currentTextChanged.connect(self.change_visualization_mode)
        controls_layout.addWidget(QLabel("Visualization Mode:"))
        controls_layout.addWidget(self.mode_selector)

        # Add controls to main layout
        layout.addLayout(controls_layout)

        # Create video display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Create progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Create status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Initialize hardware detection and optimization
        self.hardware = HardwareDetector()

        # Initialize visualization
        self.viz_config = VisualizationConfig(
            mode=VisualizationMode.FULL,
            show_player_ids=True,
            show_confidence=True,
            show_ball_trajectory=True,
            show_formation_lines=True,
            show_motion_trails=True,
            trail_length=30,
            heat_map_opacity=0.4,
            vector_scale=1.0,
            line_thickness=2,
            font_scale=0.5,
            enable_gpu=self.hardware.supports_gpu_acceleration(),
        )
        self.viz_manager = VisualizationManager(config=self.viz_config)

        # Initialize formation analyzer
        config = FormationConfig(
            temporal_smoothing=True,
            gpu_acceleration=self.hardware.supports_gpu_acceleration(),
            confidence_threshold=0.7,
        )
        self.analyzer = FormationAnalyzer(
            config=config,
            progress_callback=lambda status, progress: self.progress_bar.setValue(
                int(progress * 100)
            ),
        )

        # Setup video capture
        video_path = str(Path(__file__).parent / "test_data" / "test_video.mp4")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        # Get video info
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Setup timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps)  # Convert FPS to milliseconds

        # Initialize state
        self.current_frame = None
        self.current_mode = VisualizationMode.FULL
        self.frame_number = 0

        # Show hardware info
        capabilities = self.hardware.get_capabilities()
        logger.info("Hardware capabilities:")
        for key, value in capabilities.items():
            logger.info(f"  {key}: {value}")

    def change_visualization_mode(self, mode_text):
        """Change the current visualization mode."""
        mode_map = {
            "Tracking Only": VisualizationMode.TRACKING_ONLY,
            "Motion Vectors": VisualizationMode.MOTION_VECTORS,
            "Heat Map": VisualizationMode.HEAT_MAP,
            "Formation Analysis": VisualizationMode.FORMATION,
            "Full Visualization": VisualizationMode.FULL,
        }
        self.current_mode = mode_map[mode_text]
        self.viz_config.mode = self.current_mode
        logger.info(f"Changed visualization mode to: {mode_text}")

    def update_frame(self):
        """Update the current frame and visualization."""
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        # Store original frame
        self.current_frame = frame.copy()

        # Update progress
        progress = (self.frame_number % self.frame_count) / self.frame_count * 100
        self.progress_bar.setValue(int(progress))

        # Extract object positions (simple example using color thresholding)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))  # Red objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get object positions and create tracking data
        tracking_data = {"players": {}}
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:  # Min area threshold
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)

                    tracking_data["players"][str(i)] = {
                        "position": (cx, cy),
                        "bbox": (x, y, x + w, y + h),
                        "confidence": 0.95,
                    }

        # Update visualization
        display_frame = self.viz_manager.update_frame(frame, tracking_data, self.frame_number)

        # Convert frame to Qt format
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

        # Scale to fit window while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Display frame
        self.video_label.setPixmap(scaled_pixmap)

        # Update status
        stats = self.viz_manager.get_performance_stats()
        self.status_label.setText(
            f"Frame: {self.frame_number} | "
            f"FPS: {stats['fps']:.1f} | "
            f"Mode: {self.current_mode.name} | "
            f"Hardware Tier: {self.hardware.tier.name}"
        )

        self.frame_number += 1


def main():
    """Run the formation analysis demo."""
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
