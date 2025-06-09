"""
Spygate play detector module for analyzing game footage.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mss
import numpy as np
from PyQt6.QtCore import QDateTime, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPixmap, QScreen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from .ocr_processor import OCRProcessor
from .situation_analyzer import GameSituation, SituationAnalyzer

# Constants
DEFAULT_WINDOW_SIZE = (1200, 800)
DEFAULT_FPS = 30
FRAME_INTERVAL = int(1000 / DEFAULT_FPS)  # 33ms for ~30fps
MAX_HISTORY_FRAMES = 300  # Keep last ~10 seconds at 30fps


class MonitorSelectDialog(QDialog):
    def __init__(self, monitors: list[dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Monitor")
        layout = QVBoxLayout(self)

        self.combo = QComboBox()
        for i, m in enumerate(monitors):
            self.combo.addItem(f"Monitor {i+1}: {m['width']}x{m['height']}")

        layout.addWidget(QLabel("Select the monitor showing the game:"))
        layout.addWidget(self.combo)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def get_selected_index(self) -> int:
        return self.combo.currentIndex()


class SpygateDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate Play Detector")
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        # Initialize OCR and situation analysis
        self.ocr_processor = OCRProcessor()
        self.situation_analyzer = SituationAnalyzer()
        self.current_situation: Optional[GameSituation] = None

        # Video writer
        self.video_writer: Optional[cv2.VideoWriter] = None

        # Check if model exists
        model_path = "runs/detect/train5/weights/best.pt"
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model not found at {model_path}")
            sys.exit(1)

        # Load the model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            sys.exit(1)

        # Initialize variables
        self.video_source: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame: Optional[np.ndarray] = None
        self.is_paused: bool = True
        self.last_frame: Optional[np.ndarray] = None
        self.recording: bool = False
        self.screen_capture: Optional[mss.mss] = None
        self.selected_monitor: Optional[dict[str, Any]] = None
        self.detection_history: list[dict[str, Any]] = []
        self.last_fps_update: float = time.time()
        self.frame_count: int = 0
        self.visualization_options: dict[str, bool] = {
            "show_boxes": True,
            "show_labels": True,
            "show_confidence": True,
            "highlight_huddle": False,
            "track_motion": False,
        }

        # Performance settings
        self.skip_frames: int = 0  # Number of frames to skip between detections
        self.frame_counter: int = 0
        self.last_detection_results: Optional[list[dict[str, Any]]] = None

        # Create output directories
        self.output_dir = Path("detections")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        # Setup UI
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        # UI elements
        self.file_btn: QPushButton
        self.source_combo: QComboBox
        self.skip_frames_spin: QSpinBox
        self.play_btn: QPushButton
        self.record_btn: QPushButton
        self.conf_combo: QComboBox
        self.save_detections_btn: QPushButton
        self.display_label: QLabel
        self.perf_label: QLabel
        self.down_distance_label: QLabel
        self.time_label: QLabel
        self.score_label: QLabel
        self.situations_label: QLabel
        self.viz_checkboxes: dict[str, QCheckBox]

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Top controls
        top_controls = QHBoxLayout()

        # File controls group
        file_group = QGroupBox("Input")
        file_layout = QHBoxLayout(file_group)

        self.file_btn = QPushButton("Open Video/Image")
        self.file_btn.clicked.connect(self.open_file)
        file_layout.addWidget(self.file_btn)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["File", "Screen Capture"])
        self.source_combo.currentTextChanged.connect(self.source_changed)
        file_layout.addWidget(self.source_combo)

        top_controls.addWidget(file_group)

        # Performance controls group
        perf_group = QGroupBox("Performance")
        perf_layout = QHBoxLayout(perf_group)

        perf_layout.addWidget(QLabel("Skip Frames:"))
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(0, 5)
        self.skip_frames_spin.setValue(0)
        self.skip_frames_spin.valueChanged.connect(self.update_performance)
        perf_layout.addWidget(self.skip_frames_spin)

        top_controls.addWidget(perf_group)

        # Playback controls group
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout(playback_group)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        playback_layout.addWidget(self.record_btn)

        top_controls.addWidget(playback_group)

        # Detection controls group
        detection_group = QGroupBox("Detection")
        detection_layout = QHBoxLayout(detection_group)

        detection_layout.addWidget(QLabel("Confidence:"))
        self.conf_combo = QComboBox()
        self.conf_combo.addItems(["0.25", "0.5", "0.75", "0.9"])
        self.conf_combo.setCurrentText("0.25")
        detection_layout.addWidget(self.conf_combo)

        self.save_detections_btn = QPushButton("Save Detections")
        self.save_detections_btn.clicked.connect(self.save_detections)
        self.save_detections_btn.setEnabled(False)
        detection_layout.addWidget(self.save_detections_btn)

        top_controls.addWidget(detection_group)

        layout.addLayout(top_controls)

        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QHBoxLayout(viz_group)

        self.viz_checkboxes = {}
        for option in [
            "Show Boxes",
            "Show Labels",
            "Show Confidence",
            "Highlight No-Huddle",
            "Track Motion",
        ]:
            cb = QCheckBox(option)
            cb.setChecked(
                option.lower().replace("-", "_").replace(" ", "_") in self.visualization_options
            )
            cb.stateChanged.connect(self.update_visualization_options)
            self.viz_checkboxes[option] = cb
            viz_layout.addWidget(cb)

        layout.addWidget(viz_group)

        # Display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        layout.addWidget(self.display_label)

        # Info area
        info_layout = QHBoxLayout()

        # Detection info
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.info_label)

        # FPS and performance info
        self.perf_label = QLabel()
        self.perf_label.setStyleSheet("font-family: monospace;")
        info_layout.addWidget(self.perf_label)

        layout.addLayout(info_layout)

        # Add enhanced situation display
        situation_group = QGroupBox("Game Situation Analysis")
        situation_layout = QVBoxLayout(situation_group)

        # Basic game state
        basic_state_layout = QHBoxLayout()
        self.down_distance_label = QLabel("Down & Distance: --")
        self.time_label = QLabel("Time: --:--")
        self.score_label = QLabel("Score: 0-0")
        basic_state_layout.addWidget(self.down_distance_label)
        basic_state_layout.addWidget(self.time_label)
        basic_state_layout.addWidget(self.score_label)
        situation_layout.addLayout(basic_state_layout)

        # Specific situations
        self.situations_label = QLabel("No specific situations detected")
        self.situations_label.setWordWrap(True)
        situation_layout.addWidget(self.situations_label)

        # Confidence indicator
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Detection Confidence:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        confidence_layout.addWidget(self.confidence_bar)
        situation_layout.addLayout(confidence_layout)

        layout.addWidget(situation_group)

        # Status bar
        self.statusBar().showMessage("Ready - Open a video or image file to start")

    def update_performance(self) -> None:
        """Update performance settings."""
        self.skip_frames = self.skip_frames_spin.value()
        self.frame_counter = 0

    def update_visualization_options(self) -> None:
        """Update visualization options based on checkbox states."""
        for option, checkbox in self.viz_checkboxes.items():
            self.visualization_options[option.lower().replace(" ", "_")] = checkbox.isChecked()

    def toggle_recording(self) -> None:
        """Toggle video recording state."""
        if not self.recording:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            video_path = str(self.output_dir / "videos" / f"recording_{timestamp}.mp4")
            
            # Get frame dimensions
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
                self.video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    DEFAULT_FPS,
                    (width, height)
                )
                self.recording = True
                self.record_btn.setText("Stop Recording")
            else:
                QMessageBox.warning(self, "Warning", "No frame available to start recording")
        else:
            if hasattr(self, "video_writer") and self.video_writer is not None:
                self.video_writer.release()
            self.recording = False
            self.record_btn.setText("Start Recording")

    def save_detections(self) -> None:
        """Save detection history to CSV and JSON files."""
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        
        # Save to CSV
        csv_path = self.output_dir / "data" / f"detections_{timestamp}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "class", "confidence", "bbox"])
            for det in self.detection_history:
                writer.writerow([det["frame"], det["class"], det["confidence"], det["bbox"]])

        # Save to JSON
        json_path = self.output_dir / "data" / f"detections_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.detection_history, f, indent=2)

        self.detection_history = []  # Clear history after saving

    def setup_screen_capture(self) -> None:
        """Set up screen capture using mss."""
        with mss.mss() as sct:
            monitors = sct.monitors[1:]  # Skip first monitor (entire virtual screen)
            if not monitors:
                QMessageBox.critical(self, "Error", "No monitors detected")
                return

            # Show monitor selection dialog
            dialog = MonitorSelectDialog(monitors, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                selected_idx = dialog.get_selected_index()
                self.selected_monitor = monitors[selected_idx]
                self.screen_capture = mss.mss()
                self.play_btn.setEnabled(True)
                self.record_btn.setEnabled(True)
            else:
                self.source_combo.setCurrentText("File")

    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture the selected monitor screen.

        Returns:
            Captured frame as numpy array or None if capture fails
        """
        try:
            if self.screen_capture and self.selected_monitor:
                screenshot = self.screen_capture.grab(self.selected_monitor)
                return np.array(screenshot)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Screen capture failed: {str(e)}")
        return None

    def source_changed(self, source: str) -> None:
        """Handle source selection change.

        Args:
            source: Selected source type
        """
        if source == "Screen Capture":
            self.setup_screen_capture()
        else:
            if hasattr(self, "screen_capture") and self.screen_capture is not None:
                self.screen_capture.close()
            self.screen_capture = None
            self.selected_monitor = None
            self.play_btn.setEnabled(False)
            self.record_btn.setEnabled(False)

    def toggle_playback(self) -> None:
        """Toggle video playback state."""
        if self.is_paused:
            self.play_btn.setText("Pause")
            self.timer.start(FRAME_INTERVAL)  # ~30 fps
            self.is_paused = False
            self.last_fps_update = time.time()
            self.frame_count = 0
        else:
            self.play_btn.setText("Play")
            self.timer.stop()
            self.is_paused = True

    def update_frame(self) -> None:
        """Process and display the next video frame."""
        frame = None

        # Get frame from appropriate source
        if self.video_source == "screen":
            frame = self.capture_screen()
        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

        if frame is None:
            return

        # Store original frame
        self.current_frame = frame.copy()

        # Skip frames for performance if needed
        self.frame_counter += 1
        if self.frame_counter <= self.skip_frames:
            return

        # Reset frame counter
        self.frame_counter = 0

        # Run detection
        results = self.model(frame, conf=float(self.conf_combo.currentText()))[0]
        detections = []

        # Process detections
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_name = results.names[int(class_id)]
            detections.append({
                "frame": len(self.detection_history),
                "class": class_name,
                "confidence": score,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        # Store detection history
        self.detection_history.extend(detections)
        if len(self.detection_history) > MAX_HISTORY_FRAMES:
            self.detection_history = self.detection_history[-MAX_HISTORY_FRAMES:]

        # Process OCR and analyze situation
        ocr_results = self.ocr_processor.process_hud(frame, detections)
        self.current_situation = self.situation_analyzer.analyze_frame(ocr_results, detections)

        # Visualize detections
        self.last_frame = frame.copy()
        if self.visualization_options["show_boxes"]:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                if self.visualization_options["show_labels"]:
                    label = det["class"]
                    if self.visualization_options["show_confidence"]:
                        label += f" {det['confidence']:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Record frame if needed
        if self.recording and hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert frame for display
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit display area while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.display_label.setPixmap(scaled_pixmap)

        # Update FPS counter
        self.frame_count += 1
        elapsed = time.time() - self.last_fps_update
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.perf_label.setText(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_update = time.time()

        # Update situation display
        if self.current_situation:
            # Update basic game state
            if self.current_situation.down and self.current_situation.distance:
                self.down_distance_label.setText(
                    f"Down & Distance: {self.current_situation.down} & {self.current_situation.distance}"
                )

            if self.current_situation.time_remaining:
                self.time_label.setText(f"Time: {self.current_situation.time_remaining}")

            if (
                self.current_situation.score_home is not None
                and self.current_situation.score_away is not None
            ):
                self.score_label.setText(
                    f"Score: {self.current_situation.score_home}-{self.current_situation.score_away}"
                )

            # Update specific situations
            situation_desc = self.situation_analyzer.get_situation_description(
                self.current_situation
            )
            if situation_desc:
                self.situations_label.setText(situation_desc)
            else:
                self.situations_label.setText("No specific situations detected")

            # Update confidence bar
            self.confidence_bar.setValue(int(self.current_situation.confidence * 100))

        # Enable save button if we have detections
        self.save_detections_btn.setEnabled(len(self.detection_history) > 0)

        # Update status
        self.statusBar().showMessage(f"Processing: {len(detections)} objects detected")

    def open_file(self) -> None:
        """Open a video or image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video/Image",
            "",
            "Video/Image Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png);;All Files (*.*)"
        )
        if file_path:
            self.video_source = "file"
            self.setup_video_source(file_path)

    def setup_video_source(self, file_path: str) -> None:
        """Set up video source from file.

        Args:
            file_path: Path to video/image file
        """
        # Close existing video capture if any
        if self.cap is not None:
            self.cap.release()

        # Check if image or video
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            self.current_frame = cv2.imread(file_path)
            if self.current_frame is None:
                QMessageBox.critical(self, "Error", "Failed to load image")
                return
            self.play_btn.setEnabled(False)
        else:
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video file")
                return
            self.play_btn.setEnabled(True)

        self.record_btn.setEnabled(True)
        self.save_detections_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpygateDetector()
    window.show()
    sys.exit(app.exec())
