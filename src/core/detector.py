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


class MonitorSelectDialog(QDialog):
    def __init__(self, monitors: List[Dict[str, Any]], parent: Optional[QWidget] = None) -> None:
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
        self.setMinimumSize(1200, 800)

        # Initialize OCR and situation analysis
        self.ocr_processor = OCRProcessor()
        self.situation_analyzer = SituationAnalyzer()
        self.current_situation = None

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
        self.video_source = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.is_paused = True
        self.last_frame = None
        self.recording = False
        self.screen_capture = None
        self.selected_monitor = None
        self.detection_history = []
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.visualization_options = {
            "show_boxes": True,
            "show_labels": True,
            "show_confidence": True,
            "highlight_huddle": False,
            "track_motion": False,
        }

        # Performance settings
        self.skip_frames = 0  # Number of frames to skip between detections
        self.frame_counter = 0
        self.last_detection_results = None

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
        self.skip_frames = self.skip_frames_spin.value()
        self.frame_counter = 0

    def update_visualization_options(self) -> None:
        for name, cb in self.viz_checkboxes.items():
            option_key = name.lower().replace("-", "_").replace(" ", "_")
            self.visualization_options[option_key] = cb.isChecked()

    def toggle_recording(self) -> None:
        if not self.recording:
            # Start recording
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            self.video_writer = None  # Will be initialized on first frame
            self.csv_file = open(
                self.output_dir / "data" / f"detections_{timestamp}.csv", "w", newline=""
            )
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                ["timestamp", "class", "confidence", "x", "y", "width", "height"]
            )

            self.recording = True
            self.record_btn.setText("Stop Recording")
            self.statusBar().showMessage("Recording...")
        else:
            # Stop recording
            self.recording = False
            if hasattr(self, "video_writer") and self.video_writer:
                self.video_writer.release()
            if hasattr(self, "csv_file"):
                self.csv_file.close()
            self.record_btn.setText("Start Recording")
            self.statusBar().showMessage("Recording saved")

    def save_detections(self) -> None:
        if not self.detection_history:
            QMessageBox.warning(self, "Warning", "No detections to save")
            return

        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")

        # Save detection data as JSON
        json_path = self.output_dir / "data" / f"detections_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(self.detection_history, f, indent=2)

        # Save current frame with detections
        if self.last_frame is not None:
            img_path = self.output_dir / "images" / f"detection_{timestamp}.png"
            cv2.imwrite(str(img_path), self.last_frame)

        QMessageBox.information(self, "Success", f"Detections saved to {json_path}")
        self.detection_history = []  # Clear history after saving

    def setup_screen_capture(self) -> None:
        try:
            self.screen_capture = mss.mss()
            monitors = self.screen_capture.monitors[1:]  # Skip the "all monitors" monitor

            # Show monitor selection dialog
            dialog = MonitorSelectDialog(monitors, self)
            if dialog.exec():
                selected_idx = dialog.get_selected_index()
                self.selected_monitor = monitors[selected_idx]

                # Add monitor info to status
                monitor_info = f"Monitor {selected_idx + 1}: {self.selected_monitor['width']}x{self.selected_monitor['height']}"
                self.statusBar().showMessage(f"Screen capture ready - {monitor_info}")

                # Enable controls
                self.play_btn.setEnabled(True)
                self.record_btn.setEnabled(True)

                # Start capture immediately
                self.toggle_playback()
            else:
                self.source_combo.setCurrentText("File")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to setup screen capture: {str(e)}")
            self.source_combo.setCurrentText("File")

    def capture_screen(self) -> Optional[np.ndarray]:
        if self.screen_capture and self.selected_monitor:
            try:
                screenshot = self.screen_capture.grab(self.selected_monitor)
                frame = np.array(screenshot)
                # Convert from BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame
            except Exception as e:
                print(f"Screen capture error: {str(e)}")
                return None
        return None

    def source_changed(self, source: str) -> None:
        if source == "Screen Capture":
            try:
                self.setup_screen_capture()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to setup screen capture: {str(e)}")
        else:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.display_label.clear()
            self.play_btn.setEnabled(False)
            self.record_btn.setEnabled(False)

    def toggle_playback(self) -> None:
        if self.source_combo.currentText() == "Screen Capture":
            self.is_paused = not self.is_paused
            self.play_btn.setText("Pause" if not self.is_paused else "Play")

            if not self.is_paused:
                self.timer.start(1)  # Update as fast as possible for screen capture
            else:
                self.timer.stop()
        else:
            if self.cap is None:
                return

            self.is_paused = not self.is_paused
            self.play_btn.setText("Pause" if not self.is_paused else "Play")

            if not self.is_paused:
                self.timer.start(30)  # ~30 fps for video files
            else:
                self.timer.stop()

    def update_frame(self) -> None:
        try:
            start_time = time.time()

            # Get frame from either video or screen capture
            if self.source_combo.currentText() == "Screen Capture":
                frame = self.capture_screen()
                if frame is None:
                    raise Exception("Failed to capture screen")
            else:
                if self.cap is None:
                    return
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str) and os.path.isfile(self.video_source):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            raise Exception("Failed to read frame")
                    else:
                        raise Exception("Failed to capture frame")

            # Store last valid frame
            self.last_frame = frame.copy()

            # Update frame counter for skipping
            self.frame_counter += 1

            # Run detection if needed
            if self.frame_counter > self.skip_frames:
                self.frame_counter = 0
                results = self.model.predict(
                    source=frame, conf=float(self.conf_combo.currentText()), verbose=False
                )
                self.last_detection_results = results[0]

            # Use last detection results if we're skipping frames
            if self.last_detection_results is not None:
                detections = self.last_detection_results.boxes
                classes = ["hud", "gamertag", "preplay", "playcall", "no huddle", "audible"]
                class_counts = {c: 0 for c in classes}

                # Store detections for recording/saving
                frame_detections = []
                for det in detections:
                    cls_id = int(det.cls)
                    if cls_id < len(classes):
                        class_name = classes[cls_id]
                        conf = float(det.conf)
                        x1, y1, x2, y2 = map(int, det.xyxy[0])

                        detection = {
                            "class": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                        frame_detections.append(detection)
                        class_counts[class_name] += 1

                        # Record detection if recording
                        if self.recording:
                            self.csv_writer.writerow(
                                [
                                    QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                                    class_name,
                                    conf,
                                    x1,
                                    y1,
                                    x2 - x1,
                                    y2 - y1,
                                ]
                            )

                self.detection_history.append(frame_detections)

                # Process OCR and analyze situation
                ocr_results = self.ocr_processor.process_hud(frame, frame_detections)
                self.current_situation = self.situation_analyzer.analyze_frame(
                    ocr_results, frame_detections
                )

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

                # Draw results based on visualization options
                if self.visualization_options["show_boxes"]:
                    annotated_frame = self.last_detection_results.plot()
                else:
                    annotated_frame = frame.copy()

                # Additional visualization options
                if self.visualization_options["highlight_huddle"]:
                    # Highlight no-huddle plays in red
                    for det in frame_detections:
                        if det["class"] == "no huddle":
                            x1, y1, x2, y2 = det["bbox"]
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                if self.visualization_options["track_motion"]:
                    # Simple motion tracking (frame difference)
                    if hasattr(self, "prev_frame"):
                        frame_diff = cv2.absdiff(self.prev_frame, frame)
                        motion_mask = cv2.threshold(
                            cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY
                        )[1]
                        annotated_frame = cv2.addWeighted(
                            annotated_frame,
                            1,
                            cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR),
                            0.3,
                            0,
                        )
                    self.prev_frame = frame.copy()

                # Convert to Qt format
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

                # Scale to fit display area while maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.display_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                self.display_label.setPixmap(scaled_pixmap)

                # Save frame if recording
                if self.recording:
                    if self.video_writer is None:
                        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
                        output_path = str(self.output_dir / "videos" / f"recording_{timestamp}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                    self.video_writer.write(annotated_frame)

                # Update detection info
                info_text = "Detections:\n"
                for cls, count in class_counts.items():
                    if count > 0:
                        info_text += f"{cls}: {count}\n"

                self.info_label.setText(info_text)

                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:  # Update FPS every second
                    fps = self.frame_count / (current_time - self.last_fps_update)
                    processing_time = (current_time - start_time) * 1000
                    self.perf_label.setText(f"FPS: {fps:.1f}\nProcessing: {processing_time:.1f}ms")
                    self.frame_count = 0
                    self.last_fps_update = current_time

                # Enable save button if we have detections
                self.save_detections_btn.setEnabled(len(self.detection_history) > 0)

                # Update status
                self.statusBar().showMessage(f"Processing: {len(detections)} objects detected")

        except Exception as e:
            self.timer.stop()
            self.is_paused = True
            self.play_btn.setText("Play")
            QMessageBox.warning(self, "Error", f"Frame processing error: {str(e)}")

    def open_file(self) -> None:
        """Open a video or image file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video/Image",
            "",
            "Video/Image Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png);;All Files (*)",
        )

        if file_name:
            self.video_source = file_name
            self.setup_video_source()

    def setup_video_source(self) -> None:
        """Set up video source for capture."""
        try:
            if isinstance(self.video_source, str):
                # Check if it's an image file
                if self.video_source.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.cap = None
                    frame = cv2.imread(self.video_source)
                    if frame is None:
                        raise Exception("Failed to load image")
                    self.current_frame = frame
                    self.last_frame = frame.copy()
                else:
                    # Assume it's a video file
                    self.cap = cv2.VideoCapture(self.video_source)
                    if not self.cap.isOpened():
                        raise Exception("Failed to open video file")

                # Enable controls
                self.play_btn.setEnabled(True)
                self.record_btn.setEnabled(True)
                self.save_detections_btn.setEnabled(True)

                # Start playback
                self.toggle_playback()

            else:
                raise Exception("Invalid video source")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
            self.cap = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpygateDetector()
    window.show()
    sys.exit(app.exec())
