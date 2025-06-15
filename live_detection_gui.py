#!/usr/bin/env python3
"""GUI for SpygateAI Live Screen Detection with real-time controls and statistics."""

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSlider,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("❌ PyQt6 not available. Install with: pip install PyQt6")

try:
    import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class ScreenCapture:
    """Screen capture class with multiple backends."""

    def __init__(self, monitor=1, region=None):
        self.monitor = monitor
        self.region = region
        self.method = None
        self.sct = None

        if MSS_AVAILABLE:
            try:
                self.sct = mss.mss()
                self.method = "mss"
            except Exception:
                pass

        if self.method is None and PYAUTOGUI_AVAILABLE:
            try:
                test_img = pyautogui.screenshot()
                self.method = "pyautogui"
            except Exception:
                pass

        if self.method is None:
            raise RuntimeError("No screen capture method available!")

    def get_monitors(self):
        """Get available monitors."""
        if self.method == "mss" and self.sct:
            return self.sct.monitors
        else:
            return [{"top": 0, "left": 0, "width": 1920, "height": 1080}]

    def capture(self):
        """Capture screen and return as OpenCV image."""
        if self.method == "mss" and self.sct:
            monitors = self.sct.monitors
            if self.monitor < len(monitors):
                monitor = monitors[self.monitor]
                if self.region:
                    monitor = {
                        "top": self.region[1],
                        "left": self.region[0],
                        "width": self.region[2],
                        "height": self.region[3],
                    }

                sct_img = self.sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img

        elif self.method == "pyautogui":
            if self.region:
                img = pyautogui.screenshot(region=self.region)
            else:
                img = pyautogui.screenshot()

            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        return None


class DetectionThread(QThread):
    """Background thread for running detection."""

    frame_ready = pyqtSignal(np.ndarray, dict, dict)  # frame, detections, stats
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.running = False
        self.capture = None

        # Detection settings
        self.conf_threshold = 0.15
        self.monitor = 1
        self.region = None
        self.enabled_classes = {
            "hud": True,
            "possession_triangle_area": True,
            "territory_triangle_area": True,
            "preplay_indicator": True,
            "play_call_screen": True,
            "down_distance_area": True,
            "game_clock_area": True,
            "play_clock_area": True,
        }

        # Statistics
        self.detection_stats = {}
        self.frame_count = 0
        self.fps_queue = deque(maxlen=30)

        # Video recording
        self.video_writer = None
        self.save_video = False
        self.video_path = None

    def load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            self.status_update.emit(f"✅ Model loaded: {self.model_path}")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {e}")
            return False

    def setup_capture(self):
        """Setup screen capture."""
        try:
            self.capture = ScreenCapture(monitor=self.monitor, region=self.region)
            self.status_update.emit("✅ Screen capture initialized")
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to setup capture: {e}")
            return False

    def start_recording(self, video_path: str):
        """Start video recording."""
        self.save_video = True
        self.video_path = video_path
        self.status_update.emit(f"📹 Recording to: {video_path}")

    def stop_recording(self):
        """Stop video recording."""
        self.save_video = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.status_update.emit("⏹️ Recording stopped")

    def reset_stats(self):
        """Reset detection statistics."""
        self.detection_stats = {name: 0 for name in self.enabled_classes.keys()}
        self.frame_count = 0
        self.fps_queue.clear()
        self.status_update.emit("🔄 Statistics reset")

    def run(self):
        """Main detection loop."""
        if not self.load_model() or not self.setup_capture():
            return

        self.reset_stats()
        self.running = True

        class_names = [
            "hud",
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ]

        while self.running:
            try:
                start_time = time.time()

                # Capture frame
                frame = self.capture.capture()
                if frame is None:
                    continue

                self.frame_count += 1

                # Setup video writer if needed
                if self.save_video and self.video_writer is None and self.video_path:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.video_writer = cv2.VideoWriter(
                        self.video_path, fourcc, 15.0, (width, height)
                    )

                # Run detection
                detection_start = time.time()
                results = self.model(frame, conf=self.conf_threshold, iou=0.45, verbose=False)
                detection_time = time.time() - detection_start

                # Process detections
                frame_detections = []
                detections_this_frame = {}

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for j in range(len(boxes)):
                            x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy().astype(int)
                            cls_id = int(boxes.cls[j].item())
                            conf = boxes.conf[j].item()

                            if cls_id < len(class_names):
                                class_name = class_names[cls_id]

                                # Only process enabled classes
                                if self.enabled_classes.get(class_name, True):
                                    frame_detections.append(
                                        {
                                            "bbox": (x1, y1, x2, y2),
                                            "class": class_name,
                                            "confidence": conf,
                                            "class_id": cls_id,
                                        }
                                    )

                                    # Update stats
                                    if class_name not in self.detection_stats:
                                        self.detection_stats[class_name] = 0
                                    self.detection_stats[class_name] += 1

                                    if class_name not in detections_this_frame:
                                        detections_this_frame[class_name] = []
                                    detections_this_frame[class_name].append(conf)

                # Calculate FPS
                frame_time = time.time() - start_time
                self.fps_queue.append(frame_time)
                avg_fps = len(self.fps_queue) / sum(self.fps_queue) if self.fps_queue else 0

                # Create stats dict
                stats = {
                    "fps": avg_fps,
                    "detection_time": detection_time * 1000,
                    "frame_count": self.frame_count,
                    "total_detections": sum(self.detection_stats.values()),
                    "detection_stats": self.detection_stats.copy(),
                    "detections_this_frame": detections_this_frame,
                }

                # Save frame if recording
                if self.video_writer:
                    self.video_writer.write(frame)

                # Emit results
                self.frame_ready.emit(frame, {"detections": frame_detections}, stats)

                # Small delay to prevent overwhelming
                time.sleep(0.01)

            except Exception as e:
                self.error_occurred.emit(f"Detection error: {e}")
                break

        # Cleanup
        if self.video_writer:
            self.video_writer.release()

    def stop(self):
        """Stop the detection thread."""
        self.running = False
        self.quit()
        self.wait()


class LiveDetectionGUI(QMainWindow):
    """Main GUI window for live detection."""

    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.model_path = None
        self.recording = False
        self.save_path = None

        # Initialize UI first so status_label exists
        self.init_ui()

        # Then find the model
        self.find_model()

        # Initialize timer for GUI updates
        from PyQt6.QtCore import QTimer

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)

    def find_model(self):
        """Find the trained model automatically."""
        # First check for your new 8-class model (correct path)
        new_model_path = Path(
            "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
        )
        if new_model_path.exists():
            self.model_path = str(new_model_path)
            self.status_label.setText(f"✅ Found 8-class model (44.9% mAP50): {self.model_path}")
            self.start_button.setEnabled(True)
            return

        # Fallback to old detection paths
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            model_dirs = list(runs_dir.glob("spygate_hud_detection_fast*"))
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                model_path = latest_dir / "weights" / "best.pt"
                if model_path.exists():
                    self.model_path = str(model_path)
                    self.status_label.setText(f"✅ Found model: {self.model_path}")
                    self.start_button.setEnabled(True)
                    return

        # Check for any other 8-class models
        eight_class_dir = Path("hud_region_training/hud_region_training_8class")
        if eight_class_dir.exists():
            for run_dir in eight_class_dir.glob("runs/*/weights/best.pt"):
                if run_dir.exists():
                    self.model_path = str(run_dir)
                    self.status_label.setText(f"✅ Found 8-class model: {self.model_path}")
                    self.start_button.setEnabled(True)
                    return

        self.status_label.setText("❌ No trained model found")
        self.start_button.setEnabled(False)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🏈 SpygateAI Live Detection GUI")
        self.setGeometry(100, 100, 1400, 900)

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QGroupBox { font-weight: bold; border: 2px solid #555; border-radius: 5px; margin: 5px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #0078d4; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:disabled { background-color: #555; }
            QSlider::groove:horizontal { height: 6px; background: #555; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0078d4; width: 18px; height: 18px; border-radius: 9px; margin: -6px 0; }
            QCheckBox::indicator:checked { background-color: #0078d4; }
            QComboBox { padding: 5px; border: 1px solid #555; border-radius: 3px; }
            QTextEdit { border: 1px solid #555; border-radius: 3px; }
            QLabel { color: #ffffff; }
        """
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Status bar (create early so it's available for other methods)
        self.status_label = QLabel("🔄 Initializing...")
        self.statusBar().addWidget(self.status_label)

        # Left panel - controls
        self.create_control_panel(layout)

        # Right panel - display and stats
        self.create_display_panel(layout)

    def create_control_panel(self, parent_layout):
        """Create the control panel."""
        control_panel = QVBoxLayout()
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setMaximumWidth(350)
        parent_layout.addWidget(control_widget)

        # Model and capture controls
        model_group = QGroupBox("🎯 Detection Control")
        model_layout = QVBoxLayout(model_group)

        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("▶️ Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setEnabled(False)

        self.stop_button = QPushButton("⏹️ Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        model_layout.addLayout(button_layout)

        # Load model button
        self.load_model_button = QPushButton("📁 Load Custom Model")
        self.load_model_button.clicked.connect(self.load_custom_model)
        model_layout.addWidget(self.load_model_button)

        control_panel.addWidget(model_group)

        # Detection settings
        settings_group = QGroupBox("⚙️ Detection Settings")
        settings_layout = QGridLayout(settings_group)

        # Confidence threshold
        settings_layout.addWidget(QLabel("Confidence:"), 0, 0)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 50)
        self.conf_slider.setValue(15)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        settings_layout.addWidget(self.conf_slider, 0, 1)

        self.conf_label = QLabel("0.15")
        settings_layout.addWidget(self.conf_label, 0, 2)

        # Monitor selection
        settings_layout.addWidget(QLabel("Monitor:"), 1, 0)
        self.monitor_combo = QComboBox()
        self.populate_monitor_list()
        settings_layout.addWidget(self.monitor_combo, 1, 1, 1, 2)

        # Monitor refresh button
        refresh_monitor_btn = QPushButton("🔄")
        refresh_monitor_btn.setMaximumWidth(30)
        refresh_monitor_btn.setToolTip("Refresh monitor list")
        refresh_monitor_btn.clicked.connect(self.populate_monitor_list)
        settings_layout.addWidget(refresh_monitor_btn, 1, 3)

        # Monitor preview button
        preview_monitor_btn = QPushButton("👁️")
        preview_monitor_btn.setMaximumWidth(30)
        preview_monitor_btn.setToolTip("Preview selected monitor")
        preview_monitor_btn.clicked.connect(self.preview_monitor)
        settings_layout.addWidget(preview_monitor_btn, 1, 4)

        control_panel.addWidget(settings_group)

        # Class toggles
        classes_group = QGroupBox("🎮 Detection Classes")
        classes_layout = QVBoxLayout(classes_group)

        self.class_checkboxes = {}
        # Updated for 8-class model
        class_names = [
            "hud",
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ]
        class_colors = ["🔴", "🟢", "🔵", "🟡", "🟣", "🟦", "🟧", "⚪"]

        for name, color in zip(class_names, class_colors):
            checkbox = QCheckBox(f"{color} {name.replace('_', ' ').title()}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, n=name: self.toggle_class(n, state))
            self.class_checkboxes[name] = checkbox
            classes_layout.addWidget(checkbox)

        control_panel.addWidget(classes_group)

        # Recording controls
        recording_group = QGroupBox("📹 Recording")
        recording_layout = QVBoxLayout(recording_group)

        self.record_button = QPushButton("🔴 Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        recording_layout.addWidget(self.record_button)

        self.save_path_label = QLabel("No file selected")
        recording_layout.addWidget(self.save_path_label)

        choose_path_button = QPushButton("📁 Choose Save Location")
        choose_path_button.clicked.connect(self.choose_save_path)
        recording_layout.addWidget(choose_path_button)

        control_panel.addWidget(recording_group)

        # Reset button
        reset_button = QPushButton("🔄 Reset Statistics")
        reset_button.clicked.connect(self.reset_stats)
        control_panel.addWidget(reset_button)

        control_panel.addStretch()

    def create_display_panel(self, parent_layout):
        """Create the display panel."""
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        parent_layout.addWidget(display_widget)

        # Tabs for different views
        tabs = QTabWidget()
        display_layout.addWidget(tabs)

        # Live view tab
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)

        # Video display
        self.video_label = QLabel("📺 Live Detection Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("border: 2px solid #555; background-color: #1a1a1a;")
        live_layout.addWidget(self.video_label)

        # Performance info
        perf_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        self.detection_time_label = QLabel("Detection: --ms")
        self.frame_count_label = QLabel("Frames: 0")

        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.detection_time_label)
        perf_layout.addWidget(self.frame_count_label)
        perf_layout.addStretch()

        live_layout.addLayout(perf_layout)
        tabs.addTab(live_tab, "📺 Live View")

        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        # Detection counts table (8 classes)
        self.stats_table = QTableWidget(8, 3)
        self.stats_table.setHorizontalHeaderLabels(["Class", "Count", "Rate %"])
        self.stats_table.setMaximumHeight(250)
        stats_layout.addWidget(self.stats_table)

        # Live log
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        stats_layout.addWidget(QLabel("📋 Detection Log:"))
        stats_layout.addWidget(self.log_text)

        tabs.addTab(stats_tab, "📊 Statistics")

    def populate_monitor_list(self):
        """Populate the monitor selection combo box with available monitors."""
        self.monitor_combo.clear()

        try:
            if MSS_AVAILABLE:
                import mss

                with mss.mss() as sct:
                    monitors = sct.monitors[1:]  # Skip index 0 (all monitors combined)

                    for i, monitor in enumerate(monitors, 1):
                        monitor_text = f"Monitor {i}: {monitor['width']}x{monitor['height']}"
                        if monitor["left"] != 0 or monitor["top"] != 0:
                            monitor_text += f" at ({monitor['left']}, {monitor['top']})"
                        self.monitor_combo.addItem(monitor_text)

                    if not monitors:
                        self.monitor_combo.addItem("No monitors detected")

                    self.status_label.setText(f"✅ Found {len(monitors)} monitor(s)")
            else:
                self.monitor_combo.addItem("Monitor 1 (Default)")
                self.status_label.setText("⚠️ MSS not available, using default monitor")

        except Exception as e:
            self.monitor_combo.addItem("Monitor 1 (Fallback)")
            self.status_label.setText(f"❌ Error detecting monitors: {e}")

    def preview_monitor(self):
        """Show a preview of the selected monitor."""
        if not MSS_AVAILABLE:
            QMessageBox.warning(self, "Error", "MSS not available for monitor preview!")
            return

        monitor_num = self.monitor_combo.currentIndex() + 1

        try:
            capture = ScreenCapture(monitor=monitor_num)

            # Create preview dialog
            preview_dialog = QDialog(self)
            preview_dialog.setWindowTitle(f"Monitor {monitor_num} Preview")
            preview_dialog.setModal(True)
            preview_dialog.resize(800, 600)

            layout = QVBoxLayout(preview_dialog)

            # Preview label
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_label.setStyleSheet("border: 2px solid #555; background-color: #1a1a1a;")
            layout.addWidget(preview_label)

            # Info label
            info_label = QLabel(f"Previewing Monitor {monitor_num}")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(info_label)

            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(preview_dialog.close)
            layout.addWidget(close_btn)

            # Capture and display frame
            frame = capture.capture()
            if frame is not None:
                # Resize for preview
                height, width = frame.shape[:2]
                if width > 800 or height > 600:
                    scale = min(800 / width, 600 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Convert to QPixmap
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)

                preview_label.setPixmap(pixmap)
                preview_label.setFixedSize(pixmap.size())

                self.status_label.setText(f"✅ Showing preview of Monitor {monitor_num}")
            else:
                info_label.setText(f"❌ Failed to capture Monitor {monitor_num}")
                self.status_label.setText(f"❌ Failed to capture Monitor {monitor_num}")

            preview_dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to preview monitor: {e}")
            self.status_label.setText(f"❌ Preview error: {e}")

    def update_confidence(self, value):
        """Update confidence threshold."""
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        if self.detection_thread:
            self.detection_thread.conf_threshold = conf

    def toggle_class(self, class_name, state):
        """Toggle detection class."""
        enabled = state == Qt.CheckState.Checked.value
        if self.detection_thread:
            self.detection_thread.enabled_classes[class_name] = enabled

    def load_custom_model(self):
        """Load a custom model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "Model files (*.pt *.onnx);;All files (*)"
        )
        if file_path:
            self.model_path = file_path
            self.status_label.setText(f"✅ Custom model loaded: {file_path}")
            self.start_button.setEnabled(True)

    def choose_save_path(self):
        """Choose video save location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Video As", "detection_recording.mp4", "Video files (*.mp4);;All files (*)"
        )
        if file_path:
            self.save_path = file_path
            self.save_path_label.setText(f"📁 {Path(file_path).name}")

    def start_detection(self):
        """Start detection."""
        if not self.model_path:
            QMessageBox.warning(self, "Error", "No model loaded!")
            return

        self.detection_thread = DetectionThread(self.model_path)
        self.detection_thread.frame_ready.connect(self.on_frame_ready)
        self.detection_thread.status_update.connect(self.on_status_update)
        self.detection_thread.error_occurred.connect(self.on_error)

        # Apply settings
        self.detection_thread.conf_threshold = self.conf_slider.value() / 100.0
        self.detection_thread.monitor = self.monitor_combo.currentIndex() + 1

        for class_name, checkbox in self.class_checkboxes.items():
            self.detection_thread.enabled_classes[class_name] = checkbox.isChecked()

        self.detection_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)

        self.timer.start(33)  # ~30 FPS GUI updates

    def stop_detection(self):
        """Stop detection."""
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None

        self.timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)

        if self.recording:
            self.toggle_recording()

    def toggle_recording(self):
        """Toggle video recording."""
        if not self.recording:
            if not self.save_path:
                QMessageBox.warning(self, "Error", "Please choose a save location first!")
                return

            if self.detection_thread:
                self.detection_thread.start_recording(self.save_path)
                self.recording = True
                self.record_button.setText("⏹️ Stop Recording")
        else:
            if self.detection_thread:
                self.detection_thread.stop_recording()
            self.recording = False
            self.record_button.setText("🔴 Start Recording")

    def reset_stats(self):
        """Reset statistics."""
        if self.detection_thread:
            self.detection_thread.reset_stats()
        self.log_text.clear()

    def on_frame_ready(self, frame, detections, stats):
        """Handle new frame with detections."""
        # Draw detections on frame
        display_frame = frame.copy()
        # Updated colors for 8 classes
        colors = [
            (0, 0, 255),  # Red - hud
            (0, 255, 0),  # Green - possession_triangle_area
            (255, 0, 0),  # Blue - territory_triangle_area
            (0, 255, 255),  # Yellow - preplay_indicator
            (255, 0, 255),  # Magenta - play_call_screen
            (255, 255, 0),  # Cyan - down_distance_area
            (255, 165, 0),  # Orange - game_clock_area
            (255, 255, 255),  # White - play_clock_area
        ]

        for det in detections["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            # Safety check for class_id to prevent crashes
            class_id = det["class_id"]
            if class_id >= len(colors):
                class_id = 0  # Default to first color if unexpected class_id
            color = colors[class_id]

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det['class']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                display_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 5, y1),
                color,
                -1,
            )
            cv2.putText(
                display_frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Convert to Qt format and display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled_pixmap)

        # Update performance labels
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        self.detection_time_label.setText(f"Detection: {stats['detection_time']:.1f}ms")
        self.frame_count_label.setText(f"Frames: {stats['frame_count']}")

        # Update statistics table
        self.update_stats_table(stats["detection_stats"], stats["frame_count"])

        # Add to log if new detections
        if stats["detections_this_frame"]:
            log_entry = f"Frame {stats['frame_count']}: "
            for class_name, confs in stats["detections_this_frame"].items():
                log_entry += f"{class_name}({len(confs)}) "
            self.log_text.append(log_entry)

    def update_stats_table(self, detection_stats, frame_count):
        """Update the statistics table."""
        class_names = [
            "hud",
            "possession_triangle_area",
            "territory_triangle_area",
            "preplay_indicator",
            "play_call_screen",
            "down_distance_area",
            "game_clock_area",
            "play_clock_area",
        ]

        for i, class_name in enumerate(class_names):
            count = detection_stats.get(class_name, 0)
            rate = (count / frame_count * 100) if frame_count > 0 else 0

            self.stats_table.setItem(i, 0, QTableWidgetItem(class_name.replace("_", " ").title()))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{rate:.1f}%"))

    def update_display(self):
        """Update display elements."""
        pass  # Main updates handled by signals

    def on_status_update(self, message):
        """Handle status updates."""
        self.status_label.setText(message)

    def on_error(self, error_message):
        """Handle errors."""
        QMessageBox.critical(self, "Error", error_message)
        self.status_label.setText(f"❌ {error_message}")


def main():
    """Main function."""
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is required for the GUI!")
        print("Install with: pip install PyQt6")
        return

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern look

    window = LiveDetectionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
