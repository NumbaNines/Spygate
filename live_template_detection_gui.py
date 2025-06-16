#!/usr/bin/env python3
"""Enhanced Live Template Detection GUI for SpygateAI with real-time template matching feedback."""

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Import our optimized template detection system
sys.path.append('src')
from spygate.ml.down_template_detector import DownTemplateDetector, DownDetectionContext
from spygate.ml.template_triangle_detector import TemplateTriangleDetector
from spygate.ml.enhanced_game_analyzer import UI_CLASSES

try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
        QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
        QProgressBar, QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
        QTabWidget, QTextEdit, QVBoxLayout, QWidget
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


class TemplateDetectionThread(QThread):
    """Background thread for running YOLO + Template detection."""

    frame_ready = pyqtSignal(np.ndarray, dict, dict, dict)  # frame, detections, stats, template_results
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    template_detected = pyqtSignal(str, float, str)  # down_text, confidence, quality_mode
    triangle_detected = pyqtSignal(str, str, float, str)  # triangle_type, direction, confidence, area

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.template_detector = None
        self.triangle_detector = None
        self.running = False
        self.capture = None

        # Detection settings
        self.conf_threshold = 0.15
        self.triangle_conf_threshold = 0.45  # Updated to production-ready threshold based on empirical testing
        self.monitor = 1
        self.region = None
        
        # Template detection settings
        self.template_enabled = True
        self.template_quality_mode = "live"
        
        # Statistics
        self.detection_stats = {}
        self.template_stats = {
            "total_attempts": 0,
            "successful_detections": 0,
            "template_matches": 0,
            "ocr_fallbacks": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0
        }
        self.triangle_stats = {
            "possession_attempts": 0,
            "possession_detections": 0,
            "territory_attempts": 0,
            "territory_detections": 0,
            "avg_triangle_confidence": 0.0,
            "avg_triangle_processing_time": 0.0
        }
        self.frame_count = 0
        self.fps_queue = deque(maxlen=30)

    def load_model(self):
        """Load the YOLO model and template detector."""
        try:
            # Load YOLO model
            self.model = YOLO(self.model_path)
            self.status_update.emit(f"✅ YOLO model loaded: {self.model_path}")
            
            # Initialize template detector with expert optimizations
            self.template_detector = DownTemplateDetector(
                quality_mode=self.template_quality_mode,
                debug_output_dir=Path("debug_output")
            )
            self.status_update.emit("✅ Expert-optimized template detector initialized")
            
            # Initialize triangle detector (97.6% accuracy)
            self.triangle_detector = TemplateTriangleDetector(
                debug_output_dir=Path("debug_output")
            )
            # Set initial confidence threshold for live capture
            self.triangle_detector.MIN_MATCH_CONFIDENCE = self.triangle_conf_threshold
            self.status_update.emit(f"✅ Triangle detector initialized with {len(self.triangle_detector.templates)} templates")
            self.status_update.emit(f"✅ Triangle template detector (conf: {self.triangle_conf_threshold:.3f})")
            
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

    def update_triangle_confidence_threshold(self, new_threshold: float):
        """Update triangle detector confidence threshold during runtime."""
        if self.triangle_detector:
            old_threshold = self.triangle_detector.MIN_MATCH_CONFIDENCE
            self.triangle_detector.MIN_MATCH_CONFIDENCE = new_threshold
            self.triangle_conf_threshold = new_threshold
            self.status_update.emit(f"🔺 Triangle threshold updated: {old_threshold:.3f} → {new_threshold:.3f}")

    def run(self):
        """Main detection loop with template matching."""
        if not self.load_model() or not self.setup_capture():
            return

        self.running = True
        self.status_update.emit("🚀 Live template detection started!")

        while self.running:
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture.capture()
                if frame is None:
                    continue

                # Run YOLO detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Process detections
                detections = {}
                template_results = {}
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            # Get class name using UI_CLASSES list
                            class_id = int(box.cls[0])
                            if class_id < len(UI_CLASSES):
                                class_name = UI_CLASSES[class_id]
                            else:
                                class_name = f"unknown_{class_id}"
                            confidence = float(box.conf[0])
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            if class_name not in detections:
                                detections[class_name] = []
                            
                            detections[class_name].append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence
                            })
                            
                            # Update detection stats
                            if class_name not in self.detection_stats:
                                self.detection_stats[class_name] = 0
                            self.detection_stats[class_name] += 1
                            
                            # Template detection for down_distance_area
                            if (class_name == "down_distance_area" and 
                                self.template_enabled and 
                                self.template_detector):
                                
                                template_start = time.time()
                                self.template_stats["total_attempts"] += 1
                                
                                try:
                                    # Expert pre-validation: check if region has content
                                    roi = frame[y1:y2, x1:x2]
                                    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 20:
                                        continue  # Skip tiny or empty regions
                                    
                                    # Check for actual content (not just blank space)
                                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    text_pixels = np.sum(gray_roi < 200)  # Count dark pixels (text)
                                    total_pixels = gray_roi.size
                                    text_ratio = text_pixels / total_pixels if total_pixels > 0 else 0
                                    
                                    if text_ratio < 0.05:  # Less than 5% dark pixels = likely blank
                                        self.status_update.emit(f"🚫 Rejected blank region (text ratio: {text_ratio:.3f})")
                                        continue  # Skip regions with no meaningful content
                                    
                                    # Create detection context
                                    context = DownDetectionContext(
                                        quarter=1,  # Default values for live detection
                                        time_remaining="15:00",
                                        field_position="50",
                                        possession_team="HOME"
                                    )
                                    
                                    # Run template detection
                                    template_match = self.template_detector.detect_down_in_yolo_region(
                                        frame, (x1, y1, x2, y2), context
                                    )
                                    
                                    template_time = time.time() - template_start
                                    self.template_stats["avg_processing_time"] = (
                                        (self.template_stats["avg_processing_time"] * 
                                         (self.template_stats["total_attempts"] - 1) + template_time) /
                                        self.template_stats["total_attempts"]
                                    )
                                    
                                    # Expert validation: check for actual content
                                    if template_match:
                                        if template_match.confidence <= 0.25:
                                            self.status_update.emit(f"🚫 Rejected low confidence match: {template_match.confidence:.3f}")
                                        elif not template_match.down or not str(template_match.down).strip():
                                            self.status_update.emit(f"🚫 Rejected empty down value")
                                        else:
                                            # Valid detection - process it
                                            self.template_stats["successful_detections"] += 1
                                            
                                            # Determine detection method
                                            if template_match.template_name:
                                                self.template_stats["template_matches"] += 1
                                                detection_method = "TEMPLATE"
                                            else:
                                                self.template_stats["ocr_fallbacks"] += 1
                                                detection_method = "OCR"
                                            
                                            # Update average confidence
                                            self.template_stats["avg_confidence"] = (
                                                (self.template_stats["avg_confidence"] * 
                                                 (self.template_stats["successful_detections"] - 1) + 
                                                 template_match.confidence) /
                                                self.template_stats["successful_detections"]
                                            )
                                            
                                            # Store template result
                                            down_text = f"{template_match.down}"
                                            if template_match.distance:
                                                down_text += f" & {template_match.distance}"
                                            
                                            template_results[class_name] = {
                                                'down_text': down_text,
                                                'confidence': template_match.confidence,
                                                'method': detection_method,
                                                'template_name': template_match.template_name,
                                                'processing_time': template_time,
                                                'quality_mode': self.template_detector.quality_mode
                                            }
                                            
                                            # Emit template detection signal
                                            self.template_detected.emit(
                                                down_text,
                                                template_match.confidence,
                                                self.template_detector.quality_mode
                                            )
                                        
                                except Exception as e:
                                    self.status_update.emit(f"⚠️ Template detection error: {e}")
                            
                            # Triangle detection for possession and territory areas
                            if (class_name in ["possession_triangle_area", "territory_triangle_area"] and 
                                self.triangle_detector):
                                
                                self.status_update.emit(f"🔍 Triangle detection for {class_name} (YOLO conf: {confidence:.3f}, Triangle threshold: {self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f})")
                                triangle_start = time.time()
                                
                                try:
                                    # Extract ROI for triangle detection
                                    roi = frame[y1:y2, x1:x2]
                                    
                                    # Determine triangle type
                                    triangle_type = "possession" if class_name == "possession_triangle_area" else "territory"
                                    
                                    # Update triangle attempt stats
                                    if triangle_type == "possession":
                                        self.triangle_stats["possession_attempts"] += 1
                                    else:
                                        self.triangle_stats["territory_attempts"] += 1
                                    
                                    # Run triangle template detection
                                    triangle_matches = self.triangle_detector.detect_triangles_in_roi(roi, triangle_type)
                                    self.status_update.emit(f"🔍 Found {len(triangle_matches)} triangle matches for {triangle_type} (ROI: {roi.shape})")
                                    
                                    # Debug: show confidence of all matches with threshold comparison
                                    if triangle_matches:
                                        confidences = [m.confidence for m in triangle_matches]
                                        above_threshold = [c for c in confidences if c >= self.triangle_detector.MIN_MATCH_CONFIDENCE]
                                        below_threshold = [c for c in confidences if c < self.triangle_detector.MIN_MATCH_CONFIDENCE]
                                        
                                        self.status_update.emit(f"📊 Triangle confidences: {[f'{c:.3f}' for c in confidences[:3]]}")  # Show first 3
                                        if above_threshold:
                                            self.status_update.emit(f"✅ ABOVE threshold ({self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f}): {len(above_threshold)} matches, max: {max(above_threshold):.3f}")
                                        if below_threshold:
                                            self.status_update.emit(f"❌ BELOW threshold ({self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f}): {len(below_threshold)} matches, max: {max(below_threshold):.3f}")
                                    
                                    # Select best triangle
                                    best_triangle = self.triangle_detector.select_best_single_triangles(triangle_matches, triangle_type)
                                    self.status_update.emit(f"🎯 Best triangle result: {best_triangle is not None} (threshold: {self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f})")
                                    
                                    triangle_time = time.time() - triangle_start
                                    
                                    # Update processing time stats
                                    total_attempts = (self.triangle_stats["possession_attempts"] + 
                                                    self.triangle_stats["territory_attempts"])
                                    self.triangle_stats["avg_triangle_processing_time"] = (
                                        (self.triangle_stats["avg_triangle_processing_time"] * 
                                         (total_attempts - 1) + triangle_time) / total_attempts
                                    )
                                    
                                    if best_triangle:
                                        # Update triangle detection stats
                                        if triangle_type == "possession":
                                            self.triangle_stats["possession_detections"] += 1
                                        else:
                                            self.triangle_stats["territory_detections"] += 1
                                        
                                        # Update average confidence
                                        total_detections = (self.triangle_stats["possession_detections"] + 
                                                          self.triangle_stats["territory_detections"])
                                        self.triangle_stats["avg_triangle_confidence"] = (
                                            (self.triangle_stats["avg_triangle_confidence"] * 
                                             (total_detections - 1) + best_triangle['confidence']) / total_detections
                                        )
                                        
                                        self.status_update.emit(f"✅ Triangle detected: {triangle_type} {best_triangle['direction']} (conf: {best_triangle['confidence']:.3f})")
                                        # Store triangle result
                                        template_results[f"{class_name}_triangle"] = {
                                            'direction': best_triangle['direction'],
                                            'confidence': best_triangle['confidence'],
                                            'position': best_triangle['position'],
                                            'template_name': best_triangle['template_name'],
                                            'processing_time': triangle_time,
                                            'triangle_type': triangle_type,
                                            'threshold_used': self.triangle_detector.MIN_MATCH_CONFIDENCE
                                        }
                                        
                                        # Emit triangle detection signal
                                        self.template_detected.emit(
                                            f"{triangle_type.upper()}: {best_triangle['direction']}",
                                            best_triangle['confidence'],
                                            f"triangle_template (threshold: {self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f})"
                                        )
                                    else:
                                        self.status_update.emit(f"❌ No triangle found for {triangle_type} (threshold: {self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f})")
                                        if triangle_matches:
                                            max_conf = max(m.confidence for m in triangle_matches)
                                            self.status_update.emit(f"💡 Highest triangle confidence: {max_conf:.3f} (need ≥{self.triangle_detector.MIN_MATCH_CONFIDENCE:.3f})")
                                        
                                except Exception as e:
                                    self.status_update.emit(f"⚠️ Triangle detection error: {e}")

                # Calculate FPS
                frame_time = time.time() - start_time
                self.fps_queue.append(frame_time)
                self.frame_count += 1

                # Emit frame with all detection data
                self.frame_ready.emit(frame, detections, self.detection_stats, template_results)

                # Small delay to prevent overwhelming the GUI
                time.sleep(0.01)

            except Exception as e:
                self.error_occurred.emit(f"Detection error: {e}")
                time.sleep(0.1)

    def stop(self):
        """Stop the detection thread."""
        self.running = False
        self.wait()


class LiveTemplateDetectionGUI(QMainWindow):
    """Enhanced GUI for live template detection with real-time feedback."""

    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.current_frame = None
        self.current_detections = {}
        self.current_template_results = {}
        
        # Find the best available model
        model_path = self.find_model()
        if model_path:
            self.detection_thread = TemplateDetectionThread(model_path)
            self.detection_thread.frame_ready.connect(self.on_frame_ready)
            self.detection_thread.status_update.connect(self.on_status_update)
            self.detection_thread.error_occurred.connect(self.on_error)
            self.detection_thread.template_detected.connect(self.on_template_detected)
        
        self.init_ui()
        self.setWindowTitle("SpygateAI - Live Template Detection (Expert Optimized)")
        self.setGeometry(100, 100, 1400, 900)

    def find_model(self):
        """Find the best available YOLO model."""
        # Try 8-class model first
        model_paths = [
            "hud_region_training/hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
            "hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
            "runs/detect/spygate_latest/weights/best.pt"
        ]
        
        for path in model_paths:
            if Path(path).exists():
                return path
        
        return None

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Control panels
        self.create_control_panel(left_layout)
        self.create_template_panel(left_layout)
        self.create_stats_panel(left_layout)
        
        # Right panel - Display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.create_display_panel(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def create_control_panel(self, parent_layout):
        """Create the main control panel."""
        group = QGroupBox("🎮 Detection Controls")
        layout = QVBoxLayout(group)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("⏹️ Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # YOLO Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("YOLO Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(15)
        self.conf_slider.valueChanged.connect(self.update_confidence)
        self.conf_label = QLabel("0.15")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        layout.addLayout(conf_layout)
        
        # Triangle confidence threshold
        triangle_conf_layout = QHBoxLayout()
        triangle_conf_layout.addWidget(QLabel("Triangle Confidence:"))
        self.triangle_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.triangle_conf_slider.setRange(5, 95)  # 0.05 to 0.95
        self.triangle_conf_slider.setValue(45)  # 0.45 production-ready default
        self.triangle_conf_slider.valueChanged.connect(self.update_triangle_confidence)
        self.triangle_conf_label = QLabel("0.45")
        triangle_conf_layout.addWidget(self.triangle_conf_slider)
        triangle_conf_layout.addWidget(self.triangle_conf_label)
        layout.addLayout(triangle_conf_layout)
        
        # Monitor selection
        monitor_layout = QHBoxLayout()
        monitor_layout.addWidget(QLabel("Monitor:"))
        self.monitor_combo = QComboBox()
        self.monitor_combo.addItems(["1", "2", "3"])
        monitor_layout.addWidget(self.monitor_combo)
        layout.addLayout(monitor_layout)
        
        # Live status
        self.status_label = QLabel("⏸️ Detection Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        parent_layout.addWidget(group)

    def create_template_panel(self, parent_layout):
        """Create the template detection panel."""
        group = QGroupBox("🔤 Template Detection Status")
        layout = QVBoxLayout(group)
        
        # Template toggle
        self.template_checkbox = QCheckBox("Enable Template Detection")
        self.template_checkbox.setChecked(True)
        self.template_checkbox.stateChanged.connect(self.toggle_template_detection)
        layout.addWidget(self.template_checkbox)
        
        # Quality mode selection
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality Mode:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["live", "high", "medium", "low", "streamer", "auto"])
        self.quality_combo.setCurrentText("live")
        self.quality_combo.currentTextChanged.connect(self.update_quality_mode)
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)
        
        # Template results display
        self.template_result_label = QLabel("🎯 Template Detection\nThreshold: 0.30 | Avg: 0.000\nWaiting for detections...")
        self.template_result_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 12px;")
        layout.addWidget(self.template_result_label)
        
        # Triangle results display - FIXED: Properly create the triangle result label
        self.triangle_result_label = QLabel("🔺 Triangle Detection\nThreshold: 0.45 | Avg: 0.000\nWaiting for detections...")
        self.triangle_result_label.setStyleSheet("background-color: #f8f8f8; padding: 10px; border-radius: 5px; font-size: 11px;")
        layout.addWidget(self.triangle_result_label)
        
        parent_layout.addWidget(group)

    def create_stats_panel(self, parent_layout):
        """Create the statistics panel."""
        group = QGroupBox("📊 Live Statistics")
        layout = QVBoxLayout(group)
        
        # Create tabs for different stats
        tabs = QTabWidget()
        
        # YOLO Detection Stats
        yolo_widget = QWidget()
        yolo_layout = QVBoxLayout(yolo_widget)
        self.yolo_stats_table = QTableWidget(0, 2)
        self.yolo_stats_table.setHorizontalHeaderLabels(["Class", "Count"])
        yolo_layout.addWidget(self.yolo_stats_table)
        tabs.addTab(yolo_widget, "YOLO")
        
        # Template Detection Stats
        template_widget = QWidget()
        template_layout = QVBoxLayout(template_widget)
        self.template_stats_table = QTableWidget(0, 2)
        self.template_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        template_layout.addWidget(self.template_stats_table)
        tabs.addTab(template_widget, "Templates")
        
        layout.addWidget(tabs)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset Stats")
        reset_btn.clicked.connect(self.reset_stats)
        layout.addWidget(reset_btn)
        
        parent_layout.addWidget(group)

    def create_display_panel(self, parent_layout):
        """Create the main display panel."""
        group = QGroupBox("📺 Live Detection Feed")
        layout = QVBoxLayout(group)
        
        # Status bar
        self.status_label = QLabel("Ready to start detection...")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # Main display
        self.display_label = QLabel("Click 'Start Detection' to begin...")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        layout.addWidget(self.display_label)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #333; font-weight: bold;")
        layout.addWidget(self.fps_label)
        
        parent_layout.addWidget(group)

    def update_confidence(self, value):
        """Update YOLO confidence threshold."""
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        if self.detection_thread:
            self.detection_thread.conf_threshold = conf

    def update_triangle_confidence(self, value):
        """Update triangle detection confidence threshold."""
        conf = value / 100.0
        self.triangle_conf_label.setText(f"{conf:.2f}")
        if self.detection_thread:
            self.detection_thread.update_triangle_confidence_threshold(conf)
        
        # Update triangle display immediately to show new threshold
        if hasattr(self, 'triangle_result_label'):
            self.triangle_result_label.setText(f"🔺 Triangle Detection\nThreshold: {conf:.3f} | Avg: 0.000\nWaiting for detections...")
            self.triangle_result_label.setStyleSheet("background-color: #f8f8f8; padding: 10px; border-radius: 5px; font-size: 11px;")

    def toggle_template_detection(self, state):
        """Toggle template detection on/off."""
        enabled = state == 2  # Qt.Checked
        if self.detection_thread:
            self.detection_thread.template_enabled = enabled
        
        status = "✅ Enabled" if enabled else "❌ Disabled"
        if hasattr(self, 'template_result_label'):
            self.template_result_label.setText(f"🎯 Template Detection: {status}")
            self.template_result_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 12px;")

    def update_quality_mode(self, mode):
        """Update template detection quality mode."""
        if self.detection_thread and self.detection_thread.template_detector:
            self.detection_thread.template_detector.quality_mode = mode
            # Update quality mode display
            if hasattr(self, 'template_result_label'):
                self.template_result_label.setText(f"🎯 Template Detection\nQuality Mode: {mode}\nWaiting for detections...")
                self.template_result_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; font-size: 11px;")

    def start_detection(self):
        """Start the detection process."""
        if self.detection_thread and not self.detection_thread.running:
            self.detection_thread.monitor = int(self.monitor_combo.currentText())
            self.detection_thread.start()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def stop_detection(self):
        """Stop the detection process."""
        if self.detection_thread and self.detection_thread.running:
            self.detection_thread.stop()
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def reset_stats(self):
        """Reset all statistics."""
        if self.detection_thread:
            self.detection_thread.detection_stats.clear()
            self.detection_thread.template_stats = {
                "total_attempts": 0,
                "successful_detections": 0,
                "template_matches": 0,
                "ocr_fallbacks": 0,
                "avg_confidence": 0.0,
                "avg_processing_time": 0.0
            }
            self.detection_thread.triangle_stats = {
                "possession_attempts": 0,
                "possession_detections": 0,
                "territory_attempts": 0,
                "territory_detections": 0,
                "avg_triangle_confidence": 0.0,
                "avg_triangle_processing_time": 0.0
            }
            self.detection_thread.frame_count = 0
            self.detection_thread.fps_queue.clear()
        
        self.on_status_update("Statistics reset")

    def on_frame_ready(self, frame, detections, stats, template_results):
        """Handle new frame with detections."""
        self.current_frame = frame
        self.current_detections = detections
        self.current_template_results = template_results
        
        # Update display
        self.update_display()
        self.update_stats_tables(stats)
        
        # Calculate and display FPS
        if self.detection_thread and self.detection_thread.fps_queue:
            avg_frame_time = sum(self.detection_thread.fps_queue) / len(self.detection_thread.fps_queue)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_template_detected(self, down_text, confidence, quality_mode):
        """Handle template detection results with enhanced triangle feedback."""
        # Determine confidence color
        if confidence >= 0.35:
            color = "#006600"  # Green for high confidence
        elif confidence >= 0.25:
            color = "#ff8800"  # Orange for medium confidence  
        else:
            color = "#aa0000"  # Red for low confidence
        
        # Check if this is triangle detection or down detection
        if "triangle_template" in quality_mode:
            # Triangle detection - update triangle display
            threshold_info = quality_mode.split("(threshold: ")[1].rstrip(")")
            
            # Update triangle statistics for average calculation
            if hasattr(self, 'triangle_confidences'):
                self.triangle_confidences.append(confidence)
                if len(self.triangle_confidences) > 10:  # Keep last 10 for rolling average
                    self.triangle_confidences.pop(0)
            else:
                self.triangle_confidences = [confidence]
            
            avg_confidence = sum(self.triangle_confidences) / len(self.triangle_confidences)
            
            # Update triangle result label with current info
            if hasattr(self, 'triangle_result_label'):
                self.triangle_result_label.setText(
                    f"🔺 Triangle Detection\n"
                    f"Threshold: {threshold_info} | Avg: {avg_confidence:.3f}\n"
                    f"Latest: {down_text} ({confidence:.3f})"
                )
                self.triangle_result_label.setStyleSheet(
                    f"background-color: #f8f8f8; padding: 10px; border-radius: 5px; "
                    f"font-size: 11px; color: {color}; font-weight: bold;"
                )
        else:
            # Down template detection - update template display
            # Update template statistics for average calculation
            if hasattr(self, 'template_confidences'):
                self.template_confidences.append(confidence)
                if len(self.template_confidences) > 10:  # Keep last 10 for rolling average
                    self.template_confidences.pop(0)
            else:
                self.template_confidences = [confidence]
            
            avg_confidence = sum(self.template_confidences) / len(self.template_confidences)
            
            # Get current threshold from detection thread
            current_threshold = 0.30  # Default
            if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.template_detector:
                current_threshold = self.detection_thread.template_detector.MIN_MATCH_CONFIDENCE
            
            # Update template detection display
            if hasattr(self, 'template_result_label'):
                self.template_result_label.setText(
                    f"🎯 Template Detection\n"
                    f"Threshold: {current_threshold:.3f} | Avg: {avg_confidence:.3f}\n"
                    f"Latest: {down_text} ({confidence:.3f}) [{quality_mode}]"
                )
                self.template_result_label.setStyleSheet(
                    f"background-color: #f0f0f0; padding: 10px; border-radius: 5px; "
                    f"font-size: 12px; color: {color}; font-weight: bold;"
                )

    def update_display(self):
        """Update the main display with current frame and detections."""
        if self.current_frame is None:
            return
        
        # Create a copy of the frame for drawing
        display_frame = self.current_frame.copy()
        
        # Draw YOLO detections
        colors = {
            "hud": (0, 0, 255),  # Red
            "possession_triangle_area": (0, 255, 128),  # Lime green
            "territory_triangle_area": (255, 128, 0),  # Orange
            "preplay_indicator": (255, 0, 255),  # Magenta
            "play_call_screen": (255, 255, 0),  # Yellow
            "down_distance_area": (0, 255, 255),  # Cyan - IMPORTANT!
            "game_clock_area": (128, 255, 255),  # Light cyan
            "play_clock_area": (255, 128, 255),  # Light magenta
        }
        
        for class_name, detections in self.current_detections.items():
            color = colors.get(class_name, (128, 128, 128))
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw template detection results for down_distance_area
                if (class_name == "down_distance_area" and 
                    class_name in self.current_template_results):
                    
                    template_result = self.current_template_results[class_name]
                    template_text = f"TEMPLATE: {template_result['down_text']}"
                    template_conf = f"Conf: {template_result['confidence']:.3f}"
                    method = f"Method: {template_result['method']}"
                    
                    # Draw template result below the box
                    y_offset = y2 + 20
                    cv2.putText(display_frame, template_text, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, template_conf, (x1, y_offset + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, method, (x1, y_offset + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw triangle detection results for triangle areas
                triangle_key = f"{class_name}_triangle"
                if (class_name in ["possession_triangle_area", "territory_triangle_area"] and 
                    triangle_key in self.current_template_results):
                    
                    triangle_result = self.current_template_results[triangle_key]
                    triangle_text = f"TRIANGLE: {triangle_result['direction'].upper()}"
                    triangle_conf = f"Conf: {triangle_result['confidence']:.3f}"
                    triangle_type = f"Type: {triangle_result['triangle_type']}"
                    
                    # Draw triangle result below the box with different color
                    y_offset = y2 + 20
                    cv2.putText(display_frame, triangle_text, (x1, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)  # Magenta for triangles
                    cv2.putText(display_frame, triangle_conf, (x1, y_offset + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(display_frame, triangle_type, (x1, y_offset + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Convert to Qt format and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale to fit display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.display_label.setPixmap(scaled_pixmap)

    def update_stats_tables(self, detection_stats):
        """Update the statistics tables with current data."""
        # YOLO Detection stats
        self.yolo_stats_table.setRowCount(len(detection_stats))
        
        row = 0
        for class_name, count in detection_stats.items():
            self.yolo_stats_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.yolo_stats_table.setItem(row, 1, QTableWidgetItem(str(count)))
            row += 1
        
        # Template detection stats
        if hasattr(self, 'detection_thread') and self.detection_thread:
            template_stats = self.detection_thread.template_stats
            triangle_stats = self.detection_thread.triangle_stats
            
            # Clear existing items
            self.template_stats_table.setRowCount(0)
            
            # Add template stats
            template_rows = [
                ("Template Attempts", template_stats.get("total_attempts", 0)),
                ("Successful Detections", template_stats.get("successful_detections", 0)),
                ("Template Matches", template_stats.get("template_matches", 0)),
                ("OCR Fallbacks", template_stats.get("ocr_fallbacks", 0)),
                ("Avg Confidence", f"{template_stats.get('avg_confidence', 0):.3f}"),
                ("Avg Processing (ms)", f"{template_stats.get('avg_processing_time', 0)*1000:.1f}"),
                # Triangle stats
                ("Triangle Threshold", f"{self.detection_thread.triangle_conf_threshold:.3f}" if hasattr(self.detection_thread, 'triangle_conf_threshold') else "N/A"),
                ("Possession Attempts", triangle_stats.get("possession_attempts", 0)),
                ("Possession Detections", triangle_stats.get("possession_detections", 0)),
                ("Territory Attempts", triangle_stats.get("territory_attempts", 0)),
                ("Territory Detections", triangle_stats.get("territory_detections", 0)),
                ("Triangle Avg Confidence", f"{triangle_stats.get('avg_triangle_confidence', 0):.3f}"),
                ("Triangle Processing (ms)", f"{triangle_stats.get('avg_triangle_processing_time', 0)*1000:.1f}"),
            ]
            
            self.template_stats_table.setRowCount(len(template_rows))
            for i, (stat_name, value) in enumerate(template_rows):
                self.template_stats_table.setItem(i, 0, QTableWidgetItem(stat_name))
                self.template_stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def on_status_update(self, message):
        """Handle status updates."""
        self.status_label.setText(message)

    def on_error(self, error_message):
        """Handle errors."""
        QMessageBox.critical(self, "Error", error_message)
        self.status_label.setText(f"Error: {error_message}")


def main():
    """Main application entry point."""
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is required. Install with: pip install PyQt6")
        return
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = LiveTemplateDetectionGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 