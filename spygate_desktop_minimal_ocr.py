#!/usr/bin/env python3
"""
Minimal SpygateAI Desktop App - OCR ONLY
Shows ONLY what OCR detects without any clip logic
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


class OCROnlyWorker(QThread):
    """Worker thread that ONLY does OCR analysis"""

    ocr_result = pyqtSignal(dict)
    frame_captured = pyqtSignal(object, object)  # frame, game_state

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.analyzer = EnhancedGameAnalyzer()
        self.running = False
        self.capture_next_frame = False

    def run(self):
        """Run OCR-only analysis"""
        self.running = True
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze every 60 frames (1 second at 60fps)
            if frame_count % 60 == 0:
                timestamp = frame_count / fps

                try:
                    game_state = self.analyzer.analyze_frame(frame, current_time=timestamp)

                    # Check if we need to capture this frame for debugging
                    if self.capture_next_frame and game_state:
                        self.frame_captured.emit(frame.copy(), game_state)
                        self.capture_next_frame = False

                    if game_state:
                        result = {
                            "timestamp": timestamp,
                            "frame": frame_count,
                            "down": getattr(game_state, "down", None),
                            "distance": getattr(game_state, "distance", None),
                            "yard_line": getattr(game_state, "yard_line", None),
                            "quarter": getattr(game_state, "quarter", None),
                            "time": getattr(game_state, "time", None),
                            "possession": getattr(game_state, "possession_team", None),
                            "territory": getattr(game_state, "territory", None),
                            "confidence": getattr(game_state, "confidence", 0.0),
                        }

                        self.ocr_result.emit(result)

                except Exception as e:
                    print(f"Error at frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

    def stop(self):
        self.running = False

    def capture_frame(self):
        """Request to capture the next frame for debugging"""
        self.capture_next_frame = True


class MinimalOCRApp(QMainWindow):
    """Minimal app that shows ONLY OCR results"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI - OCR Only Test")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout(central_widget)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.browse_btn = QPushButton("Browse Video")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start OCR Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)

        # Screenshot debug button
        self.screenshot_btn = QPushButton("📸 Capture Debug Frame")
        self.screenshot_btn.clicked.connect(self.capture_debug_frame)
        self.screenshot_btn.setEnabled(False)
        self.screenshot_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.screenshot_btn)
        layout.addLayout(button_layout)

        # Main content area - split between results and debug view
        main_layout = QHBoxLayout()

        # Left side - Results display
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("OCR Analysis Results:"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        results_layout.addWidget(self.results_text)

        # Right side - Debug visualization
        debug_widget = QWidget()
        debug_layout = QVBoxLayout(debug_widget)
        debug_layout.addWidget(QLabel("Debug Frame Capture:"))

        self.debug_image_label = QLabel(
            "No frame captured yet\n\nClick '📸 Capture Debug Frame' during analysis\nto see what OCR detects"
        )
        self.debug_image_label.setMinimumSize(640, 360)
        self.debug_image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.debug_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_image_label.setScaledContents(True)
        debug_layout.addWidget(self.debug_image_label)

        # Debug info text
        self.debug_info_text = QTextEdit()
        self.debug_info_text.setFont(QFont("Consolas", 9))
        self.debug_info_text.setMaximumHeight(150)
        self.debug_info_text.setPlaceholderText(
            "Debug info will appear here when you capture a frame..."
        )
        debug_layout.addWidget(self.debug_info_text)

        # Add to main layout
        main_layout.addWidget(results_widget, 1)
        main_layout.addWidget(debug_widget, 1)
        layout.addLayout(main_layout)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.video_path = None
        self.worker = None

    def browse_file(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )

        if file_path:
            self.video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.start_btn.setEnabled(True)

    def start_analysis(self):
        """Start OCR analysis"""
        if not self.video_path:
            return

        self.results_text.clear()
        self.results_text.append("🧪 STARTING OCR-ONLY ANALYSIS")
        self.results_text.append("=" * 60)

        self.worker = OCROnlyWorker(self.video_path)
        self.worker.ocr_result.connect(self.display_ocr_result)
        self.worker.frame_captured.connect(self.display_debug_frame)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.status_label.setText("Analyzing...")

    def stop_analysis(self):
        """Stop analysis"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.status_label.setText("Stopped")

    def display_ocr_result(self, result):
        """Display OCR result"""
        timestamp = result["timestamp"]
        frame = result["frame"]
        down = result["down"]
        distance = result["distance"]
        yard_line = result["yard_line"]
        quarter = result["quarter"]
        time = result["time"]
        possession = result["possession"]
        territory = result["territory"]
        confidence = result["confidence"]

        # Format result
        text = f"\n🎯 FRAME {frame}: {timestamp:.1f}s\n"
        text += f"   Down: {down}\n"
        text += f"   Distance: {distance}\n"
        text += f"   Yard Line: {yard_line}\n"
        text += f"   Quarter: {quarter}\n"
        text += f"   Time: {time}\n"
        text += f"   Possession: {possession}\n"
        text += f"   Territory: {territory}\n"
        text += f"   Confidence: {confidence:.3f}\n"
        text += "-" * 40

        self.results_text.append(text)

        # Auto-scroll to bottom
        cursor = self.results_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.results_text.setTextCursor(cursor)

    def capture_debug_frame(self):
        """Capture the next frame for debugging"""
        if self.worker and self.worker.isRunning():
            self.worker.capture_frame()
            self.status_label.setText("Capturing next frame...")

    def display_debug_frame(self, frame, game_state):
        """Display captured frame with debug info"""
        import numpy as np

        # Convert frame to QPixmap for display
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)

        # Scale to fit the label
        scaled_pixmap = pixmap.scaled(
            self.debug_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.debug_image_label.setPixmap(scaled_pixmap)

        # Display debug info
        debug_info = f"🔍 DEBUG FRAME CAPTURE\n"
        debug_info += f"=" * 40 + "\n"
        debug_info += f"📊 OCR DETECTION RESULTS:\n"
        debug_info += f"   Down: {getattr(game_state, 'down', 'None')}\n"
        debug_info += f"   Distance: {getattr(game_state, 'distance', 'None')}\n"
        debug_info += f"   Yard Line: {getattr(game_state, 'yard_line', 'None')}\n"
        debug_info += f"   Quarter: {getattr(game_state, 'quarter', 'None')}\n"
        debug_info += f"   Time: {getattr(game_state, 'time', 'None')}\n"
        debug_info += f"   Possession: {getattr(game_state, 'possession_team', 'None')}\n"
        debug_info += f"   Territory: {getattr(game_state, 'territory', 'None')}\n"
        debug_info += f"   Confidence: {getattr(game_state, 'confidence', 0.0):.3f}\n"

        # Add visualization layers info if available
        if hasattr(game_state, "visualization_layers"):
            debug_info += f"\n🎨 VISUALIZATION LAYERS:\n"
            layers = game_state.visualization_layers
            for layer_name in layers.keys():
                debug_info += f"   ✅ {layer_name}\n"

        self.debug_info_text.setText(debug_info)
        self.status_label.setText("Frame captured! Check debug panel.")

        # Also add to results text
        self.results_text.append(f"\n📸 FRAME CAPTURED FOR DEBUG")
        self.results_text.append(
            f"   Down={getattr(game_state, 'down', 'None')}, Distance={getattr(game_state, 'distance', 'None')}"
        )
        self.results_text.append(
            f"   Yard Line={getattr(game_state, 'yard_line', 'None')}, Confidence={getattr(game_state, 'confidence', 0.0):.3f}"
        )

    def analysis_finished(self):
        """Analysis finished"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.status_label.setText("Analysis Complete")
        self.results_text.append("\n🧪 OCR-ONLY ANALYSIS COMPLETE")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MinimalOCRApp()
    window.show()
    sys.exit(app.exec())
