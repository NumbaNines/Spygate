#!/usr/bin/env python3

"""
SpygateAI Phase 1 Working GUI Demo
==================================

A fully functional Phase 1 demo with:
✅ Real drag-and-drop video import
✅ Actual video processing with YOLOv8
✅ Live HUD analysis and situation detection
✅ Working timeline and playback
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Fix import path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("🏈 SpygateAI Phase 1 Working GUI Demo")
    print("=====================================")
    print("✅ All dependencies available")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python")
    sys.exit(1)

# Import SpygateAI modules with error handling
try:
    from spygate.core.hardware import HardwareDetector
    from spygate.ml.hud_detector import HUDDetector
    from spygate.ml.situation_detector import SituationDetector

    SPYGATE_AVAILABLE = True
    print("✅ SpygateAI modules loaded successfully")
except ImportError as e:
    print(f"⚠️  SpygateAI modules not fully available: {e}")
    print("📝 Running in demo mode with mock data")
    SPYGATE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDropWidget(QLabel):
    """Drag-and-drop widget for video files."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText(
            "🎮 Drag & Drop Video Files Here\n\n📁 Or click to browse\n\nSupported: MP4, MOV, AVI"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #3B82F6;
                border-radius: 15px;
                background-color: #F0F9FF;
                min-height: 200px;
                font-size: 16px;
                padding: 20px;
                color: #1F2937;
            }
            QLabel:hover {
                background-color: #DBEAFE;
                border-color: #2563EB;
            }
        """
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.accept()
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 15px;
                        background-color: #ECFDF5;
                        min-height: 200px;
                        font-size: 16px;
                        padding: 20px;
                        color: #1F2937;
                    }
                """
                )
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #3B82F6;
                border-radius: 15px;
                background-color: #F0F9FF;
                min-height: 200px;
                font-size: 16px;
                padding: 20px;
                color: #1F2937;
            }
        """
        )

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.video_dropped.emit(file_path)
                self.setText(f"✅ Loaded: {os.path.basename(file_path)}")
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 15px;
                        background-color: #ECFDF5;
                        min-height: 200px;
                        font-size: 16px;
                        padding: 20px;
                        color: #1F2937;
                    }
                """
                )
            else:
                QMessageBox.warning(
                    self, "Invalid File", "Please drop a valid video file (MP4, MOV, AVI)"
                )

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi)"
        )
        if file_path:
            self.video_dropped.emit(file_path)
            self.setText(f"✅ Loaded: {os.path.basename(file_path)}")

    def is_video_file(self, file_path):
        return file_path.lower().endswith((".mp4", ".mov", ".avi"))


class VideoPlayer(QWidget):
    """Video player with timeline and controls."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel("Load a video to begin playback")
        self.video_label.setStyleSheet(
            """
            QLabel {
                border: 2px solid #D1D5DB;
                background-color: #000;
                color: white;
                min-height: 300px;
                text-align: center;
            }
        """
        )
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Controls
        controls = QHBoxLayout()

        self.play_btn = QPushButton("▶️")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls.addWidget(self.play_btn)

        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.timeline.setEnabled(False)
        controls.addWidget(self.timeline)

        self.frame_label = QLabel("0 / 0")
        controls.addWidget(self.frame_label)

        layout.addLayout(controls)
        self.setLayout(layout)

    def load_video(self, path):
        """Load a video file for playback."""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.current_frame = 0

            if self.total_frames > 0:
                self.timeline.setRange(0, self.total_frames - 1)
                self.timeline.setEnabled(True)
                self.play_btn.setEnabled(True)

                # Load first frame
                self.seek_frame(0)
                self.update_frame_label()

                return True
            else:
                QMessageBox.warning(self, "Error", "Could not load video file")
                return False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
            return False

    def toggle_play(self):
        """Toggle video playback."""
        if not self.cap:
            return

        if self.playing:
            self.timer.stop()
            self.play_btn.setText("▶️")
            self.playing = False
        else:
            self.timer.start(int(1000 / self.fps))
            self.play_btn.setText("⏸️")
            self.playing = True

    def next_frame(self):
        """Move to next frame."""
        if self.current_frame < self.total_frames - 1:
            self.seek_frame(self.current_frame + 1)
        else:
            self.toggle_play()  # Stop at end

    def seek_frame(self, frame_num):
        """Seek to specific frame."""
        if not self.cap:
            return

        self.current_frame = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = self.cap.read()
        if ret:
            # Convert frame to Qt format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Scale to fit display
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.video_label.setPixmap(scaled_pixmap)

        self.timeline.blockSignals(True)
        self.timeline.setValue(frame_num)
        self.timeline.blockSignals(False)

        self.update_frame_label()

    def update_frame_label(self):
        """Update frame counter label."""
        self.frame_label.setText(f"{self.current_frame + 1} / {self.total_frames}")

    def get_current_frame(self):
        """Get current frame as numpy array."""
        if not self.cap:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None


class AnalysisPanel(QWidget):
    """Panel for showing analysis results."""

    def __init__(self):
        super().__init__()
        self.situation_detector = None
        self.init_ui()

        # Initialize situation detector if available
        if SPYGATE_AVAILABLE:
            try:
                self.situation_detector = SituationDetector()
                self.status_label.setText("✅ YOLOv8 situation detector ready")
            except Exception as e:
                self.status_label.setText(f"⚠️ Detector error: {str(e)}")
        else:
            self.status_label.setText("📝 Demo mode - using mock analysis")

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("🧠 Real-time Analysis")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(header)

        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #059669; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Analyze button
        self.analyze_btn = QPushButton("🚀 Analyze Current Frame")
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #D1D5DB; }
        """
        )
        self.analyze_btn.clicked.connect(self.analyze_frame)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        # Results area
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMaximumHeight(300)
        layout.addWidget(self.results)

        self.setLayout(layout)

    def set_video_player(self, player):
        """Connect to video player."""
        self.video_player = player
        self.analyze_btn.setEnabled(True)

    def analyze_frame(self):
        """Analyze current video frame."""
        if not hasattr(self, "video_player"):
            return

        frame = self.video_player.get_current_frame()
        if frame is None:
            self.results.setText("❌ No frame available for analysis")
            return

        self.results.clear()
        self.results.append("🔍 Analyzing frame...")
        QApplication.processEvents()

        if SPYGATE_AVAILABLE and self.situation_detector:
            try:
                # Real analysis
                analysis = self.situation_detector.analyze_frame(frame)
                self.display_real_analysis(analysis)
            except Exception as e:
                self.display_mock_analysis()
                self.results.append(f"\n⚠️ Real analysis failed: {str(e)}")
        else:
            # Mock analysis
            self.display_mock_analysis()

    def display_real_analysis(self, analysis):
        """Display real analysis results."""
        self.results.clear()
        self.results.append("🎯 YOLOv8 Analysis Results:")
        self.results.append("=" * 30)

        if "hud_data" in analysis:
            hud = analysis["hud_data"]
            self.results.append("\n📊 HUD Information:")
            for key, value in hud.items():
                self.results.append(f"  • {key}: {value}")

        if "situations" in analysis:
            self.results.append("\n🚨 Detected Situations:")
            for situation in analysis["situations"]:
                conf = situation.get("confidence", 0)
                self.results.append(f"  • {situation['type']} (confidence: {conf:.2f})")

        if "performance" in analysis:
            perf = analysis["performance"]
            self.results.append(f"\n⚡ Processing time: {perf.get('time', 0):.2f}s")

    def display_mock_analysis(self):
        """Display mock analysis for demo."""
        self.results.clear()
        self.results.append("🎯 Phase 1 Analysis Results (Demo Mode):")
        self.results.append("=" * 40)

        self.results.append("\n📊 HUD Information:")
        self.results.append("  • Down: 3rd")
        self.results.append("  • Distance: 8 yards")
        self.results.append("  • Field Position: OPP 25")
        self.results.append("  • Score: HOME 14 - AWAY 21")
        self.results.append("  • Game Clock: 2:15")

        self.results.append("\n🚨 Detected Situations:")
        self.results.append("  • 3rd & Long (confidence: 0.89)")
        self.results.append("  • Red Zone (confidence: 0.85)")
        self.results.append("  • Two-Minute Warning (confidence: 0.92)")

        self.results.append("\n⚡ Processing: Real-time ready!")


class Phase1WorkingDemo(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Phase 1 - Working Demo")
        self.setGeometry(100, 100, 1400, 800)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header = QLabel("🏈 SpygateAI Phase 1 - Working Demo")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            """
            QLabel {
                padding: 20px;
                background-color: #3B82F6;
                color: white;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """
        )
        layout.addWidget(header)

        # Main content area
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Video import and player
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Drop area
        self.drop_widget = VideoDropWidget()
        self.drop_widget.video_dropped.connect(self.load_video)
        left_layout.addWidget(self.drop_widget)

        # Video player
        self.video_player = VideoPlayer()
        left_layout.addWidget(self.video_player)

        left_widget.setLayout(left_layout)
        main_splitter.addWidget(left_widget)

        # Right side - Analysis
        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.set_video_player(self.video_player)
        main_splitter.addWidget(self.analysis_panel)

        # Set splitter proportions
        main_splitter.setSizes([700, 400])

        layout.addWidget(main_splitter)
        central.setLayout(layout)

        # Status bar
        self.statusBar().showMessage("Ready - Drag video files to begin Phase 1 analysis")

    def load_video(self, file_path):
        """Load video and prepare for analysis."""
        self.statusBar().showMessage(f"Loading video: {os.path.basename(file_path)}")

        if self.video_player.load_video(file_path):
            self.statusBar().showMessage(f"✅ Ready: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Video Loaded",
                f"Successfully loaded:\n{os.path.basename(file_path)}\n\nUse playback controls or click 'Analyze Current Frame' to begin!",
            )
        else:
            self.statusBar().showMessage("❌ Failed to load video")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpygateAI Phase 1 Working Demo")

    # Show intro
    msg = QMessageBox()
    msg.setWindowTitle("SpygateAI Phase 1 Working Demo")
    msg.setText(
        """
🏈 Welcome to SpygateAI Phase 1 Working Demo!

This is a fully functional demo featuring:

✅ Real drag-and-drop video import
✅ Actual video playback and timeline
✅ Live YOLOv8-based HUD analysis
✅ Situation detection engine
✅ Phase 1 MVP functionality

Ready to analyze your gameplay footage?
    """
    )
    msg.exec()

    # Create and show main window
    window = Phase1WorkingDemo()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
