#!/usr/bin/env python3

"""
SpygateAI Phase 1 Real Analysis GUI Demo
========================================

A fully functional Phase 1 demo with real SpygateAI analysis integration.
"""

import logging
import os
import sys
import traceback
from pathlib import Path

# Set up proper Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add spygate to path
spygate_path = project_root / "spygate"
sys.path.insert(0, str(spygate_path))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("🏈 SpygateAI Phase 1 Real Analysis Demo")
    print("======================================")
    print("✅ PyQt6 and OpenCV loaded")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python")
    sys.exit(1)

# Try to import SpygateAI modules
SPYGATE_AVAILABLE = False
try:
    from ml.hud_detector import HUDDetector
    from ml.situation_detector import SituationDetector

    SPYGATE_AVAILABLE = True
    print("✅ SpygateAI modules loaded successfully")
except ImportError as e:
    print(f"⚠️  SpygateAI modules not available: {e}")
    print("📝 Running in demo mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    """Worker thread for running analysis in background."""

    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, frame, frame_number=0, fps=30):
        super().__init__()
        self.frame = frame
        self.frame_number = frame_number
        self.fps = fps

    def run(self):
        """Run analysis in background thread."""
        try:
            if SPYGATE_AVAILABLE:
                # Initialize detector
                detector = SituationDetector()
                detector.initialize()

                # Run analysis
                analysis_result = detector.detect_situations(
                    self.frame, self.frame_number, self.fps
                )

                self.analysis_complete.emit(analysis_result)
            else:
                # Fallback mock analysis
                mock_result = self.generate_mock_analysis()
                self.analysis_complete.emit(mock_result)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.error_occurred.emit(str(e))

    def generate_mock_analysis(self):
        """Generate mock analysis for demo purposes."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.frame_number / self.fps,
            "situations": [
                {
                    "type": "demo_mode",
                    "confidence": 0.95,
                    "frame": self.frame_number,
                    "timestamp": self.frame_number / self.fps,
                    "details": {
                        "message": "Real analysis requires SpygateAI modules",
                        "source": "demo_mode",
                    },
                }
            ],
            "hud_info": {
                "down": "Demo",
                "distance": "Mode",
                "score_home": 0,
                "score_away": 0,
                "game_clock": "0:00",
                "field_position": "DEMO 50",
                "confidence": 0.95,
            },
            "metadata": {
                "motion_score": 0.0,
                "hud_confidence": 0.95,
                "hardware_tier": "demo",
                "analysis_version": "demo-mode",
            },
        }


class VideoDropWidget(QLabel):
    """Enhanced drag-and-drop widget with better feedback."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText(
            "🎮 Drag & Drop Video Files Here\n\n📁 Click to browse\n\nSupported: MP4, MOV, AVI, MKV"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self.get_default_style())
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)

    def get_default_style(self):
        return """
            QLabel {
                border: 3px dashed #3B82F6;
                border-radius: 15px;
                background-color: #F0F9FF;
                font-size: 14px;
                padding: 20px;
                color: #1F2937;
            }
            QLabel:hover {
                background-color: #DBEAFE;
                border-color: #2563EB;
            }
        """

    def get_accept_style(self):
        return """
            QLabel {
                border: 3px solid #10B981;
                border-radius: 15px;
                background-color: #ECFDF5;
                font-size: 14px;
                padding: 20px;
                color: #1F2937;
            }
        """

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.accept()
                self.setStyleSheet(self.get_accept_style())
                self.setText("🎯 Drop to load video!")
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.get_default_style())
        self.setText(
            "🎮 Drag & Drop Video Files Here\n\n📁 Click to browse\n\nSupported: MP4, MOV, AVI, MKV"
        )

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.setText(f"✅ Loading: {os.path.basename(file_path)}")
                self.setStyleSheet(self.get_accept_style())
                self.video_dropped.emit(file_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File", "Please drop a valid video file (MP4, MOV, AVI, MKV)"
                )
                self.setStyleSheet(self.get_default_style())

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if file_path:
            self.setText(f"✅ Loading: {os.path.basename(file_path)}")
            self.video_dropped.emit(file_path)

    def is_video_file(self, file_path):
        return file_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


class VideoPlayer(QWidget):
    """Enhanced video player with real SpygateAI analysis."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.analysis_worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Video info
        self.info_label = QLabel("No video loaded")
        self.info_label.setStyleSheet("color: #6B7280; font-size: 12px; padding: 5px;")
        layout.addWidget(self.info_label)

        # Video display
        self.video_label = QLabel("Load a video to begin")
        self.video_label.setStyleSheet(
            """
            QLabel {
                border: 2px solid #D1D5DB;
                background-color: #000;
                color: white;
                min-height: 300px;
                max-height: 400px;
            }
        """
        )
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label)

        # Controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶️")
        self.play_btn.setMaximumWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.timeline.setEnabled(False)
        controls_layout.addWidget(self.timeline)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        controls_layout.addWidget(self.frame_label)

        layout.addLayout(controls_layout)

        # Analysis button
        self.analyze_btn = QPushButton("🧠 Analyze Current Frame with YOLOv8")
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #D1D5DB; }
        """
        )
        self.analyze_btn.clicked.connect(self.analyze_current_frame)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)

        # Progress bar for analysis
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Analysis results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(300)
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        layout.addWidget(self.results_text)

        self.setLayout(layout)

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False

    def load_video(self, path):
        """Load video with enhanced error handling."""
        try:
            # Clean up previous video
            if self.cap:
                self.cap.release()

            # Check if file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

            # Try to open video
            self.cap = cv2.VideoCapture(path)

            if not self.cap.isOpened():
                raise ValueError("Could not open video file. Check if the codec is supported.")

            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.total_frames <= 0:
                raise ValueError("Video has no frames or is corrupted")

            # Update UI
            self.current_frame = 0
            self.timeline.setRange(0, self.total_frames - 1)
            self.timeline.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)

            # Update info
            duration = self.total_frames / self.fps
            self.info_label.setText(
                f"📹 {os.path.basename(path)} | {width}x{height} | {duration:.1f}s | {self.fps:.1f}fps"
            )

            # Load first frame
            self.seek_frame(0)

            status_text = f"✅ Video loaded successfully!\n\nFile: {os.path.basename(path)}\nResolution: {width}x{height}\nDuration: {duration:.1f} seconds\nFPS: {self.fps:.1f}\nTotal frames: {self.total_frames}\n\n"

            if SPYGATE_AVAILABLE:
                status_text += (
                    "🎯 Ready for YOLOv8 analysis! Click 'Analyze Current Frame' to test."
                )
            else:
                status_text += "📝 Demo mode - install SpygateAI modules for real analysis"

            self.results_text.setText(status_text)

            return True

        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Video Load Error", error_msg)
            self.results_text.setText(f"❌ Error loading video:\n{str(e)}")
            return False

    def seek_frame(self, frame_num):
        """Seek to specific frame with error handling."""
        if not self.cap or not self.cap.isOpened():
            return

        try:
            self.current_frame = max(0, min(frame_num, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.display_frame(frame)
            else:
                logger.warning(f"Could not read frame {self.current_frame}")

            # Update timeline and counter
            self.timeline.blockSignals(True)
            self.timeline.setValue(self.current_frame)
            self.timeline.blockSignals(False)

            self.frame_label.setText(f"{self.current_frame + 1} / {self.total_frames}")

        except Exception as e:
            logger.error(f"Error seeking frame: {e}")

    def display_frame(self, frame):
        """Display frame in the video label."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # Create QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # Convert to pixmap and scale
            pixmap = QPixmap.fromImage(qt_image)

            # Scale to fit label while maintaining aspect ratio
            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.video_label.setPixmap(scaled_pixmap)
            else:
                self.video_label.setPixmap(pixmap)

        except Exception as e:
            logger.error(f"Error displaying frame: {e}")

    def toggle_play(self):
        """Toggle playback."""
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

    def get_current_frame(self):
        """Get current frame as numpy array."""
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            return frame if ret else None
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return None

    def analyze_current_frame(self):
        """Analyze current frame using real SpygateAI modules."""
        frame = self.get_current_frame()
        if frame is None:
            self.results_text.setText("❌ No frame available for analysis")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("🔍 Analyzing...")

        self.results_text.setText("🔍 Running YOLOv8 analysis...\nPlease wait...")
        QApplication.processEvents()

        # Start analysis in background thread
        self.analysis_worker = AnalysisWorker(frame, self.current_frame, self.fps)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def on_analysis_complete(self, analysis_result):
        """Handle completed analysis."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🧠 Analyze Current Frame with YOLOv8")

        self.display_analysis_results(analysis_result)

    def on_analysis_error(self, error_message):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🧠 Analyze Current Frame with YOLOv8")

        self.results_text.setText(f"❌ Analysis Error:\n{error_message}")

    def display_analysis_results(self, analysis):
        """Display real analysis results from SpygateAI."""
        self.results_text.clear()

        if SPYGATE_AVAILABLE:
            self.results_text.append("🎯 SpygateAI YOLOv8 Analysis Results:")
        else:
            self.results_text.append("📝 Demo Mode Analysis Results:")

        self.results_text.append("=" * 40)

        # Display HUD information
        hud_info = analysis.get("hud_info", {})
        if hud_info:
            self.results_text.append("\n📊 HUD Elements Detected:")

            # Game state info
            down = hud_info.get("down", "Unknown")
            distance = hud_info.get("distance", "Unknown")
            field_pos = hud_info.get("field_position", "Unknown")
            game_clock = hud_info.get("game_clock", "Unknown")

            self.results_text.append(f"  • Down & Distance: {down} & {distance}")
            self.results_text.append(f"  • Field Position: {field_pos}")
            self.results_text.append(f"  • Game Clock: {game_clock}")

            # Score info
            score_home = hud_info.get("score_home")
            score_away = hud_info.get("score_away")
            if score_home is not None and score_away is not None:
                self.results_text.append(f"  • Score: HOME {score_home} - AWAY {score_away}")

            # Confidence
            confidence = hud_info.get("confidence", 0.0)
            self.results_text.append(f"  • Detection Confidence: {confidence:.1%}")

        # Display detected situations
        situations = analysis.get("situations", [])
        if situations:
            self.results_text.append("\n🚨 Situations Detected:")

            for situation in situations:
                sit_type = situation.get("type", "Unknown")
                confidence = situation.get("confidence", 0.0)
                details = situation.get("details", {})

                self.results_text.append(
                    f"  • {sit_type.replace('_', ' ').title()} ({confidence:.1%} confidence)"
                )

                # Add situation-specific details
                if sit_type == "third_and_long":
                    down = details.get("down", "Unknown")
                    distance = details.get("distance", "Unknown")
                    self.results_text.append(f"    - Down: {down}, Distance: {distance} yards")
                elif sit_type == "red_zone":
                    field_pos = details.get("field_position", "Unknown")
                    self.results_text.append(f"    - Field Position: {field_pos}")
                elif sit_type == "two_minute_warning":
                    quarter = details.get("quarter", "Unknown")
                    clock = details.get("game_clock", "Unknown")
                    self.results_text.append(f"    - Quarter: {quarter}, Clock: {clock}")
                elif sit_type == "close_game":
                    score_diff = details.get("score_difference", "Unknown")
                    self.results_text.append(f"    - Score Difference: {score_diff} points")

        # Display metadata
        metadata = analysis.get("metadata", {})
        self.results_text.append("\n⚙️ Analysis Metadata:")

        motion_score = metadata.get("motion_score", 0.0)
        hardware_tier = metadata.get("hardware_tier", "Unknown")
        analysis_version = metadata.get("analysis_version", "Unknown")

        self.results_text.append(f"  • Motion Score: {motion_score:.2f}")
        self.results_text.append(f"  • Hardware Tier: {hardware_tier}")
        self.results_text.append(f"  • Analysis Version: {analysis_version}")

        # Frame info
        frame_number = analysis.get("frame_number", 0)
        timestamp = analysis.get("timestamp", 0.0)
        self.results_text.append(f"  • Frame: {frame_number + 1} (Time: {timestamp:.2f}s)")

        self.results_text.append("\n✅ Analysis complete!")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Phase 1 - Real Analysis Demo")
        self.setGeometry(100, 100, 1400, 900)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header_text = "🏈 SpygateAI Phase 1 - Real Analysis Demo"
        if SPYGATE_AVAILABLE:
            header_text += " (YOLOv8 Enabled)"
        else:
            header_text += " (Demo Mode)"

        header = QLabel(header_text)
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            """
            QLabel {
                padding: 15px;
                background-color: #3B82F6;
                color: white;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """
        )
        layout.addWidget(header)

        # Main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Drop area
        self.drop_widget = VideoDropWidget()
        self.drop_widget.video_dropped.connect(self.load_video)
        main_splitter.addWidget(self.drop_widget)

        # Right: Video player
        self.video_player = VideoPlayer()
        main_splitter.addWidget(self.video_player)

        main_splitter.setSizes([300, 1100])
        layout.addWidget(main_splitter)

        central.setLayout(layout)

        # Status bar
        status_text = "Ready - Drag video files or click to browse"
        if SPYGATE_AVAILABLE:
            status_text += " | Real YOLOv8 analysis available"
        else:
            status_text += " | Demo mode (install missing dependencies for real analysis)"

        self.statusBar().showMessage(status_text)

    def load_video(self, file_path):
        """Handle video loading."""
        self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")

        if self.video_player.load_video(file_path):
            self.drop_widget.setText(f"✅ Loaded: {os.path.basename(file_path)}")
            self.statusBar().showMessage(f"✅ Ready: {os.path.basename(file_path)}")
        else:
            self.drop_widget.setStyleSheet(self.drop_widget.get_default_style())
            self.drop_widget.setText("❌ Failed to load\n\nTry another video file")
            self.statusBar().showMessage("❌ Failed to load video")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpygateAI Phase 1")

    # Startup message
    msg = QMessageBox()
    msg.setWindowTitle("SpygateAI Phase 1 Real Analysis Demo")

    if SPYGATE_AVAILABLE:
        msg_text = """
🏈 SpygateAI Phase 1 Real Analysis Demo

Features:
✅ Real drag-and-drop video import
✅ Actual video playback with timeline
✅ Frame-by-frame navigation
✅ YOLOv8-based HUD analysis
✅ Real situation detection
✅ Background processing

Try dragging a video file and click "Analyze Current Frame"!
        """
    else:
        msg_text = """
🏈 SpygateAI Phase 1 Demo Mode

Features:
✅ Real drag-and-drop video import
✅ Actual video playback with timeline
✅ Frame-by-frame navigation
📝 Demo mode analysis

For real YOLOv8 analysis, ensure all SpygateAI
dependencies are installed.
        """

    msg.setText(msg_text)
    msg.exec()

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
