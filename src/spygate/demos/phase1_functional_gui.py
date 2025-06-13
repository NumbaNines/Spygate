#!/usr/bin/env python3

"""
SpygateAI Phase 1 Functional GUI Demo
====================================

A fully functional Phase 1 demo with real video processing and SpygateAI integration.
"""

import logging
import os
import sys
import traceback
from pathlib import Path

# Set up proper Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("🏈 SpygateAI Phase 1 Functional Demo")
    print("===================================")
    print("✅ PyQt6 and OpenCV loaded")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python")
    sys.exit(1)

# Try to import SpygateAI modules
SPYGATE_AVAILABLE = False
try:
    # Add the spygate package to Python path
    spygate_path = project_root / "spygate"
    sys.path.insert(0, str(spygate_path))

    from ml.hud_detector import HUDDetector
    from ml.situation_detector import SituationDetector

    SPYGATE_AVAILABLE = True
    print("✅ SpygateAI modules loaded successfully")
except ImportError as e:
    print(f"⚠️  SpygateAI modules not available: {e}")
    print("📝 Running in standalone mode")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Enhanced video player with better error handling."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
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
        self.analyze_btn = QPushButton("🧠 Analyze Current Frame")
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

        # Analysis results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
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

            self.results_text.setText(
                f"✅ Video loaded successfully!\n\nFile: {os.path.basename(path)}\nResolution: {width}x{height}\nDuration: {duration:.1f} seconds\nFPS: {self.fps:.1f}\nTotal frames: {self.total_frames}"
            )

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
        """Analyze current frame."""
        frame = self.get_current_frame()
        if frame is None:
            self.results_text.setText("❌ No frame available for analysis")
            return

        self.results_text.setText("🔍 Analyzing frame...\n")
        QApplication.processEvents()

        if SPYGATE_AVAILABLE:
            try:
                # Try real analysis
                detector = SituationDetector()
                analysis = detector.analyze_frame(frame)
                self.display_analysis_results(analysis)
            except Exception as e:
                self.display_mock_analysis()
                self.results_text.append(f"\n⚠️ Real analysis failed: {str(e)}")
        else:
            self.display_mock_analysis()

    def display_analysis_results(self, analysis):
        """Display real analysis results."""
        self.results_text.clear()
        self.results_text.append("🎯 YOLOv8 Analysis Results:")
        self.results_text.append("=" * 30)

        # Display results based on analysis structure
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                self.results_text.append(f"{key}: {value}")
        else:
            self.results_text.append(str(analysis))

    def display_mock_analysis(self):
        """Display mock analysis for demo."""
        self.results_text.clear()
        self.results_text.append("🎯 Phase 1 Analysis (Demo Mode):")
        self.results_text.append("=" * 35)
        self.results_text.append("\n📊 Detected HUD Elements:")
        self.results_text.append("  • Down: 3rd")
        self.results_text.append("  • Distance: 8 yards")
        self.results_text.append("  • Field Position: OPP 25")
        self.results_text.append("  • Score: HOME 14 - AWAY 21")
        self.results_text.append("  • Game Clock: 2:15")
        self.results_text.append("\n🚨 Situations Detected:")
        self.results_text.append("  • 3rd & Long (89% confidence)")
        self.results_text.append("  • Red Zone (85% confidence)")
        self.results_text.append("  • Two-Minute Warning (92% confidence)")
        self.results_text.append("\n⚡ Frame analysis complete!")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Phase 1 - Functional Demo")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header = QLabel("🏈 SpygateAI Phase 1 - Functional Demo")
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

        main_splitter.setSizes([300, 900])
        layout.addWidget(main_splitter)

        central.setLayout(layout)

        # Status bar
        status_text = "Ready - Drag video files or click to browse"
        if SPYGATE_AVAILABLE:
            status_text += " | YOLOv8 analysis available"
        else:
            status_text += " | Demo mode (install SpygateAI for full analysis)"

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
    msg.setWindowTitle("SpygateAI Phase 1 Functional Demo")
    msg.setText(
        f"""
🏈 SpygateAI Phase 1 Functional Demo

Features:
✅ Real drag-and-drop video import
✅ Actual video playback with timeline
✅ Frame-by-frame navigation
✅ Enhanced error handling
{"✅ YOLOv8 situation analysis" if SPYGATE_AVAILABLE else "📝 Demo mode analysis"}

Try dragging a video file to test!
    """
    )
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
