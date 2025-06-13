#!/usr/bin/env python3

"""
SpygateAI Phase 1 Complete Workflow GUI Demo
============================================

A complete Phase 1 demo showing the full workflow: analyze → bookmark → organize → act
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
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

    print("🏈 SpygateAI Phase 1 Complete Workflow Demo")
    print("==========================================")
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
                # Fallback mock analysis with varied results
                mock_result = self.generate_mock_analysis()
                self.analysis_complete.emit(mock_result)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.error_occurred.emit(str(e))

    def generate_mock_analysis(self):
        """Generate varied mock analysis for demo purposes."""
        import random

        # Generate varied situations for demo
        situations = []
        if random.random() > 0.3:  # 70% chance of 3rd down
            situations.append(
                {
                    "type": "third_and_long",
                    "confidence": 0.89,
                    "frame": self.frame_number,
                    "timestamp": self.frame_number / self.fps,
                    "details": {
                        "down": 3,
                        "distance": random.randint(7, 15),
                        "field_position": f"OPP {random.randint(20, 45)}",
                        "source": "hud_analysis",
                    },
                }
            )

        if random.random() > 0.6:  # 40% chance of red zone
            situations.append(
                {
                    "type": "red_zone",
                    "confidence": 0.92,
                    "frame": self.frame_number,
                    "timestamp": self.frame_number / self.fps,
                    "details": {
                        "field_position": f"OPP {random.randint(5, 20)}",
                        "source": "hud_analysis",
                    },
                }
            )

        if random.random() > 0.8:  # 20% chance of two-minute warning
            situations.append(
                {
                    "type": "two_minute_warning",
                    "confidence": 0.95,
                    "frame": self.frame_number,
                    "timestamp": self.frame_number / self.fps,
                    "details": {
                        "quarter": random.choice([2, 4]),
                        "game_clock": f"{random.randint(1, 2)}:{random.randint(10, 59):02d}",
                        "source": "hud_analysis",
                    },
                }
            )

        return {
            "frame_number": self.frame_number,
            "timestamp": self.frame_number / self.fps,
            "situations": situations,
            "hud_info": {
                "down": random.randint(1, 4),
                "distance": random.randint(1, 15),
                "score_home": random.randint(0, 35),
                "score_away": random.randint(0, 35),
                "game_clock": f"{random.randint(0, 15)}:{random.randint(0, 59):02d}",
                "field_position": f"{'OWN' if random.random() > 0.5 else 'OPP'} {random.randint(5, 50)}",
                "confidence": 0.85 + random.random() * 0.15,
            },
            "metadata": {
                "motion_score": random.random() * 0.8,
                "hud_confidence": 0.85 + random.random() * 0.15,
                "hardware_tier": "demo",
                "analysis_version": "demo-mode",
            },
        }


class ActionButtonsWidget(QWidget):
    """Widget containing action buttons for post-analysis workflow."""

    bookmark_clicked = pyqtSignal(dict)
    create_clip_clicked = pyqtSignal(dict)
    add_to_gameplan_clicked = pyqtSignal(dict)
    find_similar_clicked = pyqtSignal(dict)
    export_analysis_clicked = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_analysis = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("📋 What do you want to do with this analysis?")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #374151; padding: 10px 0;")
        layout.addWidget(title)

        # Primary actions
        primary_layout = QHBoxLayout()

        self.bookmark_btn = QPushButton("🔖 Bookmark Moment")
        self.bookmark_btn.setStyleSheet(self.get_primary_button_style("#059669"))
        self.bookmark_btn.clicked.connect(lambda: self.bookmark_clicked.emit(self.current_analysis))
        self.bookmark_btn.setEnabled(False)
        primary_layout.addWidget(self.bookmark_btn)

        self.create_clip_btn = QPushButton("✂️ Create Clip")
        self.create_clip_btn.setStyleSheet(self.get_primary_button_style("#DC2626"))
        self.create_clip_btn.clicked.connect(
            lambda: self.create_clip_clicked.emit(self.current_analysis)
        )
        self.create_clip_btn.setEnabled(False)
        primary_layout.addWidget(self.create_clip_btn)

        layout.addLayout(primary_layout)

        # Secondary actions
        secondary_layout = QHBoxLayout()

        self.gameplan_btn = QPushButton("📚 Add to Gameplan")
        self.gameplan_btn.setStyleSheet(self.get_secondary_button_style("#7C3AED"))
        self.gameplan_btn.clicked.connect(
            lambda: self.add_to_gameplan_clicked.emit(self.current_analysis)
        )
        self.gameplan_btn.setEnabled(False)
        secondary_layout.addWidget(self.gameplan_btn)

        self.similar_btn = QPushButton("🔍 Find Similar")
        self.similar_btn.setStyleSheet(self.get_secondary_button_style("#0891B2"))
        self.similar_btn.clicked.connect(
            lambda: self.find_similar_clicked.emit(self.current_analysis)
        )
        self.similar_btn.setEnabled(False)
        secondary_layout.addWidget(self.similar_btn)

        self.export_btn = QPushButton("📤 Export Analysis")
        self.export_btn.setStyleSheet(self.get_secondary_button_style("#EA580C"))
        self.export_btn.clicked.connect(
            lambda: self.export_analysis_clicked.emit(self.current_analysis)
        )
        self.export_btn.setEnabled(False)
        secondary_layout.addWidget(self.export_btn)

        layout.addLayout(secondary_layout)

        # Situation-specific actions
        self.situation_widget = QWidget()
        self.situation_layout = QVBoxLayout(self.situation_widget)
        layout.addWidget(self.situation_widget)

        self.setLayout(layout)

    def get_primary_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 12px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{ background-color: {color}CC; }}
            QPushButton:disabled {{ background-color: #D1D5DB; color: #9CA3AF; }}
        """

    def get_secondary_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: white;
                color: {color};
                border: 2px solid {color};
                padding: 10px 14px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {color}10; }}
            QPushButton:disabled {{ border-color: #D1D5DB; color: #9CA3AF; }}
        """

    def update_analysis(self, analysis):
        """Update the widget with new analysis results."""
        self.current_analysis = analysis

        # Enable all buttons
        for button in [
            self.bookmark_btn,
            self.create_clip_btn,
            self.gameplan_btn,
            self.similar_btn,
            self.export_btn,
        ]:
            button.setEnabled(True)

        # Clear previous situation-specific actions
        for i in reversed(range(self.situation_layout.count())):
            self.situation_layout.itemAt(i).widget().setParent(None)

        # Add situation-specific actions
        situations = analysis.get("situations", [])
        if situations:
            title = QLabel("🎯 Situation-Specific Actions:")
            title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            title.setStyleSheet("color: #6B7280; margin-top: 10px;")
            self.situation_layout.addWidget(title)

            for situation in situations:
                self.add_situation_actions(situation)

    def add_situation_actions(self, situation):
        """Add action buttons specific to the detected situation."""
        sit_type = situation.get("type", "unknown")
        confidence = situation.get("confidence", 0.0)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 5, 10, 5)

        # Situation label
        label = QLabel(f"{sit_type.replace('_', ' ').title()} ({confidence:.0%})")
        label.setStyleSheet("color: #374151; font-weight: 600;")
        layout.addWidget(label)

        # Situation-specific buttons
        if sit_type == "third_and_long":
            btn = QPushButton("📖 Study 3rd Down Plays")
            btn.setToolTip("Find successful 3rd down conversions in this situation")
        elif sit_type == "red_zone":
            btn = QPushButton("🏆 Red Zone Playbook")
            btn.setToolTip("Review red zone scoring strategies")
        elif sit_type == "two_minute_warning":
            btn = QPushButton("⏰ Clock Management")
            btn.setToolTip("Analyze clock management decisions")
        elif sit_type == "close_game":
            btn = QPushButton("🔥 Clutch Moments")
            btn.setToolTip("Study high-pressure game situations")
        else:
            btn = QPushButton("🎮 Study This Play")
            btn.setToolTip("Analyze this specific situation")

        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #F3F4F6;
                color: #374151;
                border: 1px solid #D1D5DB;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #E5E7EB; }
        """
        )
        btn.clicked.connect(lambda: self.handle_situation_action(sit_type, situation))
        layout.addWidget(btn)

        self.situation_layout.addWidget(container)

    def handle_situation_action(self, sit_type, situation):
        """Handle situation-specific action."""
        QMessageBox.information(
            self,
            "Situation Action",
            f"Opening {sit_type.replace('_', ' ').title()} analysis tools...\n\n"
            f"This would launch the specific analysis workflow for this situation type. "
            f"In the full version, this would:\n\n"
            f"• Filter your video library for similar situations\n"
            f"• Show success rates and recommended counters\n"
            f"• Launch formation analysis for this scenario\n"
            f"• Open relevant strategy guides",
        )


class VideoDropWidget(QLabel):
    """Enhanced drag-and-drop widget."""

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
    """Enhanced video player with complete workflow."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.analysis_worker = None
        self.bookmarks = []  # Store bookmarked moments
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
                padding: 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
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

        # Create a splitter for results and actions
        results_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Analysis results
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        results_splitter.addWidget(self.results_text)

        # Action buttons
        self.action_buttons = ActionButtonsWidget()
        self.action_buttons.bookmark_clicked.connect(self.bookmark_moment)
        self.action_buttons.create_clip_clicked.connect(self.create_clip)
        self.action_buttons.add_to_gameplan_clicked.connect(self.add_to_gameplan)
        self.action_buttons.find_similar_clicked.connect(self.find_similar)
        self.action_buttons.export_analysis_clicked.connect(self.export_analysis)
        results_splitter.addWidget(self.action_buttons)

        results_splitter.setSizes([400, 300])
        layout.addWidget(results_splitter)

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

            status_text = f"✅ Video loaded successfully!\n\n🎬 Ready for Phase 1 workflow:\n1. Navigate to any frame\n2. Click 'Analyze Current Frame'\n3. Choose your action from the options\n\nFile: {os.path.basename(path)}\nDuration: {duration:.1f} seconds"

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
        """Analyze current frame using SpygateAI modules."""
        frame = self.get_current_frame()
        if frame is None:
            self.results_text.setText("❌ No frame available for analysis")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("🔍 Analyzing...")

        self.results_text.setText("🔍 Running analysis...\nPlease wait...")
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
        self.analyze_btn.setText("🧠 Analyze Current Frame")

        self.display_analysis_results(analysis_result)
        self.action_buttons.update_analysis(analysis_result)

    def on_analysis_error(self, error_message):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🧠 Analyze Current Frame")

        self.results_text.setText(f"❌ Analysis Error:\n{error_message}")

    def display_analysis_results(self, analysis):
        """Display analysis results."""
        self.results_text.clear()

        if SPYGATE_AVAILABLE:
            self.results_text.append("🎯 SpygateAI Analysis Results:")
        else:
            self.results_text.append("📝 Demo Analysis Results:")

        self.results_text.append("=" * 40)

        # Display HUD information
        hud_info = analysis.get("hud_info", {})
        if hud_info:
            self.results_text.append("\n📊 HUD Elements:")

            down = hud_info.get("down", "Unknown")
            distance = hud_info.get("distance", "Unknown")
            field_pos = hud_info.get("field_position", "Unknown")
            game_clock = hud_info.get("game_clock", "Unknown")

            self.results_text.append(f"  • Down & Distance: {down} & {distance}")
            self.results_text.append(f"  • Field Position: {field_pos}")
            self.results_text.append(f"  • Game Clock: {game_clock}")

            score_home = hud_info.get("score_home")
            score_away = hud_info.get("score_away")
            if score_home is not None and score_away is not None:
                self.results_text.append(f"  • Score: HOME {score_home} - AWAY {score_away}")

        # Display detected situations
        situations = analysis.get("situations", [])
        if situations:
            self.results_text.append("\n🚨 Situations Detected:")

            for situation in situations:
                sit_type = situation.get("type", "Unknown")
                confidence = situation.get("confidence", 0.0)

                self.results_text.append(
                    f"  • {sit_type.replace('_', ' ').title()} ({confidence:.0%})"
                )

        self.results_text.append("\n👆 Choose an action from the options on the right!")

    # Action handlers
    def bookmark_moment(self, analysis):
        """Bookmark the current moment."""
        bookmark = {
            "frame": self.current_frame,
            "timestamp": self.current_frame / self.fps,
            "analysis": analysis,
            "created": datetime.now().isoformat(),
            "video_file": self.current_video,
        }
        self.bookmarks.append(bookmark)

        QMessageBox.information(
            self,
            "Moment Bookmarked",
            f"✅ Frame {self.current_frame + 1} has been bookmarked!\n\n"
            f"Timestamp: {self.current_frame / self.fps:.1f}s\n"
            f"Situations: {len(analysis.get('situations', []))}\n\n"
            f"Total bookmarks: {len(self.bookmarks)}",
        )

    def create_clip(self, analysis):
        """Create a clip around the current moment."""
        duration, ok = QInputDialog.getDouble(
            self, "Create Clip", "Enter clip duration (seconds):", 10.0, 1.0, 60.0, 1
        )

        if ok:
            start_time = max(0, self.current_frame / self.fps - duration / 2)
            end_time = min(
                self.total_frames / self.fps, self.current_frame / self.fps + duration / 2
            )

            QMessageBox.information(
                self,
                "Clip Created",
                f"✂️ Clip created successfully!\n\n"
                f"Duration: {duration}s\n"
                f"Start: {start_time:.1f}s\n"
                f"End: {end_time:.1f}s\n\n"
                f"In the full version, this would:\n"
                f"• Extract the video segment\n"
                f"• Save with analysis metadata\n"
                f"• Add to your clip library\n"
                f"• Tag with detected situations",
            )

    def add_to_gameplan(self, analysis):
        """Add the analysis to a gameplan."""
        opponents = ["ProPlayer123", "TopCompetitor", "MCSChamp", "New Opponent..."]
        opponent, ok = QInputDialog.getItem(
            self, "Add to Gameplan", "Select opponent gameplan:", opponents, 0, False
        )

        if ok:
            if opponent == "New Opponent...":
                opponent, ok = QInputDialog.getText(self, "New Opponent", "Enter opponent name:")
                if not ok:
                    return

            QMessageBox.information(
                self,
                "Added to Gameplan",
                f"📚 Analysis added to {opponent}'s gameplan!\n\n"
                f"This moment will now be available when preparing to face {opponent}.\n\n"
                f"In the full version, this would:\n"
                f"• Save the analysis to opponent's profile\n"
                f"• Update tendency statistics\n"
                f"• Suggest counter-strategies\n"
                f"• Link to similar situations",
            )

    def find_similar(self, analysis):
        """Find similar situations in the video library."""
        QMessageBox.information(
            self,
            "Finding Similar Situations",
            f"🔍 Searching for similar situations...\n\n"
            f"In the full version, this would:\n"
            f"• Search your entire video library\n"
            f"• Find clips with similar HUD states\n"
            f"• Show success rates for each situation\n"
            f"• Recommend best counter-strategies\n"
            f"• Display formation matchups\n\n"
            f"Found situations would be organized by:\n"
            f"• Situation type\n"
            f"• Success rate\n"
            f"• Opponent tendencies\n"
            f"• Your performance history",
        )

    def export_analysis(self, analysis):
        """Export the analysis data."""
        formats = ["JSON (for sharing)", "CSV (for spreadsheets)", "Text (readable)"]
        format_choice, ok = QInputDialog.getItem(
            self, "Export Analysis", "Choose export format:", formats, 0, False
        )

        if ok:
            QMessageBox.information(
                self,
                "Analysis Exported",
                f"📤 Analysis exported as {format_choice}!\n\n"
                f"In the full version, this would:\n"
                f"• Save the analysis data\n"
                f"• Include video thumbnail\n"
                f"• Add metadata and timestamps\n"
                f"• Enable sharing with team/coach\n"
                f"• Support Discord integration",
            )


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Phase 1 - Complete Workflow Demo")
        self.setGeometry(100, 100, 1600, 1000)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header_text = "🏈 SpygateAI Phase 1 - Complete Workflow Demo"
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

        # Workflow description
        workflow_desc = QLabel(
            "📋 Workflow: Load Video → Navigate → Analyze Frame → Choose Action → Organize Results"
        )
        workflow_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        workflow_desc.setStyleSheet(
            """
            QLabel {
                padding: 10px;
                background-color: #F9FAFB;
                color: #374151;
                border-radius: 6px;
                font-size: 13px;
                margin-bottom: 10px;
            }
        """
        )
        layout.addWidget(workflow_desc)

        # Main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Drop area
        self.drop_widget = VideoDropWidget()
        self.drop_widget.video_dropped.connect(self.load_video)
        main_splitter.addWidget(self.drop_widget)

        # Right: Video player
        self.video_player = VideoPlayer()
        main_splitter.addWidget(self.video_player)

        main_splitter.setSizes([300, 1300])
        layout.addWidget(main_splitter)

        central.setLayout(layout)

        # Status bar
        status_text = "Ready - Drag video files to start the Phase 1 workflow"
        if SPYGATE_AVAILABLE:
            status_text += " | Real YOLOv8 analysis available"
        else:
            status_text += " | Demo mode active"

        self.statusBar().showMessage(status_text)

    def load_video(self, file_path):
        """Handle video loading."""
        self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")

        if self.video_player.load_video(file_path):
            self.drop_widget.setText(f"✅ Loaded: {os.path.basename(file_path)}")
            self.statusBar().showMessage(f"✅ Ready for analysis: {os.path.basename(file_path)}")
        else:
            self.drop_widget.setStyleSheet(self.drop_widget.get_default_style())
            self.drop_widget.setText("❌ Failed to load\n\nTry another video file")
            self.statusBar().showMessage("❌ Failed to load video")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpygateAI Phase 1")

    # Startup message
    msg = QMessageBox()
    msg.setWindowTitle("SpygateAI Phase 1 Complete Workflow")

    msg_text = """
🏈 SpygateAI Phase 1 Complete Workflow Demo

This demo shows the complete Phase 1 workflow:

1. 📹 Load Video - Drag & drop any gameplay video
2. 🎯 Analyze Frame - Click to analyze any moment
3. 📋 Choose Action - Multiple post-analysis options:
   • 🔖 Bookmark important moments
   • ✂️ Create clips for review
   • 📚 Add to opponent gameplans
   • 🔍 Find similar situations
   • 📤 Export analysis data

Each detected situation (3rd & Long, Red Zone, etc.) gets
situation-specific action buttons for targeted analysis.

Try the full workflow with your gameplay videos!
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
