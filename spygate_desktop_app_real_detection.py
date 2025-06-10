#!/usr/bin/env python3

"""
SpygateAI Desktop Application - Real Situation Detection Version
===============================================================

Enhanced version with actual SituationDetector integration for real gameplay analysis.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    # Import real detection classes
    from spygate.ml.situation_detector import SituationDetector
    from spygate.ml.hud_detector import HUDDetector
    from spygate.core.hardware import HardwareDetector
    from spygate.core.optimizer import TierOptimizer

    print("✅ All core modules imported successfully")
    REAL_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📝 Running in fallback mode...")
    REAL_DETECTION_AVAILABLE = False
    # PyQt6 imports for basic functionality
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *


@dataclass
class ClipData:
    """Data structure for detected clips with real situation data."""
    start_frame: int
    end_frame: int
    situation: str
    confidence: float
    timestamp: str
    approved: bool = False
    # Real detection data
    hud_info: Dict[str, Any] = None
    detection_metadata: Dict[str, Any] = None


class RealAutoClipDetector(QObject):
    """Real auto-clip detection using actual SituationDetector."""

    clip_detected = pyqtSignal(ClipData)
    analysis_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.is_analyzing = False
        self.config = {}
        self.detected_clips = []
        
        # Initialize real detection components if available
        if REAL_DETECTION_AVAILABLE:
            self.hardware = HardwareDetector()
            self.optimizer = TierOptimizer(self.hardware)
            self.situation_detector = None
            self.hud_detector = None
            self._initialize_detectors()
        else:
            print("⚠️ Real detection unavailable - using fallback mode")

    def _initialize_detectors(self):
        """Initialize the real detection components."""
        try:
            # Initialize HUD detector
            self.hud_detector = HUDDetector()
            print("✅ HUDDetector initialized")
            
            # Initialize situation detector
            self.situation_detector = SituationDetector()
            if hasattr(self.situation_detector, 'initialize'):
                self.situation_detector.initialize()
            print("✅ SituationDetector initialized")
            
        except Exception as e:
            print(f"❌ Error initializing detectors: {e}")
            self.situation_detector = None
            self.hud_detector = None

    def setup_optimization_settings(self, hardware_tier: str):
        """Setup optimization based on hardware tier."""
        if REAL_DETECTION_AVAILABLE and self.optimizer:
            tier_config = self.optimizer.get_tier_config()
            frame_skip = tier_config.get('frame_skip', 30)
        else:
            # Fallback configuration
            tier_configs = {
                'ultra': {'frame_skip': 15, 'max_clips_per_minute': 8},
                'high': {'frame_skip': 30, 'max_clips_per_minute': 6},
                'medium': {'frame_skip': 60, 'max_clips_per_minute': 4},
                'low': {'frame_skip': 90, 'max_clips_per_minute': 2}
            }
            frame_skip = tier_configs.get(hardware_tier, {}).get('frame_skip', 30)

        self.config = {
            'frame_skip': frame_skip,
            'scene_check_interval': 10,
            'max_clips_per_minute': tier_configs.get(hardware_tier, {}).get('max_clips_per_minute', 4)
        }
        
        print(f"🚀 Real detection optimized for {hardware_tier} tier: frame skip {frame_skip}")

    def analyze_video(self, video_path: str, hardware_tier: str):
        """Analyze video using real situation detection."""
        self.is_analyzing = True
        self.detected_clips = []
        self.setup_optimization_settings(hardware_tier)

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"🎬 Starting REAL analysis: {total_frames} frames at {fps} FPS")
            print(f"⚡ Frame skip: {self.config['frame_skip']}, Real detection: {REAL_DETECTION_AVAILABLE}")

            start_time = time.time()
            frames_processed = 0
            clips_detected = 0
            frame_count = 0

            while cap.isOpened() and self.is_analyzing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Smart frame skipping
                should_process = (frame_count % self.config["frame_skip"]) == 0

                if should_process:
                    frames_processed += 1

                    # Update progress
                    progress = int((frame_count / total_frames) * 100)
                    self.analysis_progress.emit(
                        progress,
                        f"🔍 Real Analysis: {frame_count}/{total_frames} (Processed: {frames_processed})"
                    )

                    # REAL SITUATION DETECTION
                    situation_result = self._analyze_frame_real(frame, frame_count, fps)
                    
                    if situation_result and self._is_significant_situation(situation_result):
                        timestamp = frame_count / fps
                        clips_in_last_minute = len([
                            clip for clip in self.detected_clips
                            if hasattr(clip, 'start_frame') and clip.start_frame > (frame_count - 60 * fps)
                        ])

                        if clips_in_last_minute < self.config["max_clips_per_minute"]:
                            clip_data = ClipData(
                                start_frame=max(0, frame_count - 150),
                                end_frame=min(total_frames, frame_count + 150),
                                situation=situation_result.get('situation', 'Unknown'),
                                confidence=situation_result.get('confidence', 0.0),
                                timestamp=self._frame_to_timestamp(frame_count, fps),
                                hud_info=situation_result.get('hud_info', {}),
                                detection_metadata=situation_result.get('metadata', {})
                            )
                            self.detected_clips.append(clip_data)
                            self.clip_detected.emit(clip_data)
                            clips_detected += 1

            cap.release()

            # Log results
            end_time = time.time()
            total_time = end_time - start_time
            print(f"🎯 Real Detection Results:")
            print(f"  Processing mode: {'REAL DETECTION' if REAL_DETECTION_AVAILABLE else 'FALLBACK'}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Frames processed: {frames_processed}/{total_frames}")
            print(f"  Real clips detected: {clips_detected}")

            self.analysis_complete.emit(self.detected_clips)

        except Exception as e:
            print(f"❌ Error in real video analysis: {e}")
        finally:
            self.is_analyzing = False

    def _analyze_frame_real(self, frame: np.ndarray, frame_count: int, fps: float) -> Optional[Dict[str, Any]]:
        """Analyze frame using real situation detection."""
        if not REAL_DETECTION_AVAILABLE or not self.situation_detector:
            # Fallback to simplified detection
            return self._fallback_detection(frame, frame_count, fps)

        try:
            # REAL SITUATION DETECTION
            result = self.situation_detector.detect_situations(frame, frame_count, fps)
            
            if result and result.get('situations'):
                situations = result['situations']
                if situations:
                    # Get the highest confidence situation
                    best_situation = max(situations, key=lambda x: x.get('confidence', 0))
                    
                    return {
                        'situation': best_situation.get('type', 'Unknown'),
                        'confidence': best_situation.get('confidence', 0.0),
                        'hud_info': result.get('hud_info', {}),
                        'metadata': result.get('metadata', {}),
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps
                    }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Real detection error: {e}")
            return self._fallback_detection(frame, frame_count, fps)

    def _fallback_detection(self, frame: np.ndarray, frame_count: int, fps: float) -> Optional[Dict[str, Any]]:
        """Fallback detection when real detection is unavailable."""
        # Calculate frame variance for action detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_variance = np.var(gray)

        situations = [
            "1st & 10", "2nd & 5", "3rd & Long", "4th & Goal",
            "Red Zone Opportunity", "Two Minute Warning", 
            "Turnover", "Scoring Play", "Defensive Stop"
        ]

        if frame_variance > 1000:  # High action frame
            if np.random.random() > 0.6:  # Moderate chance
                situation = np.random.choice(situations)
                return {
                    'situation': f"{situation} (Simulated)",
                    'confidence': 0.7 + (np.random.random() * 0.2),
                    'hud_info': {'detection_mode': 'fallback'},
                    'metadata': {'variance': float(frame_variance)},
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps
                }
        
        return None

    def _is_significant_situation(self, situation_result: Dict[str, Any]) -> bool:
        """Determine if situation is significant enough for clipping."""
        if not situation_result:
            return False
            
        confidence = situation_result.get('confidence', 0.0)
        situation = situation_result.get('situation', '')
        
        # High-value situations (lower confidence threshold)
        high_value = ["3rd & Long", "Red Zone", "Two Minute Warning", "Turnover", "Scoring Play"]
        
        if any(term in situation for term in high_value):
            return confidence > 0.6
        
        # Regular situations need higher confidence
        return confidence > 0.75

    def _frame_to_timestamp(self, frame: int, fps: float) -> str:
        """Convert frame number to timestamp."""
        seconds = frame / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def stop_analysis(self):
        """Stop the analysis process."""
        self.is_analyzing = False


# Continue with existing UI classes but updated to use RealAutoClipDetector
class VideoDropZone(QLabel):
    """FACEIT-style drag & drop zone."""
    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        self.setText("🎮 Drop Video Here\n\nSupported: MP4, MOV, AVI\nMax size: 2GB\n\n🔍 REAL DETECTION MODE")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #ff6b35;
                border-radius: 12px;
                background-color: #1a1a1a;
                color: #888;
                font-size: 16px;
                font-weight: bold;
                padding: 40px;
                min-height: 200px;
            }
            QLabel:hover {
                border-color: #ff8c42;
                background-color: #2a2a2a;
                color: #fff;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(self.styleSheet().replace("#ff6b35", "#00ff00"))
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.styleSheet().replace("#00ff00", "#ff6b35"))

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and self._is_video_file(files[0]):
            self.video_dropped.emit(files[0])
        self.setStyleSheet(self.styleSheet().replace("#00ff00", "#ff6b35"))

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.video_dropped.emit(file_path)

    def _is_video_file(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))


class AutoDetectWidget(QWidget):
    """Enhanced auto-detect widget with real situation detection."""

    def __init__(self):
        super().__init__()
        self.detector = RealAutoClipDetector()
        self.current_video = None
        self.detected_clips = []
        self.approved_clips = []
        
        self.init_ui()
        self.connect_signals()
        self.detect_hardware()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("🎯 SpygateAI - Real Situation Detection")
        header.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #ff6b35;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(header)

        # Detection status
        status_text = "🔍 REAL DETECTION MODE" if REAL_DETECTION_AVAILABLE else "⚠️ FALLBACK MODE"
        self.status_label = QLabel(status_text)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #00ff00;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(self.status_label)

        # Drop zone
        self.drop_zone = VideoDropZone()
        layout.addWidget(self.drop_zone)

        # Controls
        self._create_controls(layout)

        # Clips panel
        self._create_clips_panel(layout)

        self.setLayout(layout)

    def _create_controls(self, layout):
        """Create control buttons."""
        controls_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("🔍 Start Real Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff6b35, stop:1 #e55a2b);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ff8c42, stop:1 #ff6b35);
            }
            QPushButton:disabled {
                background: #444;
                color: #888;
            }
        """)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self.analyze_btn.styleSheet().replace("#ff6b35", "#dc3545").replace("#e55a2b", "#c82333"))
        
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #333;
                border-radius: 6px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b35, stop:1 #ff8c42);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

    def _create_clips_panel(self, layout):
        """Create the clips management panel."""
        clips_label = QLabel("📋 Detected Clips (Real Analysis)")
        clips_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #ff6b35;
                margin-top: 20px;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(clips_label)

        # Clips scroll area
        self.clips_scroll = QScrollArea()
        self.clips_widget = QWidget()
        self.clips_layout = QVBoxLayout(self.clips_widget)
        self.clips_scroll.setWidget(self.clips_widget)
        self.clips_scroll.setWidgetResizable(True)
        self.clips_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #333;
                border-radius: 6px;
                background-color: #1a1a1a;
            }
        """)
        layout.addWidget(self.clips_scroll)

    def connect_signals(self):
        """Connect signals and slots."""
        self.drop_zone.video_dropped.connect(self.load_video)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.detector.stop_analysis)
        self.detector.analysis_progress.connect(self.update_progress)
        self.detector.analysis_complete.connect(self.analysis_finished)
        self.detector.clip_detected.connect(self.add_clip_card)

    def detect_hardware(self):
        """Detect hardware and update status."""
        if REAL_DETECTION_AVAILABLE and hasattr(self.detector, 'hardware'):
            hardware = self.detector.hardware
            tier = hardware.get_performance_tier()
            self.hardware_tier = tier.lower() if hasattr(tier, 'lower') else str(tier).lower()
            hardware_text = f"💻 Hardware: {tier} tier"
        else:
            self.hardware_tier = "medium"
            hardware_text = "💻 Hardware: Detected (Fallback mode)"
        
        self.status_label.setText(
            f"🔍 {'REAL DETECTION' if REAL_DETECTION_AVAILABLE else 'FALLBACK'} MODE | {hardware_text}"
        )

    def load_video(self, file_path: str):
        """Load video for analysis."""
        self.current_video = file_path
        self.analyze_btn.setEnabled(True)
        self.clear_clips()
        
        filename = Path(file_path).name
        self.drop_zone.setText(f"✅ Video Loaded: {filename}\n\nReady for Real Analysis!")

    def start_analysis(self):
        """Start the real analysis process."""
        if not self.current_video:
            return

        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start analysis in separate thread
        analysis_thread = threading.Thread(
            target=self.detector.analyze_video,
            args=(self.current_video, self.hardware_tier),
            daemon=True
        )
        analysis_thread.start()

    def update_progress(self, value: int, message: str):
        """Update progress display."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")

    def analysis_finished(self, clips: List[ClipData]):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        detection_mode = "Real Detection" if REAL_DETECTION_AVAILABLE else "Fallback Mode"
        self.drop_zone.setText(
            f"✅ Analysis Complete!\n\n{len(clips)} clips detected\nMode: {detection_mode}"
        )

    def add_clip_card(self, clip_data: ClipData):
        """Add a clip card to the display."""
        card = self.create_clip_card(clip_data)
        self.clips_layout.addWidget(card)

    def create_clip_card(self, clip_data: ClipData) -> QWidget:
        """Create a clip card widget."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(card)
        
        # Clip info
        info_text = f"🎯 {clip_data.situation}\n"
        info_text += f"⏰ {clip_data.timestamp}\n"
        info_text += f"📊 Confidence: {clip_data.confidence:.1%}"
        
        if clip_data.hud_info:
            info_text += f"\n🎮 HUD Data: {len(clip_data.hud_info)} elements"
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: white; font-size: 12px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Approve/Reject buttons
        approve_btn = QPushButton("✅ Approve")
        approve_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        reject_btn = QPushButton("❌ Reject")
        reject_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        
        layout.addWidget(approve_btn)
        layout.addWidget(reject_btn)
        
        return card

    def clear_clips(self):
        """Clear all clips from display."""
        for i in reversed(range(self.clips_layout.count())):
            self.clips_layout.itemAt(i).widget().setParent(None)
        self.detected_clips.clear()


class SpygateDesktopAppReal(QMainWindow):
    """Main application window with real detection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop - Real Detection")
        self.setMinimumSize(1200, 800)
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the auto-detect widget
        self.auto_detect_widget = AutoDetectWidget()
        layout.addWidget(self.auto_detect_widget)

    def apply_dark_theme(self):
        """Apply FACEIT-style dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f0f;
                color: white;
            }
            QWidget {
                background-color: #0f0f0f;
                color: white;
            }
        """)


def main():
    """Main function."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better appearance
    
    # Print startup information
    print("🏈 SpygateAI Desktop - Real Detection Mode")
    print(f"🔍 Detection available: {REAL_DETECTION_AVAILABLE}")
    
    window = SpygateDesktopAppReal()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 