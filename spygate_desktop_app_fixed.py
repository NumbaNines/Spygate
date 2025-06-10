#!/usr/bin/env python3

"""
SpygateAI Desktop Application - Fixed Version
============================================

Fixed version addressing the 30% freeze issue with improved threading and error handling.
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

    # Try to import core modules, but continue if they fail
    try:
        from spygate.core.hardware import HardwareDetector, HardwareTier
        print("✅ Hardware detector imported")
    except ImportError:
        print("⚠️ Hardware detector not available, using fallback")
        HardwareDetector = None
        
    print("🏈 SpygateAI Desktop - Fixed Version Loading...")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python")
    sys.exit(1)

@dataclass
class ClipData:
    """Data structure for detected clips."""
    start_frame: int
    end_frame: int
    situation: str
    confidence: float
    timestamp: str
    approved: bool = False

class FixedAutoClipDetector(QObject):
    """Fixed auto-clip detection engine with improved error handling."""
    
    clip_detected = pyqtSignal(ClipData)
    analysis_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(list)
    analysis_error = pyqtSignal(str)  # New error signal
    
    def __init__(self):
        super().__init__()
        self.is_analyzing = False
        self.detected_clips = []
        self.config = {
            "frame_skip": 30,
            "max_clips_per_minute": 5,
            "analysis_resolution": (1280, 720)
        }
        
    def setup_optimization_settings(self, hardware_tier: str):
        """Configure optimization settings based on hardware tier."""
        optimization_configs = {
            "low": {
                "frame_skip": 90,
                "max_clips_per_minute": 2,
                "analysis_resolution": (640, 360),
            },
            "medium": {
                "frame_skip": 60,
                "max_clips_per_minute": 3,
                "analysis_resolution": (854, 480),
            },
            "high": {
                "frame_skip": 30,
                "max_clips_per_minute": 5,
                "analysis_resolution": (1280, 720),
            },
            "ultra": {
                "frame_skip": 15,
                "max_clips_per_minute": 8,
                "analysis_resolution": (1920, 1080),
            },
        }
        
        self.config = optimization_configs.get(
            hardware_tier.lower(), optimization_configs["medium"]
        )
        print(f"🚀 Auto-clip detection optimized for {hardware_tier} tier")
        
    def analyze_video(self, video_path: str, hardware_tier: str = "medium"):
        """Analyze video with improved error handling and progress tracking."""
        self.is_analyzing = True
        self.detected_clips = []
        self.setup_optimization_settings(hardware_tier)
        
        try:
            print(f"🎬 Starting analysis: {video_path}")
            
            # Test video file opening with better error handling
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                error_msg = f"Cannot open video file: {video_path}"
                print(f"❌ {error_msg}")
                self.analysis_error.emit(error_msg)
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or fps <= 0:
                error_msg = f"Invalid video properties: frames={total_frames}, fps={fps}"
                print(f"❌ {error_msg}")
                cap.release()
                self.analysis_error.emit(error_msg)
                return
                
            print(f"📹 Video: {total_frames} frames at {fps} FPS")
            
            frame_count = 0
            clips_detected = 0
            last_progress = -1
            
            while cap.isOpened() and self.is_analyzing:
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                    
                frame_count += 1
                
                # Calculate progress
                progress = int((frame_count / total_frames) * 100)
                
                # Only emit progress updates when progress changes (avoid spam)
                if progress != last_progress:
                    last_progress = progress
                    message = f"Analyzing frame {frame_count}/{total_frames}"
                    
                    # Use QTimer.singleShot for thread-safe signal emission
                    QTimer.singleShot(0, lambda p=progress, m=message: self.analysis_progress.emit(p, m))
                    
                    print(f"🔄 Progress: {progress}% (Frame {frame_count})")
                    
                    # Add extra logging around the 30% mark
                    if 25 <= progress <= 35:
                        print(f"🎯 CRITICAL ZONE: {progress}% - Monitoring for issues...")
                        
                # Process frames based on skip rate
                if frame_count % self.config["frame_skip"] == 0:
                    try:
                        # Simulate situation detection (replace with actual YOLO logic)
                        situation = self._simulate_situation_detection(frame_count, fps, frame)
                        
                        if situation:
                            clip_data = ClipData(
                                start_frame=max(0, frame_count - 150),
                                end_frame=min(total_frames, frame_count + 150),
                                situation=situation,
                                confidence=0.85 + (np.random.random() * 0.1),
                                timestamp=self._frame_to_timestamp(frame_count, fps),
                            )
                            
                            self.detected_clips.append(clip_data)
                            clips_detected += 1
                            
                            # Use QTimer.singleShot for thread-safe signal emission
                            QTimer.singleShot(0, lambda c=clip_data: self.clip_detected.emit(c))
                            
                            print(f"🎬 Clip detected: {situation} at {clip_data.timestamp}")
                            
                    except Exception as e:
                        print(f"⚠️ Warning: Frame processing error at {progress}%: {e}")
                        # Continue processing instead of stopping
                        
                # Yield control periodically to prevent UI freeze
                if frame_count % 100 == 0:
                    time.sleep(0.001)  # Very small delay to yield control
                    
            cap.release()
            
            print(f"✅ Analysis complete: {clips_detected} clips detected")
            
            # Use QTimer.singleShot for thread-safe signal emission
            QTimer.singleShot(0, lambda clips=self.detected_clips: self.analysis_complete.emit(clips))
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Use QTimer.singleShot for thread-safe error emission
            QTimer.singleShot(0, lambda msg=error_msg: self.analysis_error.emit(msg))
            
        finally:
            self.is_analyzing = False
            print("🏁 Analysis thread finished")
            
    def _simulate_situation_detection(self, frame_count: int, fps: float, frame: np.ndarray) -> Optional[str]:
        """Simulate situation detection."""
        situations = [
            "3rd & Long",
            "Red Zone Opportunity", 
            "Turnover",
            "Scoring Play",
            "Defensive Stop",
            "Big Play"
        ]
        
        # Simple detection logic - replace with actual YOLO implementation
        if frame_count % 300 == 0:  # Detect every 10 seconds at 30fps
            if np.random.random() > 0.6:
                return np.random.choice(situations)
        return None
        
    def _frame_to_timestamp(self, frame: int, fps: float) -> str:
        """Convert frame number to timestamp."""
        seconds = frame / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def stop_analysis(self):
        """Stop the analysis process."""
        print("🛑 Stopping analysis...")
        self.is_analyzing = False

class VideoDropZone(QLabel):
    """Improved drag & drop zone with better error handling."""
    
    video_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()
        
    def init_ui(self):
        self.setText("🎮 Drop Video Here\n\nSupported: MP4, MOV, AVI\nMax size: 2GB")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #ff6b35;
                border-radius: 12px;
                background-color: #1a1a1a;
                color: #888;
                font-size: 16px;
                padding: 40px;
                min-height: 200px;
            }
            QLabel:hover {
                border-color: #ff8c42;
                background-color: #222;
            }
        """)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and self._is_video_file(urls[0].toLocalFile()):
                event.accept()
                self.setStyleSheet(self.styleSheet() + """
                    QLabel {
                        border-color: #4CAF50 !important;
                        background-color: #1b2a1b !important;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.init_ui()  # Reset style
        
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self._is_video_file(file_path):
                self.video_dropped.emit(file_path)
                self.setText(f"✅ Loaded: {Path(file_path).name}")
            else:
                self.setText("❌ Unsupported file type")
                QTimer.singleShot(2000, self.init_ui)  # Reset after 2 seconds
        self.init_ui()  # Reset drag style
        
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.video_dropped.emit(file_path)
            
    def _is_video_file(self, file_path: str) -> bool:
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return Path(file_path).suffix.lower() in video_extensions

class FixedAutoDetectWidget(QWidget):
    """Fixed auto-detection widget with improved error handling."""
    
    def __init__(self):
        super().__init__()
        self.current_video = None
        self.hardware_tier = "medium"
        self.auto_detector = FixedAutoClipDetector()
        self.analysis_thread = None
        
        # Detect hardware
        self.detect_hardware()
        
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("🎬 Auto-Clip Detection")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: white; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Drop zone
        self.drop_zone = VideoDropZone()
        layout.addWidget(self.drop_zone)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("🔍 Start Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover:enabled {
                background-color: #e55a2b;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover:enabled {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
        """)
        
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #222;
            }
            QProgressBar::chunk {
                background-color: #ff6b35;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to analyze video")
        self.status_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        # Clips area
        clips_label = QLabel("🎬 Detected Clips")
        clips_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        clips_label.setStyleSheet("color: white; margin-top: 20px;")
        layout.addWidget(clips_label)
        
        self.clips_scroll = QScrollArea()
        self.clips_scroll.setWidgetResizable(True)
        self.clips_scroll.setMaximumHeight(300)
        self.clips_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #444;
                border-radius: 5px;
                background-color: #2a2a2a;
            }
        """)
        
        self.clips_widget = QWidget()
        self.clips_layout = QVBoxLayout()
        self.clips_layout.addStretch()
        self.clips_widget.setLayout(self.clips_layout)
        self.clips_scroll.setWidget(self.clips_widget)
        layout.addWidget(self.clips_scroll)
        
        self.setLayout(layout)
        
    def connect_signals(self):
        self.drop_zone.video_dropped.connect(self.load_video)
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        
        # Connect detector signals
        self.auto_detector.clip_detected.connect(self.add_clip_card)
        self.auto_detector.analysis_progress.connect(self.update_progress)
        self.auto_detector.analysis_complete.connect(self.analysis_finished)
        self.auto_detector.analysis_error.connect(self.analysis_error)
        
    def detect_hardware(self):
        """Detect hardware and set appropriate tier."""
        try:
            if HardwareDetector:
                detector = HardwareDetector()
                tier_enum = detector.tier
                
                tier_map = {
                    HardwareTier.ULTRA: "ultra",
                    HardwareTier.HIGH: "high", 
                    HardwareTier.MEDIUM: "medium",
                    HardwareTier.LOW: "low",
                    HardwareTier.ULTRA_LOW: "low",
                }
                
                self.hardware_tier = tier_map.get(tier_enum, "medium")
                print(f"Hardware detected: {tier_enum.name} tier -> {self.hardware_tier}")
            else:
                self.hardware_tier = "medium"
                print("Hardware detection not available, using medium tier")
                
        except Exception as e:
            print(f"Hardware detection failed: {e}")
            self.hardware_tier = "medium"
            
    def load_video(self, file_path: str):
        """Load video file."""
        self.current_video = file_path
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Video loaded: {Path(file_path).name}")
        print(f"📹 Video loaded: {file_path}")
        
    def start_analysis(self):
        """Start auto-clip detection analysis."""
        if not self.current_video:
            return
            
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis...")
        
        # Clear previous clips
        self.clear_clips()
        
        # Start analysis in thread with better error handling
        self.analysis_thread = threading.Thread(
            target=self.auto_detector.analyze_video,
            args=(self.current_video, self.hardware_tier),
            daemon=True  # Important: make thread daemon
        )
        self.analysis_thread.start()
        print(f"🚀 Analysis started for {self.hardware_tier} tier")
        
    def stop_analysis(self):
        """Stop analysis."""
        self.auto_detector.stop_analysis()
        self.stop_btn.setEnabled(False)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis stopped")
        
    def update_progress(self, value: int, message: str):
        """Update analysis progress."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def analysis_finished(self, clips: List[ClipData]):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Analysis complete! Found {len(clips)} key moments")
        print(f"✅ Analysis complete: {len(clips)} clips detected")
        
    def analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Analysis failed: {error_message}")
        
        # Show error dialog
        QMessageBox.critical(self, "Analysis Error", f"Video analysis failed:\n\n{error_message}")
        
    def add_clip_card(self, clip_data: ClipData):
        """Add a new clip card to the panel."""
        card = QLabel(f"🎬 {clip_data.situation} ({clip_data.timestamp}) - {clip_data.confidence:.1%}")
        card.setStyleSheet("""
            QLabel {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
                margin: 2px;
                color: white;
            }
        """)
        
        # Insert before the stretch
        self.clips_layout.insertWidget(self.clips_layout.count() - 1, card)
        
    def clear_clips(self):
        """Clear all clip cards."""
        while self.clips_layout.count() > 1:  # Keep the stretch
            child = self.clips_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class FixedSpygateDesktopApp(QMainWindow):
    """Fixed main desktop application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop - Fixed Version")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.apply_dark_theme()
        
    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Auto-detect widget
        self.auto_detect_widget = FixedAutoDetectWidget()
        layout.addWidget(self.auto_detect_widget)
        
        central_widget.setLayout(layout)
        
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f0f;
                color: white;
            }
            QWidget {
                background-color: #1a1a1a;
                color: white;
            }
        """)

def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SpygateAI Desktop - Fixed")
    app.setApplicationVersion("1.0.1")
    app.setOrganizationName("SpygateAI")
    
    # Apply global dark theme
    app.setStyle("Fusion")
    
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(255, 107, 53))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)
    
    # Create and show main window
    window = FixedSpygateDesktopApp()
    window.show()
    
    print("🏈 SpygateAI Desktop - Fixed Version Ready!")
    print("💡 This version addresses the 30% freeze issue with:")
    print("  - Improved threading with daemon threads")
    print("  - Thread-safe signal emission using QTimer.singleShot")
    print("  - Better error handling and recovery")
    print("  - Progress update optimization")
    print("  - Stop button functionality")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main()) 