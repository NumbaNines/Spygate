#!/usr/bin/env python3

"""
SpygateAI Desktop Application - Production Version
================================================

Production-ready PyQt6 desktop application integrating all core modules
with auto-clip detection workflow and FACEIT-style interface.
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

    from spygate.core.game_detector import GameDetector
    from spygate.core.gpu_memory_manager import AdvancedGPUMemoryManager

    # Import core modules
    from spygate.core.hardware import HardwareDetector, HardwareTier
    from spygate.core.optimizer import TierOptimizer
    from spygate.core.performance_monitor import PerformanceMonitor

    # Import real detection classes
    from spygate.ml.situation_detector import SituationDetector
    from spygate.ml.hud_detector import HUDDetector

    print("🏈 SpygateAI Desktop - Initializing Core Systems...")
    REAL_DETECTION_AVAILABLE = True

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python ultralytics")
    sys.exit(1)
    REAL_DETECTION_AVAILABLE = False


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


class AutoClipDetector(QObject):
    """Auto-clip detection using real SituationDetector."""

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
            self._initialize_real_detection()
        else:
            print("⚠️ Real detection unavailable - using fallback mode")

    def _initialize_real_detection(self):
        """Initialize the real detection components."""
        try:
            # Initialize HUD detector
            self.hud_detector = HUDDetector()
            print("✅ HUDDetector initialized")

            # Initialize situation detector with enhanced capabilities
            self.situation_detector = SituationDetector()
            if hasattr(self.situation_detector, "initialize"):
                self.situation_detector.initialize()
            print("✅ SituationDetector initialized")

        except Exception as e:
            print(f"❌ Error initializing real detection: {e}")
            self.situation_detector = None
            self.hud_detector = None

    def setup_optimization_config(self, hardware_tier: str):
        """Configure optimization settings based on hardware tier."""
        if not REAL_DETECTION_AVAILABLE:
            # Fallback configuration
            self.config = {
                "frame_skip": 30,
                "target_resolution": (640, 360),
                "confidence_threshold": 0.5,
                "max_clips_per_minute": 2,
            }
            return

        # Get tier-specific configuration
        tier_configs = {
            "ULTRA": {
                "frame_skip": 15,
                "target_resolution": (1920, 1080),
                "confidence_threshold": 0.6,
                "max_clips_per_minute": 5,
            },
            "HIGH": {
                "frame_skip": 30,
                "target_resolution": (1280, 720),
                "confidence_threshold": 0.55,
                "max_clips_per_minute": 4,
            },
            "MEDIUM": {
                "frame_skip": 60,
                "target_resolution": (854, 480),
                "confidence_threshold": 0.5,
                "max_clips_per_minute": 3,
            },
            "LOW": {
                "frame_skip": 90,
                "target_resolution": (640, 360),
                "confidence_threshold": 0.45,
                "max_clips_per_minute": 2,
            },
        }

        self.config = tier_configs.get(hardware_tier, tier_configs["MEDIUM"])
        print(f"🔧 Optimization configured for {hardware_tier} tier")
        print(f"   Frame skip: {self.config['frame_skip']}")
        print(f"   Target resolution: {self.config['target_resolution']}")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for faster analysis."""
        target_res = self.config.get("target_resolution", (640, 360))
        if frame.shape[:2][::-1] != target_res:
            frame = cv2.resize(frame, target_res)
        return frame

    def analyze_video(self, video_path: str, hardware_tier: str):
        """Analyze video with real situation detection."""
        if self.is_analyzing:
            return

        self.is_analyzing = True
        self.detected_clips = []
        self.setup_optimization_config(hardware_tier)

        def analysis_thread():
            self._run_real_analysis(video_path, hardware_tier)

        thread = threading.Thread(target=analysis_thread, daemon=True)
        thread.start()

    def _run_real_analysis(self, video_path: str, hardware_tier: str):
        """Run the actual video analysis with real detection."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return

            # Video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"📹 Video: {frame_width}x{frame_height}, {total_frames} frames at {fps:.1f} FPS")
            
            # Analysis variables
            frame_count = 0
            frames_processed = 0
            frames_skipped = 0
            clips_detected = 0
            scene_changes_detected = 0
            last_histogram = None

            start_time = time.time()

            while frame_count < total_frames and self.is_analyzing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Scene change detection (histogram-based)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

                scene_change = False
                if last_histogram is not None:
                    correlation = cv2.compareHist(last_histogram, hist, cv2.HISTCMP_CORREL)
                    if correlation < 0.8:  # Scene change threshold
                        scene_change = True
                        scene_changes_detected += 1

                last_histogram = hist

                # Smart frame skipping with scene change consideration
                should_process = (
                    frame_count % self.config["frame_skip"] == 0
                    or scene_change
                    or frame_count == 1
                    or frame_count == total_frames
                )

                if not should_process:
                    frames_skipped += 1
                    continue

                frames_processed += 1

                # Preprocess frame for faster analysis
                processed_frame = self.preprocess_frame(frame)

                # Update progress
                progress = int((frame_count / total_frames) * 100)
                self.analysis_progress.emit(
                    progress,
                    f"Analyzing frame {frame_count}/{total_frames} (processed: {frames_processed}, skipped: {frames_skipped})",
                )

                # Real situation detection
                situation_result = self._analyze_frame_with_real_detection(
                    processed_frame, frame_count, fps
                )

                if situation_result and self._is_significant_situation(situation_result):
                    # Create clip data with real detection results
                    clip_data = ClipData(
                        start_frame=max(0, frame_count - 150),
                        end_frame=min(total_frames, frame_count + 150),
                        situation=situation_result.get("situation", "Unknown"),
                        confidence=situation_result.get("confidence", 0.0),
                        timestamp=self._frame_to_timestamp(frame_count, fps),
                        hud_info=situation_result.get("hud_info", {}),
                        detection_metadata=situation_result.get("metadata", {}),
                    )

                    self.detected_clips.append(clip_data)
                    clips_detected += 1

                    self.clip_detected.emit(clip_data)
                    print(f"🎬 Real Clip detected: {clip_data.situation} at {clip_data.timestamp} (confidence: {clip_data.confidence:.2f})")

            cap.release()

            # Analysis complete
            end_time = time.time()
            processing_time = end_time - start_time
            processing_rate = frames_processed / processing_time if processing_time > 0 else 0

            print(f"✅ Real Analysis complete in {processing_time:.2f}s")
            print(f"  Frames processed: {frames_processed} of {total_frames} ({frames_processed/total_frames*100:.1f}%)")
            print(f"  Frames skipped: {frames_skipped} ({frames_skipped/total_frames*100:.1f}%)")
            print(f"  Scene changes detected: {scene_changes_detected}")
            print(f"  Real clips detected: {clips_detected}")
            print(f"  Processing rate: {processing_rate:.1f} frames/sec")

            self.analysis_complete.emit(self.detected_clips)

        except Exception as e:
            print(f"❌ Error in real video analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_analyzing = False

    def _analyze_frame_with_real_detection(
        self, frame: np.ndarray, frame_count: int, fps: float
    ) -> Optional[Dict[str, Any]]:
        """Analyze frame using real situation detection."""
        if not REAL_DETECTION_AVAILABLE or not self.situation_detector:
            # Fallback to enhanced simulation
            return self._enhanced_fallback_detection(frame, frame_count, fps)

        try:
            # REAL SITUATION DETECTION INTEGRATION
            print(f"🔍 Running real detection on frame {frame_count}")
            
            result = self.situation_detector.detect_situations(frame, frame_count, fps)

            if result and result.get("situations"):
                situations = result["situations"]
                
                # Filter for high-confidence situations
                high_confidence_situations = [
                    s for s in situations 
                    if s.get("confidence", 0) >= self.config.get("confidence_threshold", 0.5)
                ]
                
                if high_confidence_situations:
                    # Get the best situation
                    best_situation = max(high_confidence_situations, key=lambda x: x.get("confidence", 0))
                    
                    # Format situation type for display
                    situation_type = best_situation.get("type", "Unknown")
                    situation_display = self._format_situation_for_display(situation_type, result.get("hud_info", {}))

                    return {
                        "situation": situation_display,
                        "confidence": best_situation.get("confidence", 0.0),
                        "hud_info": result.get("hud_info", {}),
                        "metadata": result.get("metadata", {}),
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "raw_situations": situations,  # Include all detected situations
                    }

            return None

        except Exception as e:
            print(f"⚠️ Real detection error on frame {frame_count}: {e}")
            return self._enhanced_fallback_detection(frame, frame_count, fps)

    def _format_situation_for_display(self, situation_type: str, hud_info: Dict[str, Any]) -> str:
        """Format situation type for user-friendly display."""
        # Extract HUD data
        down = hud_info.get("down")
        distance = hud_info.get("distance")
        yard_line = hud_info.get("yard_line") or hud_info.get("field_position")
        score_home = hud_info.get("score_home")
        score_away = hud_info.get("score_away")
        game_clock = hud_info.get("game_clock")

        # Format based on situation type
        if situation_type == "third_and_long":
            if down and distance:
                return f"3rd & {distance} (Real HUD)"
            return "3rd & Long (Real HUD)"
        elif situation_type == "third_and_short":
            if down and distance:
                return f"3rd & {distance} (Real HUD)"
            return "3rd & Short (Real HUD)"
        elif situation_type == "fourth_down":
            if down and distance:
                return f"4th & {distance} (Real HUD)"
            return "4th Down (Real HUD)"
        elif situation_type == "red_zone":
            return f"Red Zone (Real HUD)"
        elif situation_type == "two_minute_warning":
            return f"Two Minute Warning (Real HUD)"
        elif "goal" in situation_type.lower():
            return f"Goal Line (Real HUD)"
        elif "close_game" in situation_type:
            if score_home is not None and score_away is not None:
                return f"Close Game {score_home}-{score_away} (Real HUD)"
            return "Close Game (Real HUD)"
        else:
            # Generic formatting
            formatted = situation_type.replace("_", " ").title()
            return f"{formatted} (Real HUD)"

    def _enhanced_fallback_detection(
        self, frame: np.ndarray, frame_count: int, fps: float
    ) -> Optional[Dict[str, Any]]:
        """Enhanced fallback detection when real detection is unavailable."""
        # Calculate frame variance for action detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_variance = np.var(gray)

        situations = [
            "1st & 10 (Simulated)",
            "2nd & 5 (Simulated)",
            "3rd & Long (Simulated)",
            "4th & Goal (Simulated)",
            "Red Zone Opportunity (Simulated)",
            "Two Minute Warning (Simulated)",
            "Turnover (Simulated)",
            "Scoring Play (Simulated)",
            "Defensive Stop (Simulated)",
        ]

        # Enhanced detection logic based on frame characteristics
        if frame_variance > 1000:  # High action frame
            if np.random.random() > 0.6:  # Moderate chance for high-action frames
                situation = np.random.choice(situations)
                return {
                    "situation": situation,
                    "confidence": 0.7 + (np.random.random() * 0.2),
                    "hud_info": {"detection_mode": "fallback", "frame_variance": float(frame_variance)},
                    "metadata": {"analysis_version": "fallback-enhanced"},
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                }
        elif frame_count % 300 == 0:  # Periodic detection for static frames
            if np.random.random() > 0.7:
                situation = np.random.choice(situations)
                return {
                    "situation": situation,
                    "confidence": 0.6 + (np.random.random() * 0.25),
                    "hud_info": {"detection_mode": "fallback", "frame_variance": float(frame_variance)},
                    "metadata": {"analysis_version": "fallback-periodic"},
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                }

        return None

    def _is_significant_situation(self, situation_result: Dict[str, Any]) -> bool:
        """Determine if situation is significant enough for clipping."""
        if not situation_result:
            return False

        confidence = situation_result.get("confidence", 0.0)
        situation = situation_result.get("situation", "")

        # High-value situations (lower confidence threshold)
        high_value_keywords = ["3rd", "4th", "Red Zone", "Two Minute", "Turnover", "Scoring", "Goal"]

        if any(keyword in situation for keyword in high_value_keywords):
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


class VideoDropZone(QLabel):
    """FACEIT-style drag & drop zone."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        self.setText("🎮 Drop Video Here\n\nSupported: MP4, MOV, AVI\nMax size: 2GB")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #ff6b35;
                border-radius: 12px;
                background-color: #1a1a1a;
                color: #888;
                font-size: 16px;
                font-weight: bold;
                min-height: 200px;
                padding: 20px;
            }
            QLabel:hover {
                background-color: #2a2a2a;
                border-color: #ff8b55;
                color: #aaa;
            }
        """
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self._is_video_file(urls[0].toLocalFile()):
                event.accept()
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 12px;
                        background-color: #0a3d2e;
                        color: #10B981;
                        font-size: 16px;
                        font-weight: bold;
                        min-height: 200px;
                        padding: 20px;
                    }
                """
                )
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.init_ui()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self._is_video_file(file_path):
                self.video_dropped.emit(file_path)
                self.setText(f"✅ Ready to Analyze\n\n{Path(file_path).name}")
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 12px;
                        background-color: #0a3d2e;
                        color: #10B981;
                        font-size: 16px;
                        font-weight: bold;
                        min-height: 200px;
                        padding: 20px;
                    }
                """
                )

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi)"
        )
        if file_path:
            self.video_dropped.emit(file_path)

    def _is_video_file(self, file_path: str) -> bool:
        return file_path.lower().endswith((".mp4", ".mov", ".avi"))


class ClipCard(QWidget):
    """Individual clip card widget."""

    approve_clicked = pyqtSignal(ClipData)
    reject_clicked = pyqtSignal(ClipData)

    def __init__(self, clip_data: ClipData):
        super().__init__()
        self.clip_data = clip_data
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(280, 160)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 1px solid #444;
            }
            QWidget:hover {
                border-color: #ff6b35;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header_layout = QHBoxLayout()

        situation_label = QLabel(self.clip_data.situation)
        situation_label.setStyleSheet(
            """
            QLabel {
                color: #ff6b35;
                font-weight: bold;
                font-size: 14px;
                background: transparent;
                border: none;
            }
        """
        )
        header_layout.addWidget(situation_label)

        confidence_label = QLabel(f"{self.clip_data.confidence:.0%}")
        confidence_label.setStyleSheet(
            """
            QLabel {
                color: #10B981;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """
        )
        header_layout.addWidget(confidence_label)

        layout.addLayout(header_layout)

        # Timestamp
        time_label = QLabel(f"⏱️ {self.clip_data.timestamp}")
        time_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(time_label)

        # Duration
        duration = (self.clip_data.end_frame - self.clip_data.start_frame) / 30  # Assume 30fps
        duration_label = QLabel(f"📏 {duration:.1f}s")
        duration_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(duration_label)

        layout.addStretch()

        # Action buttons
        button_layout = QHBoxLayout()

        approve_btn = QPushButton("✅ Keep")
        approve_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """
        )
        approve_btn.clicked.connect(lambda: self.approve_clicked.emit(self.clip_data))

        reject_btn = QPushButton("❌ Remove")
        reject_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """
        )
        reject_btn.clicked.connect(lambda: self.reject_clicked.emit(self.clip_data))

        button_layout.addWidget(approve_btn)
        button_layout.addWidget(reject_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)


class SidebarWidget(QWidget):
    """FACEIT-style sidebar navigation."""

    tab_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_tab = "auto-detect"
        self.init_ui()

    def init_ui(self):
        self.setFixedWidth(280)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-right: 1px solid #333;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Navigation
        nav_widget = self._create_navigation()
        layout.addWidget(nav_widget)

        # Hardware status
        status_widget = self._create_hardware_status()
        layout.addWidget(status_widget)

        self.setLayout(layout)

    def _create_header(self):
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #333;
            }
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)

        logo = QLabel("🏈")
        logo.setFont(QFont("Arial", 24))
        logo.setStyleSheet("color: #ff6b35; background: transparent;")

        title = QLabel("SpygateAI")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")

        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addStretch()

        header.setLayout(layout)
        return header

    def _create_navigation(self):
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 20, 0, 20)
        nav_layout.setSpacing(4)

        nav_items = [
            ("▶️", "Auto-Detect", "auto-detect"),
            ("📚", "Library", "library"),
            ("📊", "Analysis", "analysis"),
            ("🔍", "Search", "search"),
            ("📤", "Export", "export"),
            ("⚙️", "Settings", "settings"),
        ]

        for icon, text, action in nav_items:
            btn = self._create_nav_button(icon, text, action)
            nav_layout.addWidget(btn)

        nav_layout.addStretch()
        nav_widget.setLayout(nav_layout)
        return nav_widget

    def _create_nav_button(self, icon: str, text: str, action: str):
        btn = QPushButton(f"{icon}  {text}")
        btn.setFixedHeight(48)
        is_active = action == self.current_tab

        if is_active:
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #ff6b35;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    text-align: left;
                    padding-left: 20px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """
            )
        else:
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: transparent;
                    color: #888;
                    border: none;
                    text-align: left;
                    padding-left: 20px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #2a2a2a;
                    color: #fff;
                }
            """
            )

        btn.clicked.connect(lambda: self.tab_changed.emit(action))
        return btn

    def _create_hardware_status(self):
        status_widget = QWidget()
        status_widget.setFixedHeight(120)
        status_widget.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                border-top: 1px solid #333;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)

        # Hardware tier indicator
        tier_label = QLabel("⚡ Hardware: High Tier")
        tier_label.setStyleSheet(
            """
            QLabel {
                color: #10B981;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(tier_label)

        # Performance indicator
        perf_label = QLabel("🎯 Optimized for YOLOv8")
        perf_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                font-size: 11px;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(perf_label)

        layout.addStretch()

        # Pro upgrade button
        upgrade_btn = QPushButton("⚡ Upgrade to Pro")
        upgrade_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """
        )
        layout.addWidget(upgrade_btn)

        status_widget.setLayout(layout)
        return status_widget


class AutoDetectWidget(QWidget):
    """Main auto-detect interface."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.detected_clips = []
        self.auto_detector = AutoClipDetector()
        self.hardware_detector = HardwareDetector()
        self.hardware_tier = "medium"  # Default
        self.init_ui()
        self.connect_signals()
        self.detect_hardware()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        title = QLabel("🎮 Auto-Clip Detection")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Drop a video file and let SpygateAI automatically detect key moments")
        desc.setStyleSheet("color: #888; font-size: 14px; background: transparent;")
        layout.addWidget(desc)

        # Main content area
        content_layout = QHBoxLayout()

        # Left side - Drop zone and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)

        # Drop zone
        self.drop_zone = VideoDropZone()
        left_layout.addWidget(self.drop_zone)

        # Analysis controls
        controls_widget = self._create_controls()
        left_layout.addWidget(controls_widget)

        left_panel.setLayout(left_layout)
        content_layout.addWidget(left_panel, 1)

        # Right side - Detected clips
        right_panel = self._create_clips_panel()
        content_layout.addWidget(right_panel, 1)

        layout.addLayout(content_layout)
        self.setLayout(layout)

    def _create_controls(self):
        controls = QWidget()
        controls.setStyleSheet(
            """
            QWidget {
                background-color: #2a2a2a;
                border-radius: 8px;
                padding: 15px;
            }
        """
        )

        layout = QVBoxLayout()

        # Hardware status
        hardware_label = QLabel(f"⚡ Hardware Tier: {self.hardware_tier.title()}")
        hardware_label.setStyleSheet(
            """
            QLabel {
                color: #10B981;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(hardware_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #444;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #ff6b35;
                border-radius: 3px;
            }
        """
        )
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                font-size: 12px;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(self.status_label)

        # Analyze button
        self.analyze_btn = QPushButton("🔍 Start Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
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
        """
        )
        self.analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_btn)

        controls.setLayout(layout)
        return controls

    def _create_clips_panel(self):
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: #2a2a2a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QLabel("🎬 Detected Clips")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: white; background: transparent; border: none;")
        layout.addWidget(header)

        # Clips scroll area
        self.clips_scroll = QScrollArea()
        self.clips_scroll.setWidgetResizable(True)
        self.clips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.clips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.clips_scroll.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #444;
                border-radius: 4px;
            }
        """
        )

        self.clips_widget = QWidget()
        self.clips_layout = QVBoxLayout()
        self.clips_layout.setSpacing(10)
        self.clips_layout.addStretch()
        self.clips_widget.setLayout(self.clips_layout)
        self.clips_scroll.setWidget(self.clips_widget)

        layout.addWidget(self.clips_scroll)
        panel.setLayout(layout)
        return panel

    def connect_signals(self):
        self.drop_zone.video_dropped.connect(self.load_video)
        self.auto_detector.clip_detected.connect(self.add_clip_card)
        self.auto_detector.analysis_progress.connect(self.update_progress)
        self.auto_detector.analysis_complete.connect(self.analysis_finished)

    def detect_hardware(self):
        """Detect hardware and set appropriate tier."""
        try:
            # Use actual hardware detection
            tier_enum = self.hardware_detector.tier

            # Convert enum to string for compatibility
            tier_map = {
                HardwareTier.ULTRA: "ultra",
                HardwareTier.HIGH: "high",
                HardwareTier.MEDIUM: "medium",
                HardwareTier.LOW: "low",
                HardwareTier.ULTRA_LOW: "low",
            }

            self.hardware_tier = tier_map.get(tier_enum, "medium")

            print(f"Hardware detected: {tier_enum.name} tier -> {self.hardware_tier}")

        except Exception as e:
            print(f"Hardware detection failed: {e}")
            self.hardware_tier = "medium"

    def load_video(self, file_path: str):
        """Load video file."""
        self.current_video = file_path
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Video loaded: {Path(file_path).name}")

    def start_analysis(self):
        """Start auto-clip detection analysis."""
        if not self.current_video:
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analyzing video...")

        # Clear previous clips
        self.clear_clips()

        # Start analysis in thread
        self.analysis_thread = threading.Thread(
            target=self.auto_detector.analyze_video, args=(self.current_video, self.hardware_tier)
        )
        self.analysis_thread.start()

    def update_progress(self, value: int, message: str):
        """Update analysis progress."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def analysis_finished(self, clips: list[ClipData]):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"Analysis complete! Found {len(clips)} key moments")

    def add_clip_card(self, clip_data: ClipData):
        """Add a new clip card to the panel."""
        card = ClipCard(clip_data)
        card.approve_clicked.connect(self.approve_clip)
        card.reject_clicked.connect(self.reject_clip)

        # Insert before the stretch
        self.clips_layout.insertWidget(self.clips_layout.count() - 1, card)

    def approve_clip(self, clip_data: ClipData):
        """Approve and save clip."""
        clip_data.approved = True
        # TODO: Save clip to library
        self.status_label.setText(f"Clip saved: {clip_data.situation}")

    def reject_clip(self, clip_data: ClipData):
        """Reject and remove clip."""
        # Find and remove the card
        for i in range(self.clips_layout.count()):
            item = self.clips_layout.itemAt(i)
            if item and isinstance(item.widget(), ClipCard):
                if item.widget().clip_data == clip_data:
                    item.widget().deleteLater()
                    break

    def clear_clips(self):
        """Clear all clip cards."""
        while self.clips_layout.count() > 1:  # Keep the stretch
            child = self.clips_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


class SpygateDesktopApp(QMainWindow):
    """Main desktop application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop - Auto-Clip Detection")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 800)
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.tab_changed.connect(self.switch_tab)
        main_layout.addWidget(self.sidebar)

        # Content area
        self.content_stack = QStackedWidget()

        # Add tab widgets
        self.auto_detect_widget = AutoDetectWidget()
        self.content_stack.addWidget(self.auto_detect_widget)

        # Placeholder widgets for other tabs
        for tab_name in ["library", "analysis", "search", "export", "settings"]:
            placeholder = self.create_placeholder(tab_name)
            self.content_stack.addWidget(placeholder)

        main_layout.addWidget(self.content_stack)
        central_widget.setLayout(main_layout)

    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f0f0f;
                color: white;
            }
        """
        )

    def create_placeholder(self, tab_name: str):
        """Create placeholder widget for future tabs."""
        widget = QWidget()
        layout = QVBoxLayout()

        icon_map = {
            "library": "📚",
            "analysis": "📊",
            "search": "🔍",
            "export": "📤",
            "settings": "⚙️",
        }

        icon = QLabel(icon_map.get(tab_name, "🔧"))
        icon.setFont(QFont("Arial", 48))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("color: #ff6b35; background: transparent;")

        title = QLabel(f"{tab_name.title()} - Coming Soon")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; background: transparent; margin: 20px;")

        desc = QLabel("This feature will be available in a future update")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #888; font-size: 14px; background: transparent;")

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addStretch()

        widget.setLayout(layout)
        widget.setStyleSheet("background-color: #1a1a1a;")
        return widget

    def switch_tab(self, tab_name: str):
        """Switch to the specified tab."""
        tab_map = {
            "auto-detect": 0,
            "library": 1,
            "analysis": 2,
            "search": 3,
            "export": 4,
            "settings": 5,
        }

        index = tab_map.get(tab_name, 0)
        self.content_stack.setCurrentIndex(index)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("SpygateAI Desktop")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SpygateAI")

    # Apply global dark theme
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(255, 107, 53))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

    # Create and show main window
    window = SpygateDesktopApp()
    window.show()

    print("🏈 SpygateAI Desktop - Ready for Auto-Clip Detection!")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
