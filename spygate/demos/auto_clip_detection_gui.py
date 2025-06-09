#!/usr/bin/env python3

"""
SpygateAI Auto-Clip Detection Demo
==================================

Automated workflow: Load video → Auto-detect situations → Create clips automatically, and let users approve/reject them
"""

import json
import logging
import os
import sys
import time
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

    print("🏈 SpygateAI Auto-Clip Detection Demo")
    print("====================================")
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
    print("📝 Running in demo mode with simulated detection")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoAnalysisWorker(QThread):
    """Worker thread for continuous video analysis."""

    situation_detected = pyqtSignal(dict)  # When a situation is found
    progress_update = pyqtSignal(int, int)  # Current frame, total frames
    analysis_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, sensitivity="medium"):
        super().__init__()
        self.video_path = video_path
        self.sensitivity = sensitivity
        self.should_stop = False
        self.skip_frames = self.get_skip_frames(sensitivity)

    def get_skip_frames(self, sensitivity):
        """Get frame skip based on sensitivity setting."""
        skip_map = {
            "high": 5,  # Analyze every 5th frame (very thorough)
            "medium": 15,  # Analyze every 15th frame (balanced)
            "low": 30,  # Analyze every 30th frame (fast)
        }
        return skip_map.get(sensitivity, 15)

    def stop_analysis(self):
        """Stop the analysis process."""
        self.should_stop = True

    def run(self):
        """Run continuous analysis on the video."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("Could not open video file")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = 0

            if SPYGATE_AVAILABLE:
                detector = SituationDetector()
                detector.initialize()

            logger.info(
                f"Starting analysis: {total_frames} frames, analyzing every {self.skip_frames} frames"
            )

            while frame_count < total_frames and not self.should_stop:
                # Skip to next analysis frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()

                if not ret:
                    break

                # Emit progress
                self.progress_update.emit(frame_count, total_frames)

                # Analyze frame
                if SPYGATE_AVAILABLE:
                    analysis_result = detector.detect_situations(frame, frame_count, fps)
                else:
                    analysis_result = self.generate_mock_analysis(frame_count, fps)

                # Check if any significant situations were detected
                situations = analysis_result.get("situations", [])
                if situations and self.is_significant_moment(situations):
                    # Add frame data to result
                    analysis_result["frame"] = frame.copy()
                    analysis_result["video_path"] = self.video_path
                    self.situation_detected.emit(analysis_result)

                    # Brief pause to allow UI updates
                    time.sleep(0.1)

                frame_count += self.skip_frames

            cap.release()

            if not self.should_stop:
                self.analysis_complete.emit()

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.error_occurred.emit(str(e))

    def is_significant_moment(self, situations):
        """Determine if detected situations are significant enough for clipping."""
        for situation in situations:
            confidence = situation.get("confidence", 0.0)
            sit_type = situation.get("type", "")

            # High-value situations
            if sit_type in ["third_and_long", "red_zone", "two_minute_warning", "close_game"]:
                if confidence > 0.7:  # High confidence threshold
                    return True

            # Medium-value situations need higher confidence
            elif sit_type in ["fourth_down", "goal_line", "turnover"]:
                if confidence > 0.8:
                    return True

        return False

    def generate_mock_analysis(self, frame_number, fps):
        """Generate realistic mock analysis for demo."""
        import random

        # Only generate situations occasionally to simulate real detection
        if random.random() > 0.95:  # 5% chance per frame
            situation_type = random.choice(
                [
                    "third_and_long",
                    "red_zone",
                    "two_minute_warning",
                    "fourth_down",
                    "goal_line",
                    "close_game",
                ]
            )

            return {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "situations": [
                    {
                        "type": situation_type,
                        "confidence": 0.75 + random.random() * 0.25,
                        "frame": frame_number,
                        "timestamp": frame_number / fps,
                        "details": self.generate_situation_details(situation_type, fps),
                    }
                ],
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
                    "analysis_version": "demo-auto-detect",
                },
            }
        else:
            # Return empty result (no significant situations)
            return {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "situations": [],
                "hud_info": {},
                "metadata": {},
            }

    def generate_situation_details(self, situation_type, fps):
        """Generate appropriate details for each situation type."""
        import random

        details = {"source": "hud_analysis"}

        if situation_type == "third_and_long":
            details.update(
                {
                    "down": 3,
                    "distance": random.randint(7, 15),
                    "field_position": f"OPP {random.randint(20, 45)}",
                }
            )
        elif situation_type == "red_zone":
            details.update({"field_position": f"OPP {random.randint(5, 20)}"})
        elif situation_type == "two_minute_warning":
            details.update(
                {
                    "quarter": random.choice([2, 4]),
                    "game_clock": f"{random.randint(1, 2)}:{random.randint(10, 59):02d}",
                }
            )
        elif situation_type == "fourth_down":
            details.update(
                {
                    "down": 4,
                    "distance": random.randint(1, 8),
                    "field_position": f"OPP {random.randint(15, 45)}",
                }
            )

        return details


class ClipCard(QWidget):
    """Widget representing a detected clip that user can approve/reject."""

    clip_approved = pyqtSignal(dict)
    clip_rejected = pyqtSignal(dict)
    clip_preview = pyqtSignal(dict)

    def __init__(self, analysis_data):
        super().__init__()
        self.analysis_data = analysis_data
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(320, 200)
        self.setStyleSheet(
            """
            QWidget {
                background-color: white;
                border: 2px solid #E5E7EB;
                border-radius: 8px;
                margin: 5px;
            }
            QWidget:hover {
                border-color: #3B82F6;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        # Header with timestamp
        timestamp = self.analysis_data.get("timestamp", 0)
        header = QLabel(f"⏱️ {timestamp:.1f}s")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #374151;")
        layout.addWidget(header)

        # Situations
        situations = self.analysis_data.get("situations", [])
        for situation in situations:
            sit_type = situation.get("type", "unknown")
            confidence = situation.get("confidence", 0.0)

            sit_label = QLabel(f"🎯 {sit_type.replace('_', ' ').title()}")
            sit_label.setStyleSheet(
                f"color: {self.get_situation_color(sit_type)}; font-weight: 600;"
            )
            layout.addWidget(sit_label)

            conf_label = QLabel(f"   Confidence: {confidence:.0%}")
            conf_label.setStyleSheet("color: #6B7280; font-size: 11px;")
            layout.addWidget(conf_label)

        # HUD info
        hud_info = self.analysis_data.get("hud_info", {})
        if hud_info:
            down = hud_info.get("down", "?")
            distance = hud_info.get("distance", "?")
            field_pos = hud_info.get("field_position", "Unknown")

            hud_text = f"📊 {down} & {distance}, {field_pos}"
            hud_label = QLabel(hud_text)
            hud_label.setStyleSheet("color: #4B5563; font-size: 10px;")
            layout.addWidget(hud_label)

        layout.addStretch()

        # Action buttons
        buttons_layout = QHBoxLayout()

        reject_btn = QPushButton("❌")
        reject_btn.setFixedSize(35, 30)
        reject_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #FEE2E2;
                color: #DC2626;
                border: 1px solid #FCA5A5;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #FECACA; }
        """
        )
        reject_btn.setToolTip("Reject this clip")
        reject_btn.clicked.connect(lambda: self.clip_rejected.emit(self.analysis_data))
        buttons_layout.addWidget(reject_btn)

        preview_btn = QPushButton("👁️")
        preview_btn.setFixedSize(35, 30)
        preview_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #F3F4F6;
                color: #4B5563;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #E5E7EB; }
        """
        )
        preview_btn.setToolTip("Preview clip")
        preview_btn.clicked.connect(lambda: self.clip_preview.emit(self.analysis_data))
        buttons_layout.addWidget(preview_btn)

        approve_btn = QPushButton("✅")
        approve_btn.setFixedSize(35, 30)
        approve_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #DCFCE7;
                color: #16A34A;
                border: 1px solid #86EFAC;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #BBF7D0; }
        """
        )
        approve_btn.setToolTip("Keep this clip")
        approve_btn.clicked.connect(lambda: self.clip_approved.emit(self.analysis_data))
        buttons_layout.addWidget(approve_btn)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def get_situation_color(self, sit_type):
        """Get color for situation type."""
        colors = {
            "third_and_long": "#DC2626",
            "red_zone": "#059669",
            "two_minute_warning": "#7C2D12",
            "fourth_down": "#B91C1C",
            "goal_line": "#065F46",
            "close_game": "#7C3AED",
        }
        return colors.get(sit_type, "#6B7280")


class ClipReviewWidget(QWidget):
    """Widget for reviewing detected clips."""

    def __init__(self):
        super().__init__()
        self.pending_clips = []
        self.approved_clips = []
        self.rejected_clips = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Header
        header_layout = QHBoxLayout()

        title = QLabel("📋 Auto-Detected Clips")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #374151;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Stats
        self.stats_label = QLabel("Pending: 0 | Approved: 0 | Rejected: 0")
        self.stats_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        header_layout.addWidget(self.stats_label)

        layout.addLayout(header_layout)

        # Tab widget for different clip states
        self.tabs = QTabWidget()

        # Pending clips tab
        self.pending_scroll = QScrollArea()
        self.pending_scroll.setWidgetResizable(True)
        self.pending_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pending_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.pending_widget = QWidget()
        self.pending_layout = QGridLayout(self.pending_widget)
        self.pending_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pending_scroll.setWidget(self.pending_widget)

        self.tabs.addTab(self.pending_scroll, "🔍 Pending Review")

        # Approved clips tab
        self.approved_scroll = QScrollArea()
        self.approved_scroll.setWidgetResizable(True)
        self.approved_widget = QWidget()
        self.approved_layout = QGridLayout(self.approved_widget)
        self.approved_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.approved_scroll.setWidget(self.approved_widget)

        self.tabs.addTab(self.approved_scroll, "✅ Approved")

        layout.addWidget(self.tabs)

        # Bulk actions
        bulk_layout = QHBoxLayout()

        approve_all_btn = QPushButton("✅ Approve All Pending")
        approve_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #16A34A;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #15803D; }
        """
        )
        approve_all_btn.clicked.connect(self.approve_all_pending)
        bulk_layout.addWidget(approve_all_btn)

        reject_all_btn = QPushButton("❌ Reject All Pending")
        reject_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #DC2626;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #B91C1C; }
        """
        )
        reject_all_btn.clicked.connect(self.reject_all_pending)
        bulk_layout.addWidget(reject_all_btn)

        bulk_layout.addStretch()

        export_btn = QPushButton("📤 Export Approved Clips")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563EB; }
        """
        )
        export_btn.clicked.connect(self.export_approved_clips)
        bulk_layout.addWidget(export_btn)

        layout.addLayout(bulk_layout)
        self.setLayout(layout)

    def add_detected_clip(self, analysis_data):
        """Add a newly detected clip for review."""
        clip_card = ClipCard(analysis_data)
        clip_card.clip_approved.connect(self.approve_clip)
        clip_card.clip_rejected.connect(self.reject_clip)
        clip_card.clip_preview.connect(self.preview_clip)

        # Add to pending grid
        row = len(self.pending_clips) // 3
        col = len(self.pending_clips) % 3
        self.pending_layout.addWidget(clip_card, row, col)

        self.pending_clips.append(analysis_data)
        self.update_stats()

        # Switch to pending tab to show new clip
        self.tabs.setCurrentIndex(0)

    def approve_clip(self, analysis_data):
        """Move clip from pending to approved."""
        if analysis_data in self.pending_clips:
            self.pending_clips.remove(analysis_data)
            self.approved_clips.append(analysis_data)
            self.refresh_pending_layout()
            self.refresh_approved_layout()
            self.update_stats()

            # Show success message
            QToolTip.showText(QCursor.pos(), "✅ Clip approved!", None, QRect(), 2000)

    def reject_clip(self, analysis_data):
        """Remove clip from pending (reject it)."""
        if analysis_data in self.pending_clips:
            self.pending_clips.remove(analysis_data)
            self.rejected_clips.append(analysis_data)
            self.refresh_pending_layout()
            self.update_stats()

            # Show success message
            QToolTip.showText(QCursor.pos(), "❌ Clip rejected!", None, QRect(), 2000)

    def preview_clip(self, analysis_data):
        """Show preview of the clip."""
        timestamp = analysis_data.get("timestamp", 0)
        situations = analysis_data.get("situations", [])

        preview_text = f"🎬 Clip Preview\n\n"
        preview_text += f"⏱️ Timestamp: {timestamp:.1f}s\n\n"

        if situations:
            preview_text += "🎯 Detected Situations:\n"
            for situation in situations:
                sit_type = situation.get("type", "unknown")
                confidence = situation.get("confidence", 0.0)
                preview_text += f"  • {sit_type.replace('_', ' ').title()} ({confidence:.0%})\n"

        hud_info = analysis_data.get("hud_info", {})
        if hud_info:
            preview_text += f"\n📊 Game State:\n"
            preview_text += f"  • Down & Distance: {hud_info.get('down', '?')} & {hud_info.get('distance', '?')}\n"
            preview_text += f"  • Field Position: {hud_info.get('field_position', 'Unknown')}\n"

        preview_text += f"\n💡 In the full version, this would show the actual video clip!"

        QMessageBox.information(self, "Clip Preview", preview_text)

    def approve_all_pending(self):
        """Approve all pending clips."""
        if not self.pending_clips:
            return

        count = len(self.pending_clips)
        reply = QMessageBox.question(
            self,
            "Approve All",
            f"Approve all {count} pending clips?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.approved_clips.extend(self.pending_clips)
            self.pending_clips.clear()
            self.refresh_pending_layout()
            self.refresh_approved_layout()
            self.update_stats()

    def reject_all_pending(self):
        """Reject all pending clips."""
        if not self.pending_clips:
            return

        count = len(self.pending_clips)
        reply = QMessageBox.question(
            self,
            "Reject All",
            f"Reject all {count} pending clips?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.rejected_clips.extend(self.pending_clips)
            self.pending_clips.clear()
            self.refresh_pending_layout()
            self.update_stats()

    def export_approved_clips(self):
        """Export approved clips to various formats."""
        if not self.approved_clips:
            QMessageBox.information(self, "Export", "No approved clips to export!")
            return

        formats = ["Video Clips (.mp4)", "Analysis Data (.json)", "CSV Report (.csv)"]
        format_choice, ok = QInputDialog.getItem(
            self,
            "Export Approved Clips",
            f"Export {len(self.approved_clips)} approved clips as:",
            formats,
            0,
            False,
        )

        if ok:
            QMessageBox.information(
                self,
                "Export Complete",
                f"📤 Exported {len(self.approved_clips)} clips as {format_choice}!\n\n"
                f"In the full version, this would:\n"
                f"• Create actual video clips\n"
                f"• Include analysis metadata\n"
                f"• Organize by situation type\n"
                f"• Enable sharing and further analysis",
            )

    def refresh_pending_layout(self):
        """Refresh the pending clips layout."""
        # Clear layout
        for i in reversed(range(self.pending_layout.count())):
            self.pending_layout.itemAt(i).widget().setParent(None)

        # Re-add clips
        for i, analysis_data in enumerate(self.pending_clips):
            clip_card = ClipCard(analysis_data)
            clip_card.clip_approved.connect(self.approve_clip)
            clip_card.clip_rejected.connect(self.reject_clip)
            clip_card.clip_preview.connect(self.preview_clip)

            row = i // 3
            col = i % 3
            self.pending_layout.addWidget(clip_card, row, col)

    def refresh_approved_layout(self):
        """Refresh the approved clips layout."""
        # Clear layout
        for i in reversed(range(self.approved_layout.count())):
            self.approved_layout.itemAt(i).widget().setParent(None)

        # Re-add clips (simplified view for approved)
        for i, analysis_data in enumerate(self.approved_clips):
            timestamp = analysis_data.get("timestamp", 0)
            situations = analysis_data.get("situations", [])

            card = QLabel(
                f"✅ {timestamp:.1f}s - {situations[0].get('type', 'unknown').replace('_', ' ').title() if situations else 'Clip'}"
            )
            card.setStyleSheet(
                """
                QLabel {
                    background-color: #F0FDF4;
                    border: 1px solid #BBF7D0;
                    border-radius: 6px;
                    padding: 8px;
                    margin: 2px;
                }
            """
            )

            row = i // 3
            col = i % 3
            self.approved_layout.addWidget(card, row, col)

    def update_stats(self):
        """Update the statistics display."""
        pending_count = len(self.pending_clips)
        approved_count = len(self.approved_clips)
        rejected_count = len(self.rejected_clips)

        self.stats_label.setText(
            f"Pending: {pending_count} | Approved: {approved_count} | Rejected: {rejected_count}"
        )


class VideoDropWidget(QLabel):
    """Drag-and-drop widget for video files."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText(
            "🎮 Drag & Drop Video for Auto-Analysis\n\n📁 Click to browse\n\nAuto-detects: 3rd & Long, Red Zone, 2-Min Warning, etc."
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(self.get_default_style())
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)

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

    def get_processing_style(self):
        return """
            QLabel {
                border: 3px solid #F59E0B;
                border-radius: 15px;
                background-color: #FFFBEB;
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
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.video_dropped.emit(file_path)
            else:
                QMessageBox.warning(
                    self, "Invalid File", "Please drop a valid video file (MP4, MOV, AVI, MKV)"
                )

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video for Auto-Analysis",
            "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if file_path:
            self.video_dropped.emit(file_path)

    def is_video_file(self, file_path):
        return file_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))

    def set_processing_state(self, filename):
        """Update widget to show processing state."""
        self.setText(
            f"🔍 Auto-Analyzing: {filename}\n\nDetecting key moments...\n\nClips will appear on the right →"
        )
        self.setStyleSheet(self.get_processing_style())


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Auto-Clip Detection Demo")
        self.setGeometry(100, 100, 1400, 800)
        self.analysis_worker = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header_text = "🏈 SpygateAI Auto-Clip Detection"
        if SPYGATE_AVAILABLE:
            header_text += " (Real Analysis)"
        else:
            header_text += " (Demo Mode)"

        header = QLabel(header_text)
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            """
            QLabel {
                padding: 15px;
                background-color: #059669;
                color: white;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        """
        )
        layout.addWidget(header)

        # Workflow description
        workflow_desc = QLabel(
            "🔄 Automated Workflow: Load Video → Auto-Detect Situations → Create Clips → Review & Approve → Export & Organize"
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

        # Left side: Video drop and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Drop area
        self.drop_widget = VideoDropWidget()
        self.drop_widget.video_dropped.connect(self.start_auto_analysis)
        left_layout.addWidget(self.drop_widget)

        # Analysis controls
        controls_group = QGroupBox("🎛️ Analysis Settings")
        controls_layout = QVBoxLayout()

        # Sensitivity setting
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Detection Sensitivity:"))

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["High (Thorough)", "Medium (Balanced)", "Low (Fast)"])
        self.sensitivity_combo.setCurrentIndex(1)  # Default to Medium
        sens_layout.addWidget(self.sensitivity_combo)

        controls_layout.addLayout(sens_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        # Stop button
        self.stop_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #DC2626;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #B91C1C; }
            QPushButton:disabled { background-color: #D1D5DB; }
        """
        )
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        left_layout.addStretch()
        left_widget.setLayout(left_layout)

        main_splitter.addWidget(left_widget)

        # Right side: Clip review
        self.clip_review = ClipReviewWidget()
        main_splitter.addWidget(self.clip_review)

        main_splitter.setSizes([400, 1000])
        layout.addWidget(main_splitter)

        central.setLayout(layout)

        # Status bar
        self.statusBar().showMessage("Ready - Drop a video file to start automatic clip detection")

    def start_auto_analysis(self, video_path):
        """Start automatic analysis of the video."""
        if not os.path.exists(video_path):
            QMessageBox.critical(self, "Error", "Video file not found!")
            return

        # Update UI
        filename = os.path.basename(video_path)
        self.drop_widget.set_processing_state(filename)
        self.statusBar().showMessage(f"🔍 Auto-analyzing: {filename}")

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stop_btn.setEnabled(True)

        # Get sensitivity setting
        sensitivity_map = {
            "High (Thorough)": "high",
            "Medium (Balanced)": "medium",
            "Low (Fast)": "low",
        }
        sensitivity = sensitivity_map[self.sensitivity_combo.currentText()]

        # Start analysis worker
        self.analysis_worker = AutoAnalysisWorker(video_path, sensitivity)
        self.analysis_worker.situation_detected.connect(self.on_situation_detected)
        self.analysis_worker.progress_update.connect(self.on_progress_update)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        self.analysis_worker.start()

    def stop_analysis(self):
        """Stop the ongoing analysis."""
        if self.analysis_worker:
            self.analysis_worker.stop_analysis()
            self.analysis_worker.wait()  # Wait for thread to finish

        self.on_analysis_complete()

    def on_situation_detected(self, analysis_data):
        """Handle when a situation is detected."""
        # Add the detected clip to review
        self.clip_review.add_detected_clip(analysis_data)

        # Update status
        situations = analysis_data.get("situations", [])
        if situations:
            sit_type = situations[0].get("type", "unknown")
            timestamp = analysis_data.get("timestamp", 0)
            self.statusBar().showMessage(
                f"🎯 Found {sit_type.replace('_', ' ').title()} at {timestamp:.1f}s"
            )

    def on_progress_update(self, current_frame, total_frames):
        """Handle progress updates."""
        if total_frames > 0:
            progress = int((current_frame / total_frames) * 100)
            self.progress_bar.setValue(progress)

    def on_analysis_complete(self):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)

        # Update drop widget
        pending_count = len(self.clip_review.pending_clips)
        self.drop_widget.setText(
            f"✅ Analysis Complete!\n\n{pending_count} clips detected\n\nDrop another video to analyze more"
        )
        self.drop_widget.setStyleSheet(self.drop_widget.get_default_style())

        self.statusBar().showMessage(f"✅ Analysis complete - {pending_count} clips detected")

        if pending_count > 0:
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"🎉 Auto-analysis found {pending_count} key moments!\n\n"
                f"Review the detected clips on the right and choose which ones to keep.\n\n"
                f"You can approve/reject individual clips or use bulk actions.",
            )
        else:
            QMessageBox.information(
                self,
                "Analysis Complete",
                "🔍 Analysis complete, but no significant moments were detected.\n\n"
                "Try adjusting the sensitivity to 'High' for more thorough detection.",
            )

    def on_analysis_error(self, error_message):
        """Handle analysis errors."""
        self.progress_bar.setVisible(False)
        self.stop_btn.setEnabled(False)

        self.drop_widget.setText("❌ Analysis failed\n\nTry another video file")
        self.drop_widget.setStyleSheet(self.drop_widget.get_default_style())

        QMessageBox.critical(self, "Analysis Error", f"Failed to analyze video:\n{error_message}")
        self.statusBar().showMessage("❌ Analysis failed")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpygateAI Auto-Clip Detection")

    # Startup message
    msg = QMessageBox()
    msg.setWindowTitle("SpygateAI Auto-Clip Detection")

    msg_text = """
🏈 SpygateAI Auto-Clip Detection Demo

This demo shows the automated clip detection workflow:

1. 📹 Drop Video - The system automatically analyzes the entire video
2. 🎯 Auto-Detection - Finds key moments (3rd & Long, Red Zone, etc.)
3. 📋 Review Clips - All detected clips appear for your review
4. ✅❌ Approve/Reject - Choose which clips to keep
5. 📤 Export - Export approved clips for analysis

Features:
• Adjustable sensitivity (High/Medium/Low)
• Real-time clip detection
• Bulk approve/reject actions
• Preview detected clips
• Export in multiple formats

This is much more efficient than manual frame-by-frame analysis!
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
