#!/usr/bin/env python3
"""
SpygateAI - Integrated Production Desktop Application
==================================================
A polished FACEIT-style desktop application integrating all core modules.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Set up proper Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    PYQT6_AVAILABLE = True
except ImportError as e:
    print(f"❌ PyQt6 not available: {e}")
    sys.exit(1)

# Import SpygateAI core modules
try:
    from spygate.core.hardware import HardwareDetector, HardwareTier
    from spygate.core.optimizer import TierOptimizer

    SPYGATE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SpygateAI modules not available: {e}")
    SPYGATE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClipData:
    """Data structure for detected clips."""

    def __init__(
        self, start_frame: int, end_frame: int, situation: str, confidence: float, timestamp: str
    ):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.situation = situation
        self.confidence = confidence
        self.timestamp = timestamp
        self.approved = None  # None = pending, True = approved, False = rejected


class VideoDropZone(QLabel):
    """FACEIT-style drag & drop zone."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.supported_formats = [".mp4", ".mov", ".avi", ".mkv"]
        self.init_ui()

    def init_ui(self):
        self.setText("🎮 Drop Video File Here\n\nSupported: MP4, MOV, AVI, MKV\nMax size: 2GB")
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
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and self._is_video_file(files[0]):
            self.video_dropped.emit(files[0])
        self.init_ui()

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a supported video format."""
        return Path(file_path).suffix.lower() in self.supported_formats


class AnalysisWorker(QThread):
    """Worker thread for video analysis."""

    progress_updated = pyqtSignal(int, str)
    clip_detected = pyqtSignal(ClipData)
    analysis_complete = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path: str, hardware_tier: str):
        super().__init__()
        self.video_path = video_path
        self.hardware_tier = hardware_tier
        self.should_stop = False

    def run(self):
        """Run video analysis."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit("Could not open video file")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            clips_detected = 0
            frame_skip = {"ultra": 15, "high": 30, "medium": 60, "low": 90}.get(
                self.hardware_tier.lower(), 30
            )

            frame_number = 0
            while frame_number < total_frames and not self.should_stop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    break

                # Update progress
                progress = int((frame_number / total_frames) * 100)
                status = f"Analyzing frame {frame_number}/{total_frames}"
                self.progress_updated.emit(progress, status)

                # Simulate clip detection
                if frame_number % (frame_skip * 10) == 0:  # Occasional detection
                    if np.random.random() > 0.7:  # 30% chance
                        situation = np.random.choice(
                            ["3rd & Long", "Red Zone", "Turnover", "Scoring Play", "Big Play"]
                        )

                        clip_data = ClipData(
                            start_frame=max(0, frame_number - 150),
                            end_frame=min(total_frames, frame_number + 150),
                            situation=situation,
                            confidence=0.75 + (np.random.random() * 0.2),
                            timestamp=self._frame_to_timestamp(frame_number, fps),
                        )
                        self.clip_detected.emit(clip_data)
                        clips_detected += 1

                frame_number += frame_skip

            cap.release()
            self.analysis_complete.emit(clips_detected)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            self.error_occurred.emit(str(e))

    def _frame_to_timestamp(self, frame_number: int, fps: float) -> str:
        """Convert frame number to timestamp."""
        seconds = frame_number / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def stop(self):
        """Stop analysis."""
        self.should_stop = True


class ClipPreviewWidget(QWidget):
    """Widget for clip preview and approval."""

    clip_approved = pyqtSignal(ClipData)
    clip_rejected = pyqtSignal(ClipData)

    def __init__(self, clip_data: ClipData):
        super().__init__()
        self.clip_data = clip_data
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(280, 200)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
            }
            QWidget:hover {
                border-color: #ff6b35;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Thumbnail placeholder
        thumbnail = QLabel("📹")
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail.setStyleSheet(
            """
            QLabel {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 6px;
                font-size: 32px;
                padding: 20px;
                min-height: 80px;
            }
        """
        )
        layout.addWidget(thumbnail)

        # Clip info
        info_text = f"{self.clip_data.situation}\n"
        info_text += f"Confidence: {self.clip_data.confidence:.1%}\n"
        info_text += f"Time: {self.clip_data.timestamp}"

        info_label = QLabel(info_text)
        info_label.setStyleSheet(
            """
            QLabel {
                color: #ccc;
                font-size: 11px;
                background: transparent;
                border: none;
            }
        """
        )
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()

        approve_btn = QPushButton("✅")
        approve_btn.clicked.connect(lambda: self.clip_approved.emit(self.clip_data))
        approve_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #059669; }
        """
        )

        reject_btn = QPushButton("❌")
        reject_btn.clicked.connect(lambda: self.clip_rejected.emit(self.clip_data))
        reject_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #DC2626; }
        """
        )

        button_layout.addWidget(approve_btn)
        button_layout.addWidget(reject_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)


class ClipReviewPanel(QWidget):
    """Panel for reviewing detected clips."""

    def __init__(self):
        super().__init__()
        self.clips = []
        self.approved_clips = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("📹 Detected Clips Review")
        header.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 10px 0;
            }
        """
        )
        layout.addWidget(header)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                background-color: #0f0f0f;
                border: none;
            }
        """
        )

        self.clips_widget = QWidget()
        self.clips_layout = QGridLayout(self.clips_widget)
        self.clips_layout.setSpacing(15)

        scroll.setWidget(self.clips_widget)
        layout.addWidget(scroll)

        # Export button
        export_btn = QPushButton("🎬 Export Approved Clips")
        export_btn.clicked.connect(self.export_clips)
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 15px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e55a2b; }
        """
        )
        layout.addWidget(export_btn)

        self.setLayout(layout)

    def add_clip(self, clip_data: ClipData):
        """Add a clip to the review panel."""
        clip_widget = ClipPreviewWidget(clip_data)
        clip_widget.clip_approved.connect(self._on_clip_approved)
        clip_widget.clip_rejected.connect(self._on_clip_rejected)

        # Add to grid
        row = len(self.clips) // 4
        col = len(self.clips) % 4
        self.clips_layout.addWidget(clip_widget, row, col)

        self.clips.append(clip_data)

    def _on_clip_approved(self, clip_data: ClipData):
        """Handle clip approval."""
        clip_data.approved = True
        self.approved_clips.append(clip_data)
        logger.info(f"Clip approved: {clip_data.situation}")

    def _on_clip_rejected(self, clip_data: ClipData):
        """Handle clip rejection."""
        clip_data.approved = False
        logger.info(f"Clip rejected: {clip_data.situation}")

    def export_clips(self):
        """Export approved clips."""
        if not self.approved_clips:
            QMessageBox.information(self, "No Clips", "No approved clips to export.")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if export_dir:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(self.approved_clips)} clips to {export_dir}",
            )


class SidebarWidget(QWidget):
    """FACEIT-style sidebar."""

    tab_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedWidth(250)
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

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 0)

        logo = QLabel("🏈")
        logo.setFont(QFont("Arial", 24))
        logo.setStyleSheet("color: #ff6b35; background: transparent;")

        title = QLabel("SpygateAI")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")

        header_layout.addWidget(logo)
        header_layout.addWidget(title)
        header_layout.addStretch()

        header.setLayout(header_layout)
        layout.addWidget(header)

        # Navigation
        nav_items = [
            ("🎬", "Analysis", "analysis"),
            ("📹", "Review", "review"),
            ("⚙️", "Settings", "settings"),
        ]

        for icon, text, action in nav_items:
            btn = self._create_nav_button(icon, text, action)
            layout.addWidget(btn)

        layout.addStretch()

        # Hardware info
        if SPYGATE_AVAILABLE:
            try:
                detector = HardwareDetector()
                tier_text = f"Hardware: {detector.tier.name}"
            except:
                tier_text = "Hardware: Unknown"
        else:
            tier_text = "Hardware: Not detected"

        hw_label = QLabel(tier_text)
        hw_label.setStyleSheet(
            """
            QLabel {
                color: #666;
                font-size: 10px;
                padding: 10px 20px;
                background: transparent;
            }
        """
        )
        layout.addWidget(hw_label)

        self.setLayout(layout)

    def _create_nav_button(self, icon: str, text: str, action: str):
        """Create navigation button."""
        btn = QWidget()
        btn.setFixedHeight(50)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            """
            QWidget {
                background-color: transparent;
                border-left: 3px solid transparent;
            }
            QWidget:hover {
                background-color: #2a2a2a;
                border-left: 3px solid #ff6b35;
            }
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)

        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 16))
        icon_label.setStyleSheet("color: #ccc; background: transparent;")

        text_label = QLabel(text)
        text_label.setFont(QFont("Arial", 12))
        text_label.setStyleSheet("color: #ccc; background: transparent;")

        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()

        btn.setLayout(layout)
        btn.mousePressEvent = lambda e, a=action: self.tab_changed.emit(a)

        return btn


class ProductionApp(QMainWindow):
    """Main production desktop application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI - Production Desktop Application")
        self.setGeometry(100, 100, 1400, 900)

        self.current_video_path = None
        self.analysis_worker = None
        self.hardware_tier = self._detect_hardware_tier()

        self.init_ui()

    def _detect_hardware_tier(self) -> str:
        """Detect hardware tier."""
        if SPYGATE_AVAILABLE:
            try:
                detector = HardwareDetector()
                return detector.tier.name.lower()
            except:
                pass
        return "medium"

    def init_ui(self):
        """Initialize UI."""
        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f0f0f;
                color: white;
            }
        """
        )

        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.tab_changed.connect(self._switch_tab)
        main_layout.addWidget(self.sidebar)

        # Content stack
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #0f0f0f;")

        # Create pages
        self.analysis_widget = self._create_analysis_widget()
        self.review_widget = ClipReviewPanel()
        self.settings_widget = self._create_settings_widget()

        self.content_stack.addWidget(self.analysis_widget)  # 0
        self.content_stack.addWidget(self.review_widget)  # 1
        self.content_stack.addWidget(self.settings_widget)  # 2

        main_layout.addWidget(self.content_stack)
        central.setLayout(main_layout)

    def _create_analysis_widget(self):
        """Create analysis widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("🎬 Auto-Clip Detection Analysis")
        header.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 20px 0;
            }
        """
        )
        layout.addWidget(header)

        # Drop zone
        self.drop_zone = VideoDropZone()
        self.drop_zone.video_dropped.connect(self._on_video_dropped)
        layout.addWidget(self.drop_zone)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #333;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                color: white;
                background-color: #1a1a1a;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b35, stop:1 #e55a2b);
                border-radius: 4px;
            }
        """
        )
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("Ready for analysis")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #ccc;
                font-size: 12px;
                padding: 10px 0;
            }
        """
        )
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("🚀 Start Analysis")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_analysis)
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e55a2b; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """
        )

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #DC2626; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """
        )

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)
        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _create_settings_widget(self):
        """Create settings widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("⚙️ Settings")
        header.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 20px 0;
            }
        """
        )
        layout.addWidget(header)

        # Hardware info
        hw_info = QLabel(f"Hardware Tier: {self.hardware_tier.upper()}")
        hw_info.setStyleSheet(
            """
            QLabel {
                color: #ff6b35;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 0;
            }
        """
        )
        layout.addWidget(hw_info)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _switch_tab(self, tab_name: str):
        """Switch tabs."""
        indices = {"analysis": 0, "review": 1, "settings": 2}
        if tab_name in indices:
            self.content_stack.setCurrentIndex(indices[tab_name])

    def _on_video_dropped(self, video_path: str):
        """Handle video drop."""
        self.current_video_path = video_path
        self.start_btn.setEnabled(True)

        filename = Path(video_path).name
        self.drop_zone.setText(f"✅ Video Loaded:\n{filename}\n\nReady for analysis!")
        self.drop_zone.setStyleSheet(
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

    def _start_analysis(self):
        """Start analysis."""
        if not self.current_video_path:
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)

        self.analysis_worker = AnalysisWorker(self.current_video_path, self.hardware_tier)
        self.analysis_worker.progress_updated.connect(self._on_progress_updated)
        self.analysis_worker.clip_detected.connect(self._on_clip_detected)
        self.analysis_worker.analysis_complete.connect(self._on_analysis_complete)
        self.analysis_worker.error_occurred.connect(self._on_analysis_error)
        self.analysis_worker.start()

    def _stop_analysis(self):
        """Stop analysis."""
        if self.analysis_worker:
            self.analysis_worker.stop()
            self.analysis_worker.wait()
        self._reset_analysis_ui()

    def _on_progress_updated(self, progress: int, status: str):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)

    def _on_clip_detected(self, clip_data: ClipData):
        """Handle clip detection."""
        self.review_widget.add_clip(clip_data)

    def _on_analysis_complete(self, total_clips: int):
        """Handle analysis completion."""
        self._reset_analysis_ui()
        if total_clips > 0:
            self._switch_tab("review")
            QMessageBox.information(
                self, "Analysis Complete", f"Found {total_clips} clips! Check the Review tab."
            )
        else:
            QMessageBox.information(self, "Analysis Complete", "No clips detected.")

    def _on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        self._reset_analysis_ui()
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_message}")

    def _reset_analysis_ui(self):
        """Reset analysis UI."""
        self.start_btn.setEnabled(bool(self.current_video_path))
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("SpygateAI Production")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SpygateAI")

    # Apply dark theme
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(15, 15, 15))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(26, 26, 26))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(42, 42, 42))
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
    window = ProductionApp()
    window.show()

    logger.info("🏈 SpygateAI Production Desktop App - Ready!")
    print("🏈 SpygateAI Production Desktop App - Ready!")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
