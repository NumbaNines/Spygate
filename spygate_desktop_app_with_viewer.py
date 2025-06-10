#!/usr/bin/env python3

"""
SpygateAI Desktop Application - With Clip Viewer
===============================================

Enhanced version with clip viewing functionality for detected clips.
"""

import json
import os
import sys
import tempfile
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

    # Import detection modules
    try:
        from spygate.core.hardware import HardwareDetector

        hardware_available = True
    except ImportError:
        print("⚠️ Hardware detection not available")
        hardware_available = False

    print("✅ Core imports successful")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class DetectedClip:
    """Represents a detected clip with metadata."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    situation: str
    thumbnail_path: Optional[str] = None
    approved: Optional[bool] = None


class ClipThumbnailWidget(QWidget):
    """Widget for displaying clip thumbnails."""

    clip_selected = pyqtSignal(DetectedClip)

    def __init__(self, clip: DetectedClip, parent=None):
        super().__init__(parent)
        self.clip = clip
        self.setFixedSize(200, 150)
        self.setStyleSheet(
            """
            ClipThumbnailWidget {
                background-color: #1a1a1a;
                border: 2px solid #333;
                border-radius: 8px;
            }
            ClipThumbnailWidget:hover {
                border-color: #ff6b35;
                background-color: #2a2a2a;
            }
        """
        )
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Thumbnail placeholder
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(190, 100)
        self.thumbnail_label.setStyleSheet(
            """
            QLabel {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """
        )
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setText("🎬 Clip Preview")
        layout.addWidget(self.thumbnail_label)

        # Clip info
        info_layout = QHBoxLayout()

        # Time info
        time_label = QLabel(f"{self.clip.start_time:.1f}s - {self.clip.end_time:.1f}s")
        time_label.setStyleSheet("color: #ccc; font-size: 10px;")
        info_layout.addWidget(time_label)

        # Confidence
        conf_label = QLabel(f"{self.clip.confidence:.0%}")
        conf_label.setStyleSheet("color: #4CAF50; font-size: 10px; font-weight: bold;")
        info_layout.addWidget(conf_label)

        layout.addLayout(info_layout)

        # Situation
        situation_label = QLabel(self.clip.situation)
        situation_label.setStyleSheet("color: #ff6b35; font-size: 10px; font-weight: bold;")
        situation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(situation_label)

        # Approve/Reject buttons
        button_layout = QHBoxLayout()

        self.approve_btn = QPushButton("✓")
        self.approve_btn.setFixedSize(25, 25)
        self.approve_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                border: none;
                border-radius: 12px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        )
        self.approve_btn.clicked.connect(self.approve_clip)

        self.reject_btn = QPushButton("✗")
        self.reject_btn.setFixedSize(25, 25)
        self.reject_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                border: none;
                border-radius: 12px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
        """
        )
        self.reject_btn.clicked.connect(self.reject_clip)

        button_layout.addWidget(self.approve_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.reject_btn)

        layout.addLayout(button_layout)

    def approve_clip(self):
        self.clip.approved = True
        self.setStyleSheet(
            self.styleSheet()
            + """
            ClipThumbnailWidget { border-color: #4CAF50; }
        """
        )
        print(
            f"✅ Approved clip: {self.clip.situation} ({self.clip.start_time:.1f}s-{self.clip.end_time:.1f}s)"
        )

    def reject_clip(self):
        self.clip.approved = False
        self.setStyleSheet(
            self.styleSheet()
            + """
            ClipThumbnailWidget {
                border-color: #f44336;
                background-color: #2a1a1a;
                opacity: 0.5;
            }
        """
        )
        print(
            f"❌ Rejected clip: {self.clip.situation} ({self.clip.start_time:.1f}s-{self.clip.end_time:.1f}s)"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clip_selected.emit(self.clip)
            print(f"🎬 Selected clip: {self.clip.situation}")


class ClipViewerWidget(QWidget):
    """Main widget for viewing and managing detected clips."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips = []
        self.current_video_path = None
        self.cap = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("🎬 Detected Clips")
        header.setStyleSheet(
            """
            QLabel {
                color: #ff6b35;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )
        layout.addWidget(header)

        # Controls
        controls_layout = QHBoxLayout()

        self.approve_all_btn = QPushButton("✓ Approve All")
        self.approve_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        )
        self.approve_all_btn.clicked.connect(self.approve_all_clips)

        self.reject_all_btn = QPushButton("✗ Reject All")
        self.reject_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
        """
        )
        self.reject_all_btn.clicked.connect(self.reject_all_clips)

        self.export_btn = QPushButton("📁 Export Approved")
        self.export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e5602e; }
        """
        )
        self.export_btn.clicked.connect(self.export_approved_clips)

        controls_layout.addWidget(self.approve_all_btn)
        controls_layout.addWidget(self.reject_all_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(self.export_btn)

        layout.addLayout(controls_layout)

        # Clips grid (scrollable)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                background-color: #0f0f0f;
                border: none;
            }
        """
        )

        self.clips_widget = QWidget()
        self.clips_layout = QGridLayout(self.clips_widget)
        self.clips_layout.setSpacing(10)

        scroll_area.setWidget(self.clips_widget)
        layout.addWidget(scroll_area)

        # Status
        self.status_label = QLabel("No clips detected yet")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #888;
                padding: 10px;
                background-color: #1a1a1a;
                border-radius: 4px;
            }
        """
        )
        layout.addWidget(self.status_label)

    def load_clips(self, video_path: str, detected_clips_data: list[dict]):
        """Load detected clips from analysis results."""
        self.current_video_path = video_path
        self.clips.clear()

        # Clear existing widgets
        for i in reversed(range(self.clips_layout.count())):
            self.clips_layout.itemAt(i).widget().setParent(None)

        # Create clip objects
        for i, clip_data in enumerate(detected_clips_data):
            clip = DetectedClip(
                start_frame=clip_data.get("start_frame", i * 30),
                end_frame=clip_data.get("end_frame", (i + 1) * 30),
                start_time=clip_data.get("start_time", i * 0.5),
                end_time=clip_data.get("end_time", (i + 1) * 0.5),
                confidence=clip_data.get("confidence", 0.8),
                situation=clip_data.get("situation", f"Key Moment {i+1}"),
            )
            self.clips.append(clip)

            # Create thumbnail widget
            thumbnail_widget = ClipThumbnailWidget(clip)
            thumbnail_widget.clip_selected.connect(self.on_clip_selected)

            # Add to grid (4 clips per row)
            row = i // 4
            col = i % 4
            self.clips_layout.addWidget(thumbnail_widget, row, col)

        self.update_status()

    def update_status(self):
        total = len(self.clips)
        approved = sum(1 for clip in self.clips if clip.approved == True)
        rejected = sum(1 for clip in self.clips if clip.approved == False)
        pending = total - approved - rejected

        self.status_label.setText(
            f"📊 Total: {total} clips | "
            f"✅ Approved: {approved} | "
            f"❌ Rejected: {rejected} | "
            f"⏳ Pending: {pending}"
        )

    def approve_all_clips(self):
        for clip in self.clips:
            clip.approved = True
        self.refresh_display()
        print(f"✅ Approved all {len(self.clips)} clips")

    def reject_all_clips(self):
        for clip in self.clips:
            clip.approved = False
        self.refresh_display()
        print(f"❌ Rejected all {len(self.clips)} clips")

    def refresh_display(self):
        """Refresh the visual state of all clip widgets."""
        for i in range(self.clips_layout.count()):
            widget = self.clips_layout.itemAt(i).widget()
            if isinstance(widget, ClipThumbnailWidget):
                if widget.clip.approved == True:
                    widget.approve_clip()
                elif widget.clip.approved == False:
                    widget.reject_clip()
        self.update_status()

    def export_approved_clips(self):
        approved_clips = [clip for clip in self.clips if clip.approved == True]
        if not approved_clips:
            QMessageBox.information(self, "Export", "No approved clips to export!")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return

        try:
            self.export_clips_to_directory(approved_clips, export_dir)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(approved_clips)} clips to:\n{export_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export clips:\n{str(e)}")

    def export_clips_to_directory(self, clips: list[DetectedClip], export_dir: str):
        """Export clips to the specified directory."""
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            raise ValueError("Source video not available")

        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, clip in enumerate(clips):
            output_path = os.path.join(
                export_dir, f"clip_{i+1}_{clip.situation.replace(' ', '_')}.mp4"
            )

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )

            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)

            # Write frames
            for frame_num in range(clip.start_frame, clip.end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()
            print(f"📁 Exported: {output_path}")

        cap.release()

    def on_clip_selected(self, clip: DetectedClip):
        """Handle clip selection for preview."""
        print(f"🎬 Previewing clip: {clip.situation} ({clip.start_time:.1f}s-{clip.end_time:.1f}s)")
        # TODO: Implement clip preview in a separate window


class AutoDetectWidget(QWidget):
    """Enhanced AutoDetectWidget with clip viewing."""

    analysis_complete = pyqtSignal(str, list)  # video_path, clips_data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_thread = None
        self.detected_clips = []
        self.init_ui()
        self.detect_hardware()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Drop zone
        self.drop_zone = self.create_drop_zone()
        layout.addWidget(self.drop_zone)

        # Progress section
        self.progress_widget = self.create_progress_widget()
        layout.addWidget(self.progress_widget)

        # Hardware info
        self.hardware_widget = self.create_hardware_widget()
        layout.addWidget(self.hardware_widget)

        # Set drag and drop
        self.setAcceptDrops(True)

    def create_drop_zone(self):
        widget = QWidget()
        widget.setFixedHeight(200)
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: 3px dashed #ff6b35;
                border-radius: 12px;
            }
            QWidget:hover {
                background-color: #2a2a2a;
                border-color: #ff8c5a;
            }
        """
        )

        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("🎬")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        text = QLabel("Drop video file here or click to browse")
        text.setStyleSheet("color: #ff6b35; font-size: 16px; font-weight: bold;")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)

        browse_btn = QPushButton("📁 Browse Files")
        browse_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #e5602e; }
        """
        )
        browse_btn.clicked.connect(self.browse_file)
        layout.addWidget(browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        return widget

    def create_progress_widget(self):
        widget = QWidget()
        widget.setVisible(False)

        layout = QVBoxLayout(widget)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background: linear-gradient(to right, #ff6b35, #ff8c5a);
                border-radius: 3px;
            }
        """
        )
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to analyze...")
        self.status_label.setStyleSheet("color: #ccc; font-size: 14px;")
        layout.addWidget(self.status_label)

        self.stop_btn = QPushButton("⏹ Stop Analysis")
        self.stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #da190b; }
        """
        )
        self.stop_btn.clicked.connect(self.stop_analysis)
        layout.addWidget(self.stop_btn)

        return widget

    def create_hardware_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.hardware_label = QLabel("🔧 Hardware: Detecting...")
        self.hardware_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.hardware_label)

        return widget

    def detect_hardware(self):
        try:
            if hardware_available:
                from spygate.core.hardware import HardwareDetector

                self.hardware = HardwareDetector()
                tier = self.hardware.get_tier()
                self.hardware_label.setText(f"🔧 Hardware: {tier} tier")
                self.hardware_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
            else:
                self.hardware_label.setText("🔧 Hardware: Basic mode")
        except Exception as e:
            print(f"Hardware detection error: {e}")
            self.hardware_label.setText("🔧 Hardware: Unknown")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.process_video_file(files[0])

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.process_video_file(file_path)

    def process_video_file(self, file_path):
        print(f"🎬 Processing video: {file_path}")
        self.drop_zone.setVisible(False)
        self.progress_widget.setVisible(True)

        # Start analysis in background thread
        self.analysis_thread = AnalysisWorker(file_path)
        self.analysis_thread.progress_updated.connect(self.update_progress)
        self.analysis_thread.analysis_finished.connect(self.on_analysis_complete)
        self.analysis_thread.start()

    def update_progress(self, progress, message):
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def on_analysis_complete(self, video_path, clips_data):
        self.progress_widget.setVisible(False)
        self.drop_zone.setVisible(True)

        print(f"✅ Analysis complete! Detected {len(clips_data)} clips")
        self.analysis_complete.emit(video_path, clips_data)

    def stop_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.progress_widget.setVisible(False)
            self.drop_zone.setVisible(True)
            print("⏹ Analysis stopped")


class AnalysisWorker(QThread):
    """Worker thread for video analysis."""

    progress_updated = pyqtSignal(int, str)
    analysis_finished = pyqtSignal(str, list)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def run(self):
        try:
            # Simulate analysis with progress updates
            clips_data = []

            # Get video info
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Simulate processing
            clips_detected = 0
            for i in range(0, 100, 2):
                if self.should_stop:
                    return

                # Simulate clip detection
                if i > 20 and i % 15 == 0:  # Detect clips every 15%
                    start_time = (i / 100) * (total_frames / fps)
                    end_time = start_time + 5.0  # 5 second clips

                    situations = [
                        "3rd & Long",
                        "Red Zone",
                        "Turnover",
                        "Touchdown",
                        "Big Play",
                        "Goal Line Stand",
                    ]
                    situation = situations[clips_detected % len(situations)]

                    clips_data.append(
                        {
                            "start_frame": int(start_time * fps),
                            "end_frame": int(end_time * fps),
                            "start_time": start_time,
                            "end_time": end_time,
                            "confidence": 0.75 + (clips_detected % 3) * 0.08,
                            "situation": situation,
                        }
                    )
                    clips_detected += 1

                message = f"Analyzing... {i}% ({clips_detected} clips detected)"
                self.progress_updated.emit(i, message)
                self.msleep(100)  # Simulate processing time

            self.analysis_finished.emit(self.video_path, clips_data)

        except Exception as e:
            print(f"Analysis error: {e}")
            self.analysis_finished.emit(self.video_path, [])


class SpygateDesktopAppWithViewer(QMainWindow):
    """Main application with clip viewing functionality."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏈 SpygateAI Desktop - Clip Viewer")
        self.setGeometry(100, 100, 1400, 900)

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f0f0f;
                color: #ffffff;
            }
        """
        )

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout (horizontal split)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Left panel - Analysis
        self.auto_detect_widget = AutoDetectWidget()
        self.auto_detect_widget.analysis_complete.connect(self.on_clips_detected)
        main_layout.addWidget(self.auto_detect_widget, 1)

        # Right panel - Clip viewer
        self.clip_viewer = ClipViewerWidget()
        main_layout.addWidget(self.clip_viewer, 2)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(
            """
            QStatusBar {
                background-color: #1a1a1a;
                color: #888;
                border-top: 1px solid #333;
            }
        """
        )
        self.status_bar.showMessage("🏈 SpygateAI Desktop Ready")

    def on_clips_detected(self, video_path: str, clips_data: list[dict]):
        """Handle clips detection completion."""
        self.clip_viewer.load_clips(video_path, clips_data)
        self.status_bar.showMessage(
            f"✅ Detected {len(clips_data)} clips from {os.path.basename(video_path)}"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set app icon and name
    app.setApplicationName("SpygateAI Desktop")
    app.setApplicationVersion("1.0.0")

    window = SpygateDesktopAppWithViewer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
