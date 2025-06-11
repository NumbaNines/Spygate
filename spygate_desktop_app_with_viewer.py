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

        # Thumbnail image
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
        self.thumbnail_label.setScaledContents(True)
        
        # Load thumbnail if available
        if self.clip.thumbnail_path and os.path.exists(self.clip.thumbnail_path):
            pixmap = QPixmap(self.clip.thumbnail_path)
            self.thumbnail_label.setPixmap(pixmap)
        else:
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

            # Generate thumbnail for clip
            thumbnail_path = self.generate_thumbnail(video_path, clip.start_frame)
            clip.thumbnail_path = thumbnail_path
            
            # Create thumbnail widget
            thumbnail_widget = ClipThumbnailWidget(clip)
            thumbnail_widget.clip_selected.connect(self.on_clip_selected)

            # Add to grid (4 clips per row)
            row = i // 4
            col = i % 4
            self.clips_layout.addWidget(thumbnail_widget, row, col)

        self.update_status()

    def generate_thumbnail(self, video_path: str, frame_number: int) -> Optional[str]:
        """Generate thumbnail image for a specific frame."""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize frame for thumbnail (190x100 to match widget size)
                thumbnail = cv2.resize(frame, (190, 100))
                
                # Create temp file for thumbnail
                import tempfile
                temp_dir = tempfile.gettempdir()
                thumbnail_path = os.path.join(temp_dir, f"thumbnail_{frame_number}.jpg")
                
                # Save thumbnail
                cv2.imwrite(thumbnail_path, thumbnail)
                return thumbnail_path
                
        except Exception as e:
            print(f"❌ Thumbnail generation failed: {e}")
            
        return None

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
        
        # Show clip preview dialog
        if self.current_video_path and os.path.exists(self.current_video_path):
            self.show_clip_preview(clip)
    
    def show_clip_preview(self, clip: DetectedClip):
        """Show a simple clip preview dialog."""
        try:
            # Create preview dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Preview: {clip.situation}")
            dialog.setFixedSize(480, 320)
            dialog.setStyleSheet("background-color: #1a1a1a; color: white;")
            
            layout = QVBoxLayout(dialog)
            
            # Clip info
            info_label = QLabel(f"📹 {clip.situation}\n⏱️ {clip.start_time:.1f}s - {clip.end_time:.1f}s\n🎯 Confidence: {clip.confidence:.2f}")
            info_label.setStyleSheet("color: #ff6b35; font-weight: bold; padding: 10px;")
            layout.addWidget(info_label)
            
            # Thumbnail (larger)
            if clip.thumbnail_path and os.path.exists(clip.thumbnail_path):
                thumbnail_label = QLabel()
                thumbnail_label.setFixedSize(400, 225)
                thumbnail_label.setScaledContents(True)
                thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pixmap = QPixmap(clip.thumbnail_path)
                thumbnail_label.setPixmap(pixmap)
                layout.addWidget(thumbnail_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            play_btn = QPushButton("▶️ Play Clip")
            play_btn.clicked.connect(lambda: self.play_clip_externally(clip))
            button_layout.addWidget(play_btn)
            
            close_btn = QPushButton("❌ Close")
            close_btn.clicked.connect(dialog.close)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            dialog.exec()
            
        except Exception as e:
            print(f"❌ Preview error: {e}")
    
    def play_clip_externally(self, clip: DetectedClip):
        """Open clip in default video player."""
        try:
            import tempfile
            import subprocess
            
            # Create temp clip file
            temp_clip_path = os.path.join(tempfile.gettempdir(), f"temp_clip_{clip.start_frame}.mp4")
            
            # Extract clip using cv2
            cap = cv2.VideoCapture(self.current_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_clip_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
            
            for frame_num in range(clip.start_frame, min(clip.end_frame, clip.start_frame + int(fps * 10))):  # Max 10 sec clips
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            
            # Open with default player
            if os.path.exists(temp_clip_path):
                os.startfile(temp_clip_path)  # Windows
                print(f"🎬 Playing clip: {temp_clip_path}")
            
        except Exception as e:
            print(f"❌ Clip playback error: {e}")


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
        
        # Create and show animation during analysis
        self.create_animation_overlay()

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
        
        # Hide animation
        self.hide_animation_overlay()

        print(f"✅ Analysis complete! Detected {len(clips_data)} clips")
        self.analysis_complete.emit(video_path, clips_data)

    def stop_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.progress_widget.setVisible(False)
            self.drop_zone.setVisible(True)
            print("⏹ Analysis stopped")
            
        # Also hide animation overlay if analysis is stopped
        self.hide_animation_overlay()
        
    def create_animation_overlay(self):
        """Create and show 4-image cycling animation overlay during analysis"""
        print("🎬 Creating animation overlay...")
        
        # Create overlay widget
        self.animation_overlay = QLabel(self)
        self.animation_overlay.setFixedSize(100, 100)
        self.animation_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.animation_overlay.setScaledContents(True)  # Scale image to fit
        
        # Load animation images 1.png, 2.png, 3.png, 4.png
        self.animation_images = []
        self.animation_index = 0
        
        for i in range(1, 5):  # 1, 2, 3, 4
            image_path = f"assets/other/{i}.png"
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    self.animation_images.append(pixmap)
                    print(f"🎬 Loaded animation image: {image_path}")
            else:
                print(f"⚠️ Animation image not found: {image_path}")
        
        # Fallback if no images found
        if not self.animation_images:
            print("⚠️ No animation images found, using emoji fallback")
            self.animation_overlay.setText("🏈")
            self.animation_overlay.setStyleSheet("""
                QLabel {
                    background-color: rgba(45, 45, 45, 180);
                    border: 2px solid #ff6b35;
                    border-radius: 50px;
                    color: #ff6b35;
                    font-size: 48px;
                }
            """)
        else:
            # Set initial image
            self.animation_overlay.setPixmap(self.animation_images[0])
        
        # Position animation in center of widget
        self.position_animation_overlay()
        
        # Show overlay
        self.animation_overlay.show()
        self.animation_overlay.raise_()  # Bring to front
        
        # Start animation timer - cycle every 1 second (1000ms)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_images)
        self.animation_timer.start(1000)  # Update every 1000ms (1 second)
        
        print("🎬 Animation overlay created and cycling started")
        
    def position_animation_overlay(self):
        """Position animation overlay in center of widget"""
        if hasattr(self, 'animation_overlay'):
            # Get center of this widget
            center_x = self.width() // 2 - 50  # 50 is half of animation size (100/2)
            center_y = self.height() // 2 - 50
            self.animation_overlay.move(center_x, center_y)
            print(f"🎬 Animation positioned at center: ({center_x}, {center_y})")
            
    def animate_images(self):
        """Cycle through the 4 animation images"""
        if hasattr(self, 'animation_overlay') and self.animation_overlay.isVisible():
            if self.animation_images:
                # Move to next image
                self.animation_index = (self.animation_index + 1) % len(self.animation_images)
                self.animation_overlay.setPixmap(self.animation_images[self.animation_index])
                print(f"🎬 Showing image {self.animation_index + 1}")
            else:
                # For emoji fallback, cycle through football emojis
                emojis = ["🏈", "⚡", "🔥", "💨"]
                emoji_index = self.animation_index % len(emojis)
                self.animation_overlay.setText(emojis[emoji_index])
                self.animation_index = (self.animation_index + 1) % len(emojis)
                
    def hide_animation_overlay(self):
        """Hide and cleanup animation overlay"""
        if hasattr(self, 'animation_timer'):
            self.animation_timer.stop()
            
        if hasattr(self, 'animation_overlay'):
            self.animation_overlay.hide()
            
        print("🎬 Animation overlay hidden")
        
    def resizeEvent(self, event):
        """Handle widget resize to reposition animation"""
        super().resizeEvent(event)
        self.position_animation_overlay()


class AnalysisWorker(QThread):
    """Worker thread for video analysis."""

    progress_updated = pyqtSignal(int, str)
    analysis_finished = pyqtSignal(str, list)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.should_stop = False
        
        # Context tracking for down detection logic
        self.recent_downs = []  # Store recent down detections with frame numbers
        self.recent_flags = []  # Store recent flag detections
        self.context_window = 300  # 5 seconds at 60fps (reduced for better disambiguation)
        
        # Game flow tracking for intelligent down progression
        self.last_confirmed_down = None  # Track the last reliably detected down
        self.last_yardage = None         # Track the last distance to go
        self.expected_next_down = None   # What down we expect to see next based on game logic

    def stop(self):
        self.should_stop = True
    
    def extract_yardage(self, ocr_text):
        """Extract yardage/distance from OCR text."""
        import re
        
        # Look for numbers that represent distance to go
        # Common patterns: "1st & 10", "2nd & 7", "3rd & 3", etc.
        yardage_match = re.search(r'&\s*(\d+)', ocr_text)
        if yardage_match:
            return int(yardage_match.group(1))
            
        # Look for standalone numbers at the end (missing &)
        number_match = re.search(r'(\d+)$', ocr_text)
        if number_match:
            return int(number_match.group(1))
            
        return None
    
    def analyze_yardage_change(self, current_yardage, current_down):
        """Analyze yardage change to validate down progression."""
        if self.last_yardage is None:
            return f"No previous yardage to compare"
            
        yardage_change = self.last_yardage - current_yardage if current_yardage else None
        
        if current_yardage == 10 and current_down == "1st Down":
            return f"✅ FIRST DOWN ACHIEVED: Reset to 1st & 10 (was {self.last_yardage} yards)"
            
        if current_yardage == self.last_yardage:
            return f"🔄 SAME YARDAGE: {current_yardage} yards (incomplete pass, sack for no loss, or penalty)"
            
        if yardage_change and yardage_change > 0:
            return f"⬇️ YARDS GAINED: {yardage_change} yards (from {self.last_yardage} to {current_yardage})"
            
        if yardage_change and yardage_change < 0:
            return f"⬆️ YARDS LOST: {abs(yardage_change)} yards (penalty or sack - from {self.last_yardage} to {current_yardage})"
            
        return f"🤔 YARDAGE CHANGE: {self.last_yardage} → {current_yardage}"

    def predict_next_down(self, current_down):
        """Predict what down should come next based on football game logic."""
        if current_down == "1st Down":
            return "2nd Down"
        elif current_down == "2nd Down": 
            return "3rd Down"
        elif current_down == "3rd Down":
            return "4th Down"  # or back to 1st if they get first down
        elif current_down == "4th Down":
            return "1st Down"  # turnover or back to 1st if they convert
        return None
    
    def update_game_flow(self, detected_down, frame_number, ocr_text):
        """Update game flow tracking with new down detection."""
        # Extract current yardage
        current_yardage = self.extract_yardage(ocr_text)
        
        print(f"🏈 GAME FLOW: Current={detected_down}, Expected={self.expected_next_down}, Last={self.last_confirmed_down}")
        
        # Analyze yardage change for additional validation
        if current_yardage:
            yardage_analysis = self.analyze_yardage_change(current_yardage, detected_down)
            print(f"📏 YARDAGE ANALYSIS: {yardage_analysis}")
        
        # Update tracking
        self.last_confirmed_down = detected_down
        self.last_yardage = current_yardage
        self.expected_next_down = self.predict_next_down(detected_down)
        
        print(f"🔮 PREDICTION: Next down should be {self.expected_next_down}")
    
    def check_for_missed_third_down(self, current_frame):
        """
        REVERSE DETECTION STRATEGY: When we detect a 4th down, look backwards 
        for a missed 3rd down that should have preceded it.
        """
        print(f"🔍 REVERSE DETECTION: 4th down detected, checking for missed 3rd down...")
        
        # Look back in recent frames for a 3rd down
        recent_third_downs = [d for d in self.recent_downs if d['down'] == '3rd Down']
        
        if recent_third_downs:
            most_recent_third = recent_third_downs[-1]
            frames_since_third = current_frame - most_recent_third['frame']
            print(f"    ✅ Found recent 3rd down at frame {most_recent_third['frame']} ({frames_since_third} frames ago)")
            return
        
        # No recent 3rd down found - check if we can retroactively find one
        print(f"    ⚠️ No recent 3rd down detected before this 4th down")
        print(f"    📚 GAME LOGIC: 4th downs must be preceded by 3rd downs")
        print(f"    💡 SUGGESTION: A 3rd down was likely missed due to:")
        print(f"       - OCR misread ('3rd' → '3nd' → classified as 2nd)")
        print(f"       - Timing constraints (3-second spacing rule)")
        print(f"       - Complex text formatting in HUD")
        
        # Check if we have any 2nd downs that might have been misclassified 3rd downs
        recent_second_downs = [d for d in self.recent_downs if d['down'] == '2nd Down']
        
        if recent_second_downs:
            most_recent_second = recent_second_downs[-1]
            frames_since_second = current_frame - most_recent_second['frame']
            
            # If the most recent 2nd down was close to this 4th down, it might have been a 3rd
            if frames_since_second < 1800:  # Within 30 seconds (1800 frames at 60fps)
                print(f"    🔍 POTENTIAL MISCLASSIFICATION: Found 2nd down at frame {most_recent_second['frame']}")
                print(f"       Text: '{most_recent_second['text']}'")
                print(f"       This might have been a misclassified 3rd down")
                
                # Check if the text contains 3rd down indicators
                second_text = most_recent_second['text'].upper()
                if any(x in second_text for x in ['3ND', '3D', 'RD', '3R', 'SD', '3RO']):
                    print(f"    🎯 RETROACTIVE CORRECTION: Text contains 3rd down indicators!")
                    print(f"       Updating frame {most_recent_second['frame']} from '2nd Down' → '3rd Down'")
                    
                    # Update the misclassified detection
                    most_recent_second['down'] = '3rd Down'
                    return  # We found and corrected the 3rd down
        
        # If we still haven't found a 3rd down, try re-analyzing recent frames
        print(f"    🔄 INITIATING RE-ANALYSIS: Looking for missed 3rd downs in recent frames...")
        
        # Get video capture object for re-analysis (this is a bit hacky, but works)
        import cv2
        temp_cap = cv2.VideoCapture(self.video_path)
        
        try:
            found_third = self.reanalyze_recent_frames_for_third_down(current_frame, temp_cap)
            if found_third:
                print(f"    ✅ REVERSE DETECTION SUCCESS: Found missed 3rd down via re-analysis!")
            else:
                print(f"    😞 REVERSE DETECTION FAILED: No 3rd down found in re-analysis")
        finally:
            temp_cap.release()
        
        print(f"    📊 Recent downs context:")
        for down in self.recent_downs[-5:]:  # Show last 5
            print(f"       Frame {down['frame']}: {down['down']} ('{down['text']}')")

    def reanalyze_recent_frames_for_third_down(self, current_frame, video_cap):
        """
        Re-analyze recent video frames specifically looking for missed 3rd downs.
        This is called when we detect a 4th down but no preceding 3rd down.
        """
        if not hasattr(self, 'hud_model') or self.hud_model is None:
            print(f"    ⚠️ No HUD model available for re-analysis")
            return
            
        print(f"🔄 RE-ANALYZING recent frames for missed 3rd downs...")
        
        # Look back up to 30 seconds (1800 frames at 60fps)
        search_range = 1800
        start_frame = max(0, current_frame - search_range)
        
        # Sample frames every 3 seconds (180 frames) in the search range
        sample_interval = 180
        
        found_third_down = False
        
        for frame_number in range(start_frame, current_frame, sample_interval):
            # Skip frames we already analyzed
            analyzed_frames = [d['frame'] for d in self.recent_downs]
            if frame_number in analyzed_frames:
                continue
                
            try:
                # Seek to the frame
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video_cap.read()
                
                if not ret:
                    continue
                    
                # Run HUD detection
                hud_results = self.hud_model(frame, verbose=False)
                
                if len(hud_results[0].boxes) > 0:
                    # Get the HUD bounding box
                    hud_box = hud_results[0].boxes[0]
                    
                    # Check for down & distance specifically looking for 3rd downs
                    is_down, down_type = self.detect_down_and_distance_focused_third(frame, hud_box, frame_number)
                    
                    if is_down and down_type == "3rd Down":
                        print(f"    🎯 FOUND MISSED 3rd DOWN at frame {frame_number}!")
                        print(f"       This 3rd down was missed in the original analysis")
                        
                        # Add to context tracking
                        self.recent_downs.append({
                            'frame': frame_number,
                            'down': '3rd Down',
                            'text': 'Retroactively detected'
                        })
                        
                        found_third_down = True
                        break
                        
            except Exception as e:
                print(f"    ⚠️ Error re-analyzing frame {frame_number}: {e}")
                continue
        
        if not found_third_down:
            print(f"    😞 No missed 3rd downs found in re-analysis")
        
        return found_third_down

    def detect_down_and_distance_focused_third(self, frame, hud_box, frame_number=0):
        """
        Focused detection method that specifically looks for 3rd downs with aggressive pattern matching.
        Used in reverse detection when a 4th down is found but no preceding 3rd down was detected.
        """
        try:
            import pytesseract
            import re
            import cv2
            
            # Set Tesseract path for Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Extract HUD region from frame using bounding box
            x1, y1, x2, y2 = hud_box.xyxy[0].cpu().numpy()
            hud_region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Use PRECISE coordinates for down/distance text (columns 15-17, full height)
            hud_height, hud_width = hud_region.shape[:2]
            
            # Precise down/distance region coordinates
            x_start = int(hud_width * 0.750)  # 75% across (column 15)
            x_end = int(hud_width * 0.900)    # 90% across (column 17)  
            y_start = 0                       # Full height top
            y_end = hud_height               # Full height bottom
            
            # Extract the precise down/distance region
            down_distance_region = hud_region[y_start:y_end, x_start:x_end]
            
            # Scale up the small region for better OCR (5x)
            scale_factor = 5
            scaled_width = down_distance_region.shape[1] * scale_factor
            scaled_height = down_distance_region.shape[0] * scale_factor
            scaled_region = cv2.resize(down_distance_region, (scaled_width, scaled_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
            
            # Apply high-contrast preprocessing for clean OCR
            _, thresh_region = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR with optimal settings for Madden HUD text
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP '
            
            try:
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(thresh_region, config=custom_config).strip()
                
                if ocr_text:
                    # Clean and normalize the text
                    clean_text = re.sub(r'[^0-9A-Za-z&\s]', '', ocr_text).upper()
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    print(f"    🔍 Re-analysis OCR: '{clean_text}'")
                    
                    # AGGRESSIVE 3rd DOWN PATTERNS - look for ANY indication of 3rd down
                    aggressive_third_patterns = [
                        # Standard patterns
                        r'3RD\s*&\s*\d+',        # "3rd & 5", "3rd & 8", etc.
                        r'3RD\s*&\s*G',          # "3rd & G" (Goal)
                        r'3RD\s*&\s*GOAL',       # "3rd & Goal"
                        
                        # "3nd" OCR misreads (the key issue!)
                        r'3ND\s*&\s*\d+',        # "3nd & 5" (d read as n)
                        r'\d*3ND\s*&',           # "33nd &" (extra digits)
                        r'\d*3ND\s*&\s*\d*',     # "323nd & 5" 
                        r'3ND\s*\d+',            # "3nd5" (missing &)
                        r'\d+3ND\d+',            # "323nd5" (missing &)
                        
                        # Common OCR errors for "3rd" 
                        r'3D\s*&\s*\d+',         # "3d & 5" (missing r)
                        r'RD\s*&\s*\d+',         # "rd & 5" (missing 3)
                        r'3R\s*&\s*\d+',         # "3r & 5" (missing d)
                        r'SD\s*&\s*\d+',         # "sd & 5" (3 read as S)
                        r'3RO\s*&\s*\d+',        # "3ro & 5" (d read as o)
                        
                        # Very aggressive - any "3" + "something" + numbers
                        r'3[A-Z]*\s*&\s*\d+',    # "3rd", "3nd", "3d", etc.
                        r'3[A-Z]*\s*\d+',        # "3rd5", "3nd8", etc. (missing &)
                        r'\d*3[A-Z]*\s*&\s*\d*', # Any variation with extra digits
                        
                        # OCR with missing "&" symbol
                        r'3RD\s*\d+',            # "3rd 5" (missing &)
                        r'3D\s*\d+',             # "3d 5"
                        r'RD\s*\d+',             # "rd 5"
                        
                        # Fragment detection - sometimes OCR only gets part
                        r'RD\s*[1-9]',           # "rd5", "rd3", "rd8"
                        r'D\s*&\s*[1-9]',        # "d & 5" (missing "3r")
                        r'3\s*&\s*[1-9]',        # "3 & 5" (missing "rd")
                        
                        # Goal patterns
                        r'3.*GOAL',              # Any 3rd down with Goal
                        r'RD.*GOAL',             # "rd" with Goal
                    ]
                    
                    # Check all aggressive patterns
                    for pattern in aggressive_third_patterns:
                        try:
                            if re.search(pattern, clean_text):
                                print(f"    🎯 AGGRESSIVE 3rd DOWN MATCH! Pattern: {pattern}")
                                return True, "3rd Down"
                        except Exception as pattern_error:
                            continue
                    
                    # Secondary check: Look for any numbers that could indicate 3rd down
                    # In short yardage situations, "3rd & 1", "3rd & 2", etc.
                    if any(x in clean_text for x in ['&1', '&2', '&3', '&4', '&5']):
                        # Check if there's any indication of "3rd" nearby
                        if any(x in clean_text for x in ['3', 'RD', 'D']):
                            print(f"    🎯 SHORT YARDAGE 3rd DOWN SUSPECTED in: '{clean_text}'")
                            return True, "3rd Down"
                    
                    print(f"    ❌ No 3rd down patterns found in re-analysis")
                    return False, None
                    
            except Exception as ocr_error:
                print(f"    ⚠️ OCR error in re-analysis: {ocr_error}")
                return False, None
                
        except Exception as e:
            print(f"    ⚠️ Error in focused 3rd down detection: {e}")
            return False, None

    def validate_down_with_game_logic(self, ocr_down, ocr_text, frame_number):
        """Use game logic to validate or correct OCR down detection."""
        current_yardage = self.extract_yardage(ocr_text)
        
        if self.expected_next_down is None:
            # No prior context, trust the OCR
            print(f"🤔 NO PRIOR CONTEXT: Trusting OCR result '{ocr_down}'")
            return ocr_down
        
        # YARDAGE-BASED VALIDATION: Check for first down achievement
        if (current_yardage == 10 and ocr_down != "1st Down" and 
            self.last_yardage and self.last_yardage < 10):
            print(f"🔧 YARDAGE CORRECTION: Yardage reset to 10, should be 1st Down")
            print(f"    OCR said '{ocr_down}' but yardage indicates first down achieved")
            return "1st Down"
            
        # Check if OCR matches expected down
        if ocr_down == self.expected_next_down:
            print(f"✅ GAME LOGIC CONFIRMS: OCR '{ocr_down}' matches expected '{self.expected_next_down}'")
            return ocr_down
        
        # OCR doesn't match expected - check for common misreads that we can correct
        if self.expected_next_down == "3rd Down" and ocr_down == "2nd Down":
            # This could be "3rd" read as "3nd" and misclassified as 2nd down
            if any(x in ocr_text for x in ['3ND', '3nd', '3D', '3d']):
                print(f"🔧 GAME LOGIC CORRECTION: OCR said '{ocr_down}' but expected '3rd Down'")
                print(f"    OCR text '{ocr_text}' contains 3rd down indicators, correcting to 3rd Down")
                return "3rd Down"
                
            # Also check yardage progression - if yardage stayed same or decreased slightly, 
            # likely 3rd down progression 
            if (self.last_yardage and current_yardage and 
                current_yardage <= self.last_yardage and current_yardage < 10):
                print(f"🔧 YARDAGE LOGIC CORRECTION: Expected 3rd down, yardage progression supports this")
                print(f"    Yardage: {self.last_yardage} → {current_yardage}, correcting to 3rd Down")
                return "3rd Down"
        
        if self.expected_next_down == "2nd Down" and ocr_down == "3rd Down":
            # Maybe we missed a play or there was a penalty
            print(f"⚠️ GAME LOGIC MISMATCH: Expected '2nd Down' but OCR says '3rd Down'")
            print(f"    Possible penalty or missed play - trusting OCR")
            return ocr_down
            
        # For other mismatches, log but trust OCR (could be penalties, turnovers, etc.)
        print(f"🤷 GAME LOGIC MISMATCH: Expected '{self.expected_next_down}' but OCR says '{ocr_down}'")
        print(f"    Trusting OCR (possible penalty, turnover, or missed play)")
        return ocr_down

    def detect_down_and_distance(self, frame, hud_box, frame_number=0):
        """Detect ANY down & distance using precise HUD coordinates for down/distance text."""
        try:
            import pytesseract
            import re
            import cv2
            import numpy as np
            
            # Set Tesseract path for Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Clean up old context data (remove detections older than context window)
            current_time = frame_number
            self.recent_downs = [d for d in self.recent_downs 
                               if current_time - d['frame'] <= self.context_window]
            self.recent_flags = [f for f in self.recent_flags 
                               if current_time - f['frame'] <= self.context_window]
            
            # Extract HUD region from frame using bounding box
            x1, y1, x2, y2 = hud_box.xyxy[0].cpu().numpy()
            hud_region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Use PRECISE coordinates for down/distance text (columns 15-17, full height)
            # These coordinates were confirmed by user analysis
            hud_height, hud_width = hud_region.shape[:2]
            
            # Precise down/distance region coordinates
            x_start = int(hud_width * 0.750)  # 75% across (column 15)
            x_end = int(hud_width * 0.900)    # 90% across (column 17)  
            y_start = 0                       # Full height top
            y_end = hud_height               # Full height bottom
            
            # Extract the precise down/distance region
            down_distance_region = hud_region[y_start:y_end, x_start:x_end]
            
            # Scale up the small region for better OCR (5x)
            scale_factor = 5
            scaled_width = down_distance_region.shape[1] * scale_factor
            scaled_height = down_distance_region.shape[0] * scale_factor
            scaled_region = cv2.resize(down_distance_region, (scaled_width, scaled_height), 
                                     interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray_region = cv2.cvtColor(scaled_region, cv2.COLOR_BGR2GRAY)
            
            # Apply high-contrast preprocessing for clean OCR
            # Use OTSU thresholding for automatic threshold detection
            _, thresh_region = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Perform OCR with optimal settings for Madden HUD text
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP '
            
            try:
                # Extract text using OCR
                ocr_text = pytesseract.image_to_string(thresh_region, config=custom_config).strip()
                
                if ocr_text:
                    print(f"📝 OCR extracted: '{ocr_text}'")
                    
                    # Clean and normalize the text
                    clean_text = re.sub(r'[^0-9A-Za-z&\s]', '', ocr_text).upper()
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    # PRIORITY 4: Check for 4th down patterns FIRST (for reverse detection strategy)
                    # 4th downs are easier to detect and help us find missed 3rd downs
                    fourth_down_patterns = [
                        # Standard patterns
                        r'4TH\s*&\s*\d+',        # "4th & 1", "4th & 3", etc.
                        r'4TH\s*&\s*G',          # "4th & G" (Goal)
                        r'4TH\s*&\s*GOAL',       # "4th & Goal"
                        
                        # Common OCR errors where "4th" becomes "4TH", "ATH", etc.
                        r'ATH\s*&\s*\d+',        # "ath & 1" (4 read as A)
                        r'4T\s*&\s*\d+',         # "4t & 1" (missing h)
                        r'4H\s*&\s*\d+',         # "4h & 1" (missing t)
                        r'4TH\s*\d+',            # "4th 1" (missing &)
                        r'ATH\s*\d+',            # "ath 1"
                        
                        # OCR with spacing issues and digits
                        r'4\s*TH\s*&\s*\d+',     # "4 th & 1"
                        r'\d*4TH\s*&\s*\d+',     # "44th & 1" (extra digits)
                        r'\d*ATH\s*&\s*\d+',     # "4ath & 1"
                        r'\d+TH\s*&\s*\d+',      # "4th & 1"
                        r'\d+ATH\s*\d+',         # "4ath1" (missing &)
                        
                        # More OCR variations
                        r'4THA\d+',              # "4tha1" (garbled)
                        r'ATHA\d+',              # "atha1"
                        r'T\s*TH\s*&\s*\d+',     # OCR splitting: "t th & 1"
                        
                        # Goal patterns with OCR errors
                        r'ATH\s*&\s*G',          # "ath & G" 
                        r'4TH.*GOAL',            # Any 4th down with Goal
                        r'ATH.*GOAL',            # "ath" with Goal
                        r'\d*TH.*GOAL',          # Any "th" with Goal
                    ]
                    
                    for pattern in fourth_down_patterns:
                        try:
                            if re.search(pattern, clean_text):
                                print(f"✅ 4th DOWN DETECTED! Pattern: {pattern}, Text: '{clean_text}'")
                                
                                # REVERSE DETECTION STRATEGY: Look backwards for missed 3rd down
                                self.check_for_missed_third_down(frame_number)
                                
                                # Validate with game logic
                                validated_down = self.validate_down_with_game_logic("4th Down", clean_text, frame_number)
                                
                                # Update game flow tracking
                                self.update_game_flow(validated_down, frame_number, clean_text)
                                
                                # Add to context tracking
                                self.recent_downs.append({
                                    'frame': frame_number,
                                    'down': validated_down,
                                    'text': clean_text
                                })
                                return True, validated_down
                        except Exception as pattern_error:
                            print(f"⚠️ Pattern error with {pattern}: {pattern_error}")
                            continue
                    
                    # 1ST DOWN PATTERNS (Priority for forward detection strategy)
                    first_down_patterns = [
                        # Standard patterns
                        r'1ST\s*&\s*\d+',        # "1st & 10", "1st & 15", etc.
                        r'1ST\s*&\s*G',          # "1st & G" (Goal)
                        r'1ST\s*&\s*GOAL',       # "1st & Goal"
                        
                        # Common OCR errors where "1st" becomes "tst", "lst", etc.
                        r'TST\s*&\s*\d+',        # "tst & 10" (missing 1)
                        r'LST\s*&\s*\d+',        # "lst & 10" 
                        r'ST\s*&\s*\d+',         # "st & 10" (missing 1)
                        r'IST\s*&\s*\d+',        # "ist & 10" 
                        
                        # OCR with missing "&" symbol
                        r'1ST\s*\d+',            # "1st 10" (missing &)
                        r'TST\s*\d+',            # "tst 10" 
                        r'TSTA\d+',              # "tsta10" (garbled)
                        r'1STA\d+',              # "1sta10" 
                        
                        # OCR with spacing issues and digits
                        r'1\s*ST\s*&\s*\d+',     # "1 st & 10"
                        r'\d*1ST\s*&\s*\d+',     # "31st & 10" (extra digits)
                        r'\d*TST\s*&\s*\d+',     # "3tst & 10"
                        r'\d+TST\s*\d+',         # "1tst40" (missing &)
                        r'\d+STA\d+',            # "1sta10"
                        r'\d+ST\s*&\s*\d+',      # "4st & 40"
                        
                        # More OCR variations
                        r'1TST\s*&\s*\d+',       # "1tst & 10"
                        r'1TST\s*\d+',           # "1tst10" (missing &)
                        r'T\s*ST\s*&\s*\d+',     # OCR splitting: "t st & 10"
                        
                        # Goal patterns with OCR errors
                        r'TST\s*&\s*G',          # "tst & G" 
                        r'1ST.*GOAL',            # Any 1st down with Goal
                        r'TST.*GOAL',            # "tst" with Goal
                        r'\d*ST.*GOAL',          # Any "st" with Goal
                    ]
                    
                    for pattern in first_down_patterns:
                        try:
                            if re.search(pattern, clean_text):
                                print(f"✅ 1st DOWN DETECTED! Pattern: {pattern}, Text: '{clean_text}'")
                                
                                # Validate with game logic
                                validated_down = self.validate_down_with_game_logic("1st Down", clean_text, frame_number)
                                
                                # FORWARD DETECTION STRATEGY: Check for missed 3rd down conversions
                                self.check_for_missed_third_down_forward(frame_number, validated_down)
                                
                                # Update game flow tracking
                                self.update_game_flow(validated_down, frame_number, clean_text)
                                
                                # Add to context tracking
                                self.recent_downs.append({
                                    'frame': frame_number,
                                    'down': validated_down,
                                    'text': clean_text
                                })
                                
                                # POSSESSION CHANGE DETECTION: Look for unusual patterns
                                self.detect_possession_change_scenarios(frame_number)
                                
                                return True, validated_down
                        except Exception as pattern_error:
                            print(f"⚠️ Pattern error with {pattern}: {pattern_error}")
                            continue
                    
                    # 2ND DOWN PATTERNS 
                    second_down_patterns = [
                        # Standard 2nd down patterns
                        r'2ND\s*&\s*\d+',        # "2nd & 10", "2nd & 7", etc.
                        r'2ND\s*&\s*G',          # "2nd & G" (Goal)
                        r'2ND\s*&\s*GOAL',       # "2nd & Goal"
                        
                        # Common OCR errors for "2nd" (REMOVED ambiguous "AND" patterns)
                        r'PD\s*&\s*\d+',         # "Pd & 10" 
                        r'2D\s*&\s*\d+',         # "2d & 10" (missing n)
                        
                        # OCR with missing "&" symbol
                        r'2ND\s*\d+',            # "2nd 10" (missing &)
                        r'PD\s*\d+',             # "Pd 10"
                        r'2NDA\d+',              # "2nda10" (garbled)
                        
                        # OCR with spacing issues and digits
                        r'2\s*ND\s*&\s*\d+',     # "2 nd & 10"
                        r'\d*2ND\s*&\s*\d+',     # "32nd & 10" (extra digits)
                        
                        # More OCR variations (keep specific ones like "2AND")
                        r'2AND\s*&\s*\d+',       # "2and & 10"
                        r'2AND\s*\d+',           # "2and10" (missing &)
                        
                        # Goal patterns with OCR errors
                        r'2ND.*GOAL',            # Any 2nd down with Goal
                    ]
                    
                    for pattern in second_down_patterns:
                        try:
                            if re.search(pattern, clean_text):
                                print(f"✅ 2nd DOWN DETECTED! Pattern: {pattern}, Text: '{clean_text}'")
                                
                                # Validate with game logic
                                validated_down = self.validate_down_with_game_logic("2nd Down", clean_text, frame_number)
                                
                                # Update game flow tracking
                                self.update_game_flow(validated_down, frame_number, clean_text)
                                
                                # Add to context tracking
                                self.recent_downs.append({
                                    'frame': frame_number,
                                    'down': validated_down,
                                    'text': clean_text
                                })
                                
                                return True, validated_down
                        except Exception as pattern_error:
                            print(f"⚠️ Pattern error with {pattern}: {pattern_error}")
                            continue
                    
                    # 3RD DOWN PATTERNS (Priority detection for missing 3rd downs)
                    third_down_patterns = [
                        # Standard 3rd down patterns
                        r'3RD\s*&\s*\d+',        # "3rd & 5", "3rd & 8", etc.
                        r'3RD\s*&\s*G',          # "3rd & G" (Goal)
                        r'3RD\s*&\s*GOAL',       # "3rd & Goal"
                        
                        # FLEXIBLE patterns based on actual OCR (like '323D&')
                        r'\d*3D\s*&',            # "323d&", "3d&", "23d&" etc. (flexible with extra digits)
                        r'\d*3D\s*&\s*\d*',      # "323d&5", "3d&8" etc.
                        r'\d+3D\d+',             # "323d5", "3d8" (missing &)
                        r'3\d*D\s*&',            # "32d&", "3d&" (digit between 3 and D)
                        
                        # OCR "3rd" → "3nd" misreads (CRITICAL: these get caught by 2nd down patterns!)
                        r'3ND\s*&\s*\d+',        # "3nd & 5" (3rd read as 3nd)
                        r'\d*3ND\s*&',           # "323nd&", "3nd&" 
                        r'\d*3ND\s*&\s*\d*',     # "323nd&5", "3nd&8"
                        r'3ND\s*\d+',            # "3nd5" (missing &)
                        r'\d+3ND\d+',            # "323nd5" (missing &)
                        
                        # Common OCR errors for "3rd" 
                        r'3D\s*&\s*\d+',         # "3d & 5" (missing r)
                        r'RD\s*&\s*\d+',         # "rd & 5" (missing 3)
                        r'3R\s*&\s*\d+',         # "3r & 5" (missing d)
                        r'SD\s*&\s*\d+',         # "sd & 5" (3 read as S)
                        r'3RO\s*&\s*\d+',        # "3ro & 5" (d read as o)
                        
                        # OCR with missing "&" symbol
                        r'3RD\s*\d+',            # "3rd 5" (missing &)
                        r'3D\s*\d+',             # "3d 5"
                        r'RD\s*\d+',             # "rd 5"
                        r'3RDA\d+',              # "3rda5" (garbled)
                        r'3DA\d+',               # "3da5"
                        
                        # OCR with spacing issues and digits
                        r'3\s*RD\s*&\s*\d+',     # "3 rd & 5"
                        r'3\s*D\s*&\s*\d+',      # "3 d & 5"
                        r'\d*3RD\s*&\s*\d+',     # "33rd & 5" (extra digits)
                        r'\d*3D\s*&\s*\d+',      # "33d & 5"
                        r'\d+RD\s*&\s*\d+',      # "3rd & 5"
                        r'\d+D\s*&\s*\d+',       # "3d & 5"
                        
                        # More OCR variations
                        r'3RD\s*\d+',            # "3rd5" (missing &)
                        r'3\s*RD\s*\d+',         # "3 rd5"
                        r'S\s*RD\s*&\s*\d+',     # "s rd & 5" (3 as S)
                        r'SRD\s*&\s*\d+',        # "srd & 5"
                        r'RD\s*[1-9]',           # "rd5", "rd3", "rd8" (direct spacing)
                        r'\s*RD\s*&\s*\d+',      # " rd & 5" (leading space)
                        r'SRD\s*\d+',            # "srd5" (missing &)
                        
                        # Goal patterns with OCR errors
                        r'3D\s*&\s*G',           # "3d & G" 
                        r'RD\s*&\s*G',           # "rd & G"
                        r'3RD.*GOAL',            # Any 3rd down with Goal
                        r'3D.*GOAL',             # "3d" with Goal
                        r'RD.*GOAL',             # "rd" with Goal
                    ]
                    
                    for pattern in third_down_patterns:
                        try:
                            if re.search(pattern, clean_text):
                                print(f"✅ 3rd DOWN DETECTED! Pattern: {pattern}, Text: '{clean_text}'")
                                
                                # Validate with game logic
                                validated_down = self.validate_down_with_game_logic("3rd Down", clean_text, frame_number)
                                
                                # Update game flow tracking
                                self.update_game_flow(validated_down, frame_number, clean_text)
                                
                                # Add to context tracking
                                self.recent_downs.append({
                                    'frame': frame_number,
                                    'down': validated_down,
                                    'text': clean_text
                                })
                                
                                return True, validated_down
                        except Exception as pattern_error:
                            print(f"⚠️ Pattern error with {pattern}: {pattern_error}")
                            continue
                            
            except Exception as ocr_error:
                print(f"⚠️ OCR error: {ocr_error}")
            
            return False, None
            
        except Exception as e:
            print(f"❌ Down detection error: {e}")
            return False, None

    def run(self):
        try:
            # Real SpygateAI detection using custom 5-class model
            from ultralytics import YOLO
            
            # Load SpygateAI custom model
            model_path = "../hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
            model = YOLO(model_path)
            
            # Store the model for re-analysis
            self.hud_model = model
            
            clips_data = []
            
            # Get video info
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📹 Video: {total_frames} frames at {fps} FPS")
            
            frame_count = 0
            clips_detected = 0
            last_clip_frame = 0
            
            # Process video frame by frame
            while True:
                if self.should_stop:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                frame_count += 1
                
                # Update progress every 360 frames (every 1% for 36k frame video)
                if frame_count % 360 == 0:
                    progress = int((frame_count / total_frames) * 100)
                    message = f"🔄 Progress: {progress}% (Frame {frame_count})"
                    self.progress_updated.emit(progress, message)
                    print(message)
                
                # Run detection every 180 frames (every 3 seconds at 60fps)  
                if frame_count % 180 == 0:
                    try:
                        results = model(frame, verbose=False)
                        
                        if results and len(results) > 0:
                            boxes = results[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                # Find HUD boxes for text extraction
                                hud_boxes = [box for box in boxes if int(box.cls[0]) == 0]
                                
                                if hud_boxes:
                                    # Check for down & distance using OCR analysis with context
                                    first_down_detected, situation = self.detect_down_and_distance(frame, hud_boxes[0], frame_count)
                                    
                                    # SEPARATE: Check for team scores and possession indicator
                                    scores_detected, score_info = self.detect_team_scores_and_possession(frame, hud_boxes[0], frame_count)
                                    
                                    # Only create clips for actual 1st downs with proper spacing
                                    if first_down_detected:
                                        frames_since_last_clip = frame_count - last_clip_frame
                                        required_gap = 180  # Reduced to 180 frames (3 seconds) for quick plays
                                        
                                        if clips_detected == 0 or frames_since_last_clip > required_gap:
                                            start_time = (frame_count / fps) - 2.5  # 2.5 seconds before
                                            end_time = (frame_count / fps) + 2.5   # 2.5 seconds after
                                            
                                            # Ensure times are within video bounds
                                            start_time = max(0, start_time)
                                            end_time = min(total_frames / fps, end_time)
                                            
                                            timestamp = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                                            
                                            clips_data.append({
                                                "start_frame": int(start_time * fps),
                                                "end_frame": int(end_time * fps), 
                                                "start_time": start_time,
                                                "end_time": end_time,
                                                "confidence": 0.95,  # High confidence for OCR-detected 1st downs
                                                "situation": situation,
                                            })
                                            
                                            clips_detected += 1
                                            last_clip_frame = frame_count
                                            print(f"🎬 {situation} detected at {timestamp}")
                                        else:
                                            gap_seconds = frames_since_last_clip / fps
                                            required_seconds = required_gap / fps
                                            print(f"⏱️ {situation} detected but too close to last clip ({gap_seconds:.1f}s < {required_seconds:.1f}s required)")
                                    
                    except Exception as detection_error:
                        # Continue processing even if individual frame detection fails
                        pass
            
            cap.release()
            
            # Final progress update
            self.progress_updated.emit(100, f"✅ Analysis complete: {clips_detected} clips detected")
            print(f"✅ Analysis complete: {clips_detected} clips detected")
            
            self.analysis_finished.emit(self.video_path, clips_data)

        except Exception as e:
            print(f"Analysis error: {e}")
            self.analysis_finished.emit(self.video_path, [])

    def check_for_missed_third_down_forward(self, current_frame, detected_down):
        """
        FORWARD DETECTION STRATEGY: When we detect a 1st down, look backwards 
        for a missed 3rd down that should have preceded it (3rd down conversion scenario).
        """
        if detected_down != "1st Down":
            return  # Only run this for 1st down detections
            
        print(f"🔍 FORWARD DETECTION: 1st down detected, checking for missed 3rd down conversion...")
        
        # Look back in recent frames for a 3rd down
        recent_third_downs = [d for d in self.recent_downs if d['down'] == '3rd Down']
        
        if recent_third_downs:
            most_recent_third = recent_third_downs[-1]
            frames_since_third = current_frame - most_recent_third['frame']
            print(f"    ✅ Found recent 3rd down at frame {most_recent_third['frame']} ({frames_since_third} frames ago)")
            print(f"    🏈 SCENARIO: Likely 3rd down conversion to 1st down!")
            return
        
        # No recent 3rd down found - this 1st down might have been preceded by a missed 3rd down
        print(f"    ⚠️ No recent 3rd down detected before this 1st down")
        print(f"    📚 GAME LOGIC: 1st downs can come from:")
        print(f"       - 3rd down conversions (MISSED 3rd DOWN scenario)")
        print(f"       - Penalties granting automatic first down")
        print(f"       - Start of new drive (kickoff, punt, turnover)")
        print(f"       - Beginning of game/half")
        
        # Check if we have recent flags that might explain automatic first down
        recent_flags = len(self.recent_flags) > 0
        if recent_flags:
            print(f"    🚩 PENALTY CONTEXT: Recent flags detected - could be automatic first down")
            print(f"       Penalties like PI, roughing passer → automatic 1st down")
            return
        
        # Check if we have any 2nd downs that might have been misclassified 3rd downs
        recent_second_downs = [d for d in self.recent_downs if d['down'] == '2nd Down']
        
        if recent_second_downs:
            most_recent_second = recent_second_downs[-1]
            frames_since_second = current_frame - most_recent_second['frame']
            
            # If the most recent 2nd down was close to this 1st down, it might have been a 3rd
            if frames_since_second < 1800:  # Within 30 seconds (1800 frames at 60fps)
                print(f"    🔍 POTENTIAL MISCLASSIFICATION: Found 2nd down at frame {most_recent_second['frame']}")
                print(f"       Text: '{most_recent_second['text']}'")
                print(f"       This might have been a misclassified 3rd down that converted")
                
                # Check if the text contains 3rd down indicators
                second_text = most_recent_second['text'].upper()
                if any(x in second_text for x in ['3ND', '3D', 'RD', '3R', 'SD', '3RO']):
                    print(f"    🎯 RETROACTIVE CORRECTION: Text contains 3rd down indicators!")
                    print(f"       Updating frame {most_recent_second['frame']} from '2nd Down' → '3rd Down'")
                    print(f"       SCENARIO: 3rd down conversion to 1st down!")
                    
                    # Update the misclassified detection
                    most_recent_second['down'] = '3rd Down'
                    return
        
        # Check recent down progression to see if this could be start of new drive
        if len(self.recent_downs) < 2:
            print(f"    📝 LIMITED CONTEXT: Few recent downs - likely start of new drive or game")
            return
            
        # If we get here, we might need to re-analyze for a missed 3rd down conversion
        print(f"    🔄 CONSIDERING RE-ANALYSIS: This 1st down might follow a missed 3rd down")
        print(f"    💡 COMMON SCENARIO: Team converts 3rd & short but OCR missed the 3rd down")

    def detect_possession_change_scenarios(self, current_frame):
        """
        POSSESSION CHANGE DETECTION: Look for scenarios where possession might have 
        changed after a missed 3rd down (turnovers, punts, etc.).
        """
        print(f"🔄 POSSESSION CHANGE DETECTION: Analyzing for turnover scenarios...")
        
        # Look for sudden changes in down progression that might indicate possession change
        if len(self.recent_downs) >= 2:
            last_two_downs = self.recent_downs[-2:]
            
            # Check for unexpected down jumps (like 2nd down → 1st down without progression)
            if (len(last_two_downs) == 2 and 
                last_two_downs[0]['down'] == '2nd Down' and 
                last_two_downs[1]['down'] == '1st Down'):
                
                frames_between = last_two_downs[1]['frame'] - last_two_downs[0]['frame']
                
                if frames_between > 180:  # More than 3 seconds apart
                    print(f"    🔍 SUSPICIOUS PATTERN: 2nd → 1st down with {frames_between} frames gap")
                    print(f"    💭 POSSIBLE SCENARIOS:")
                    print(f"       - Missed 3rd down, then turnover, then new drive")
                    print(f"       - Missed 3rd down, then conversion")
                    print(f"       - 2nd down was actually a misread 3rd down")
                    
                    # Check if the "2nd down" text contains 3rd down indicators
                    second_text = last_two_downs[0]['text'].upper()
                    if any(x in second_text for x in ['3ND', '3D', 'RD', '3R', 'SD', '3RO']):
                        print(f"    🎯 CORRECTION: '2nd down' text '{second_text}' contains 3rd down indicators")
                        print(f"       Updating to 3rd Down - likely conversion scenario")
                        last_two_downs[0]['down'] = '3rd Down'

    def detect_team_scores_and_possession(self, frame, hud_box, frame_number=0):
        """
        Detect team scores, abbreviations, and possession indicator from the HUD.
        This uses SEPARATE coordinates from down & distance detection.
        """
        try:
            import pytesseract
            import re
            import cv2
            
            # Set Tesseract path for Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Extract HUD region from frame using bounding box
            x1, y1, x2, y2 = hud_box.xyxy[0].cpu().numpy()
            hud_region = frame[int(y1):int(y2), int(x1):int(x2)]
            
            h, w = hud_region.shape[:2]
            
            # TEAM SCORES & POSSESSION COORDINATES (separate from down & distance)
            # Based on Madden 25 HUD layout: team info is in upper-center area
            score_x_start = 0.200  # Start after left margin to capture team abbreviations
            score_x_end = 0.700    # End before the down & distance area (0.750)
            score_y_start = 0.050  # Very top of HUD where scores are displayed  
            score_y_end = 0.350    # Upper third of HUD (don't overlap with down/distance)
            
            # Extract the team scores region
            x_start_px = int(w * score_x_start)
            x_end_px = int(w * score_x_end)
            y_start_px = int(h * score_y_start)
            y_end_px = int(h * score_y_end)
            
            score_region = hud_region[y_start_px:y_end_px, x_start_px:x_end_px]
            
            print(f"🏈 TEAM SCORES REGION: ({score_x_start}-{score_x_end}, {score_y_start}-{score_y_end})")
            print(f"    Pixel coordinates: x({x_start_px}-{x_end_px}), y({y_start_px}-{y_end_px})")
            
            # Apply preprocessing for better OCR
            gray_score = cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better text recognition
            enhanced_score = cv2.convertScaleAbs(gray_score, alpha=2.0, beta=50)
            
            # Apply threshold to get clean text
            _, thresh_score = cv2.threshold(enhanced_score, 127, 255, cv2.THRESH_BINARY)
            
            # OCR configuration optimized for team scores and short text
            score_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789→←▲▼'
            
            # Extract text from team scores region
            score_text = pytesseract.image_to_string(thresh_score, config=score_config).strip()
            
            if score_text:
                print(f"📝 TEAM SCORES OCR: '{score_text}'")
                
                # Clean and analyze the score text
                clean_score_text = re.sub(r'[^A-Za-z0-9→←▲▼\s]', '', score_text).upper()
                
                # Look for team abbreviations (3-letter codes)
                team_abbrevs = re.findall(r'[A-Z]{2,4}', clean_score_text)
                
                # Look for scores (numbers)
                scores = re.findall(r'\d+', clean_score_text)
                
                # Look for possession indicators (triangles/arrows)
                possession_indicators = []
                if '→' in score_text or '←' in score_text:
                    possession_indicators.append('horizontal_arrow')
                if '▲' in score_text or '▼' in score_text:
                    possession_indicators.append('triangle')
                
                print(f"    🏷️  Team abbreviations found: {team_abbrevs}")
                print(f"    🎯 Scores found: {scores}")
                print(f"    👉 Possession indicators: {possession_indicators}")
                
                # Store the team/score information for possession tracking
                if not hasattr(self, 'recent_scores'):
                    self.recent_scores = []
                
                self.recent_scores.append({
                    'frame': frame_number,
                    'teams': team_abbrevs,
                    'scores': scores,
                    'possession_indicators': possession_indicators,
                    'raw_text': score_text
                })
                
                # Keep only recent score data (last 10 detections)
                if len(self.recent_scores) > 10:
                    self.recent_scores = self.recent_scores[-10:]
                
                return True, {
                    'teams': team_abbrevs,
                    'scores': scores,
                    'possession_indicators': possession_indicators,
                    'text': score_text
                }
                
            else:
                print(f"    ⚠️ No team score text detected")
                return False, None
                
        except Exception as e:
            print(f"❌ Team scores detection error: {e}")
            return False, None


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
