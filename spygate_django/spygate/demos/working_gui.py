#!/usr/bin/env python3

"""
SpygateAI Phase 1 Working GUI Demo
==================================

A fully functional Phase 1 demo with real drag-and-drop video import.
"""

import os
import sys
from pathlib import Path

# Fix import path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("🏈 SpygateAI Phase 1 Working Demo - Initializing...")

except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install PyQt6 opencv-python")
    sys.exit(1)


class VideoDropWidget(QLabel):
    """Drag-and-drop widget for video files."""

    video_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText(
            "🎮 Drag & Drop Video Files Here\n\n📁 Or click to browse\n\nSupported: MP4, MOV, AVI"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #3B82F6;
                border-radius: 15px;
                background-color: #F0F9FF;
                min-height: 200px;
                font-size: 16px;
                padding: 20px;
                color: #1F2937;
            }
            QLabel:hover {
                background-color: #DBEAFE;
                border-color: #2563EB;
            }
        """
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self.is_video_file(urls[0].toLocalFile()):
                event.accept()
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 15px;
                        background-color: #ECFDF5;
                        min-height: 200px;
                        font-size: 16px;
                        padding: 20px;
                        color: #1F2937;
                    }
                """
                )
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QLabel {
                border: 3px dashed #3B82F6;
                border-radius: 15px;
                background-color: #F0F9FF;
                min-height: 200px;
                font-size: 16px;
                padding: 20px;
                color: #1F2937;
            }
        """
        )

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_video_file(file_path):
                self.video_dropped.emit(file_path)
                self.setText(f"✅ Loaded: {os.path.basename(file_path)}")
                self.setStyleSheet(
                    """
                    QLabel {
                        border: 3px solid #10B981;
                        border-radius: 15px;
                        background-color: #ECFDF5;
                        min-height: 200px;
                        font-size: 16px;
                        padding: 20px;
                        color: #1F2937;
                    }
                """
                )
            else:
                QMessageBox.warning(
                    self, "Invalid File", "Please drop a valid video file (MP4, MOV, AVI)"
                )

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi)"
        )
        if file_path:
            self.video_dropped.emit(file_path)
            self.setText(f"✅ Loaded: {os.path.basename(file_path)}")

    def is_video_file(self, file_path):
        return file_path.lower().endswith((".mp4", ".mov", ".avi"))


class VideoPlayer(QWidget):
    """Video player with timeline."""

    def __init__(self):
        super().__init__()
        self.current_video = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel("Load a video to begin playback")
        self.video_label.setStyleSheet(
            """
            QLabel {
                border: 2px solid #D1D5DB;
                background-color: #000;
                color: white;
                min-height: 300px;
                text-align: center;
            }
        """
        )
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Timeline
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.valueChanged.connect(self.seek_frame)
        self.timeline.setEnabled(False)
        layout.addWidget(self.timeline)

        # Frame counter
        self.frame_label = QLabel("Frame: 0 / 0")
        layout.addWidget(self.frame_label)

        self.setLayout(layout)

    def load_video(self, path):
        """Load video file."""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0

            if self.total_frames > 0:
                self.timeline.setRange(0, self.total_frames - 1)
                self.timeline.setEnabled(True)
                self.seek_frame(0)
                return True
            else:
                QMessageBox.warning(self, "Error", "Could not load video file")
                return False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
            return False

    def seek_frame(self, frame_num):
        """Seek to specific frame."""
        if not self.cap:
            return

        self.current_frame = frame_num
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = self.cap.read()
        if ret:
            # Convert to Qt format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            # Scale to fit
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            self.video_label.setPixmap(scaled_pixmap)

        self.frame_label.setText(f"Frame: {frame_num + 1} / {self.total_frames}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Phase 1 - Working Demo")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        # Header
        header = QLabel("🏈 SpygateAI Phase 1 - Working Demo with Real Drag & Drop")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("padding: 20px; background-color: #3B82F6; color: white;")
        layout.addWidget(header)

        # Splitter for drop area and video player
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Drop area
        self.drop_widget = VideoDropWidget()
        self.drop_widget.video_dropped.connect(self.load_video)
        splitter.addWidget(self.drop_widget)

        # Video player
        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter)

        # Instructions
        instructions = QLabel(
            """
📝 Instructions:
1. Drag any MP4, MOV, or AVI video file into the drop area
2. Or click the drop area to browse for files
3. Use the timeline slider to navigate through frames
4. This demonstrates real video loading and playback!
        """
        )
        instructions.setStyleSheet("padding: 15px; background-color: #F3F4F6; border-radius: 5px;")
        layout.addWidget(instructions)

        central.setLayout(layout)

        self.statusBar().showMessage("Ready - Drag video files to test real functionality!")

    def load_video(self, file_path):
        """Load video and update UI."""
        self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")

        if self.video_player.load_video(file_path):
            self.statusBar().showMessage(f"✅ Loaded: {os.path.basename(file_path)}")
            QMessageBox.information(
                self,
                "Success!",
                f"Video loaded successfully!\n\n{os.path.basename(file_path)}\n\nUse the timeline to navigate frames.",
            )
        else:
            self.statusBar().showMessage("❌ Failed to load video")


def main():
    app = QApplication(sys.argv)

    # Show startup message
    msg = QMessageBox()
    msg.setWindowTitle("SpygateAI Phase 1")
    msg.setText(
        """
🏈 SpygateAI Phase 1 Working Demo

This demo features REAL drag-and-drop functionality:

✅ Actual video file import
✅ Real video playback and timeline
✅ Frame-by-frame navigation
✅ Support for MP4, MOV, AVI files

Try dragging a video file into the demo!
    """
    )
    msg.exec()

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
