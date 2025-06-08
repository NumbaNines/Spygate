"""
OpenCV-based video player component for Spygate.
"""

import logging
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.hardware import HardwareDetector
from ...core.optimizer import TierOptimizer
from ...video.codec_validator import CodecValidator
from ...video.frame_extractor import FrameExtractor
from .video_timeline import VideoTimeline

logger = logging.getLogger(__name__)


class VideoPlayerWorker(QThread):
    """Worker thread for video playback."""

    frameReady = pyqtSignal(np.ndarray, float)  # frame, timestamp
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the worker thread.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.playback_speed: float = 1.0
        self.frame_interval: float = 1.0 / 30.0
        self._should_stop = False
        self._should_pause = False
        self._current_frame = 0
        self._total_frames = 0
        self._command_queue: Queue = Queue()

    def set_video(self, video_path: str) -> None:
        """Set the video file to play.

        Args:
            video_path: Path to the video file
        """
        self.video_path = video_path
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self) -> None:
        """Main worker loop."""
        try:
            if not self.video_path:
                self.error.emit("No video file specified")
                return

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error.emit(f"Failed to open video: {self.video_path}")
                return

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_interval = 1.0 / self.fps

            while not self._should_stop:
                # Process commands from the queue
                while not self._command_queue.empty():
                    cmd, *args = self._command_queue.get()
                    if cmd == "seek":
                        frame_number = args[0]
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        self._current_frame = frame_number
                    elif cmd == "speed":
                        self.playback_speed = args[0]

                if self._should_pause:
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame = 0
                    continue

                # Convert frame to RGB for Qt
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp
                timestamp = self._current_frame / self.fps

                # Emit frame
                self.frameReady.emit(frame_rgb, timestamp)

                # Update frame counter
                self._current_frame += 1

                # Control playback speed
                time.sleep(self.frame_interval / self.playback_speed)

        except Exception as e:
            self.error.emit(f"Playback error: {str(e)}")
            logger.error(f"Playback error: {str(e)}", exc_info=True)

        finally:
            if self.cap is not None:
                self.cap.release()
            self.finished.emit()

    def stop(self) -> None:
        """Stop playback."""
        self._should_stop = True

    def pause(self) -> None:
        """Pause playback."""
        self._should_pause = True

    def resume(self) -> None:
        """Resume playback."""
        self._should_pause = False

    def seek(self, frame_number: int) -> None:
        """Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to
        """
        self._command_queue.put(("seek", frame_number))

    def set_speed(self, speed: float) -> None:
        """Set playback speed.

        Args:
            speed: Playback speed multiplier
        """
        self._command_queue.put(("speed", speed))


class VideoPlayerCV(QWidget):
    """OpenCV-based video player component."""

    # Signals
    frameChanged = pyqtSignal(int)  # Current frame number
    videoLoaded = pyqtSignal(str, int, int)  # path, total frames, fps
    playbackStateChanged = pyqtSignal(bool)  # is_playing
    error = pyqtSignal(str)  # error message

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the video player.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)

        # Initialize variables
        self.current_path: Optional[str] = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.fps: float = 30.0
        self.playback_speed: float = 1.0
        self.is_playing: bool = False

        # Initialize hardware-aware components
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Create worker thread
        self.worker = VideoPlayerWorker()
        self.worker.frameReady.connect(self._on_frame_ready)
        self.worker.error.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)

        # Set up UI
        self._setup_ui()

        # Set accessibility
        self.setAccessibleName("Video Player")
        self.setAccessibleDescription("OpenCV-based video playback component")

        logger.info(f"Initialized VideoPlayerCV with {self.hardware.tier.name} tier")
        logger.info(f"Processing parameters: {self.optimizer.get_current_params()}")

    def _setup_ui(self) -> None:
        """Set up the UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create video frame
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setStyleSheet(
            """
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #3a3a3a;
            }
        """
        )
        layout.addWidget(self.video_frame)

        # Create timeline
        self.timeline = VideoTimeline()
        self.timeline.positionChanged.connect(self._on_timeline_position_changed)
        self.timeline.frameRateChanged.connect(self._on_frame_rate_changed)
        self.timeline.playbackSpeedChanged.connect(self._on_playback_speed_changed)
        layout.addWidget(self.timeline)

        # Create controls layout
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # Create playback controls
        self.play_button = QPushButton()
        self.play_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        )
        self.play_button.clicked.connect(self.toggle_playback)

        self.stop_button = QPushButton()
        self.stop_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaStop)
        )
        self.stop_button.clicked.connect(self.stop)

        # Add controls to layout
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

    def load_video(self, path: str) -> None:
        """Load a video file.

        Args:
            path: Path to the video file
        """
        try:
            # Validate video file
            valid, error_msg, metadata = CodecValidator.validate_video(path)
            if not valid:
                self.error.emit(f"Invalid video file: {error_msg}")
                return

            # Stop current playback
            self.stop()

            # Update state
            self.current_path = path
            self.total_frames = metadata.frame_count
            self.current_frame = 0
            self.fps = metadata.fps
            self.playback_speed = 1.0

            # Update timeline
            self.timeline.setVideoInfo(self.total_frames, self.fps, path)

            # Set up worker thread
            self.worker.set_video(path)

            # Emit video loaded signal
            self.videoLoaded.emit(path, self.total_frames, int(self.fps))

            logger.info(f"Loaded video: {path}")
            logger.debug(f"Video info: {metadata.to_dict()}")

        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            self.error.emit(error_msg)
            logger.error(error_msg, exc_info=True)

    def play(self) -> None:
        """Start video playback."""
        if not self.current_path:
            return

        if not self.worker.isRunning():
            self.worker.start()
        else:
            self.worker.resume()

        self.is_playing = True
        self.play_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaPause)
        )
        self.playbackStateChanged.emit(True)

    def pause(self) -> None:
        """Pause video playback."""
        if self.worker.isRunning():
            self.worker.pause()

        self.is_playing = False
        self.play_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        )
        self.playbackStateChanged.emit(False)

    def stop(self) -> None:
        """Stop video playback."""
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.is_playing = False
        self.current_frame = 0
        self.timeline.setCurrentFrame(0)
        self.play_button.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay)
        )
        self.playbackStateChanged.emit(False)

        # Clear video frame
        self.video_frame.clear()

    def toggle_playback(self) -> None:
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _on_frame_ready(self, frame: np.ndarray, timestamp: float) -> None:
        """Handle a new frame from the worker thread.

        Args:
            frame: Video frame (RGB format)
            timestamp: Frame timestamp in seconds
        """
        try:
            # Convert frame to QImage
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )

            # Create pixmap and scale to fit
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            # Update video frame
            self.video_frame.setPixmap(scaled_pixmap)

            # Update timeline
            self.timeline.setCurrentFrame(self.current_frame)

            # Emit frame changed signal
            self.frameChanged.emit(self.current_frame)

            # Update frame counter
            self.current_frame += 1

        except Exception as e:
            error_msg = f"Error displaying frame: {str(e)}"
            self.error.emit(error_msg)
            logger.error(error_msg, exc_info=True)

    def _on_worker_error(self, error_msg: str) -> None:
        """Handle worker thread errors.

        Args:
            error_msg: Error message
        """
        self.error.emit(error_msg)
        logger.error(f"Worker error: {error_msg}")
        self.stop()

    def _on_worker_finished(self) -> None:
        """Handle worker thread completion."""
        self.stop()

    def _on_timeline_position_changed(self, position: float) -> None:
        """Handle timeline position changes.

        Args:
            position: Position as a fraction (0.0 to 1.0)
        """
        frame_number = int(position * self.total_frames)
        if self.worker.isRunning():
            self.worker.seek(frame_number)
        self.current_frame = frame_number
        self.frameChanged.emit(frame_number)

    def _on_frame_rate_changed(self, fps: float) -> None:
        """Handle frame rate changes.

        Args:
            fps: New frame rate
        """
        self.fps = fps
        if self.worker.isRunning():
            self.worker.fps = fps
            self.worker.frame_interval = 1.0 / fps

    def _on_playback_speed_changed(self, speed: float) -> None:
        """Handle playback speed changes.

        Args:
            speed: New playback speed multiplier
        """
        self.playback_speed = speed
        if self.worker.isRunning():
            self.worker.set_speed(speed)

    def resizeEvent(self, event) -> None:
        """Handle widget resize events."""
        super().resizeEvent(event)
        # Update video frame size
        if self.video_frame.pixmap():
            scaled_pixmap = self.video_frame.pixmap().scaled(
                self.video_frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_frame.setPixmap(scaled_pixmap)
