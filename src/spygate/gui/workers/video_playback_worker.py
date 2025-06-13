"""Handle video playback in a separate thread for smooth performance."""

import logging
import time
from queue import Queue
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class VideoPlaybackWorker(QThread):
    """Worker thread for video playback."""

    frameReady = pyqtSignal(np.ndarray, float)  # frame, timestamp
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the worker thread."""
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
        self._loop = True  # Whether to loop video playback

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
                    elif cmd == "loop":
                        self._loop = args[0]

                if self._should_pause:
                    time.sleep(0.01)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    if self._loop:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._current_frame = 0
                        continue
                    else:
                        # End of video
                        break

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
        self.wait()

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

    def set_loop(self, loop: bool) -> None:
        """Set whether to loop video playback.

        Args:
            loop: Whether to loop video playback
        """
        self._command_queue.put(("loop", loop))

    @property
    def current_frame(self) -> int:
        """Get the current frame number."""
        return self._current_frame

    @property
    def total_frames(self) -> int:
        """Get the total number of frames."""
        return self._total_frames

    @property
    def current_time(self) -> float:
        """Get the current playback time in seconds."""
        return self._current_frame / self.fps if self.fps > 0 else 0.0

    @property
    def total_time(self) -> float:
        """Get the total video duration in seconds."""
        return self._total_frames / self.fps if self.fps > 0 else 0.0
