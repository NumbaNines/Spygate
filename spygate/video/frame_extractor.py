import cv2
import numpy as np
from typing import Iterator, Tuple, Optional, Dict

class FrameExtractor:
    """
    Extracts frames from a video file for analysis.
    Supports single frame extraction, batch extraction, and simple frame caching.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self._frame_cache: Dict[int, np.ndarray] = {}

    def extract_frame(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at the given timestamp (in seconds).
        Uses cache if available. Returns None if frame could not be read.
        """
        frame_number = int(timestamp * self.fps)
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            self._frame_cache[frame_number] = frame
            return frame
        return None

    def extract_frames(self, interval: float = 1.0) -> Iterator[Tuple[float, np.ndarray]]:
        """
        Yield (timestamp, frame) at regular intervals (in seconds).
        Uses cache for each frame.
        """
        t = 0.0
        while t < self.duration:
            frame = self.extract_frame(t)
            if frame is not None:
                yield (t, frame)
            t += interval

    def clear_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()

    def release(self):
        """Release the video resource."""
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.release() 