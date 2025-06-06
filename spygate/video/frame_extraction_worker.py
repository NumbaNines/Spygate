from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Tuple, Optional
import numpy as np
from .frame_extractor import FrameExtractor

class FrameExtractionWorker(QThread):
    """
    QThread worker for extracting frames in the background.
    Emits progress and finished signals.
    """
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)      # List of (timestamp, frame)
    error = pyqtSignal(str)

    def __init__(self, video_path: str, interval: float = 1.0, start_time: float = 0.0, end: Optional[float] = None):
        super().__init__()
        self.video_path = video_path
        self.interval = interval
        self.start_time = start_time
        self.end = end
        self._should_stop = False

    def run(self):
        try:
            extractor = FrameExtractor(self.video_path)
            duration = extractor.duration
            end_time = self.end if self.end is not None else duration
            t = max(self.start_time, 0.0)
            frames: List[Tuple[float, np.ndarray]] = []
            total = int((end_time - t) / self.interval) + 1
            count = 0
            while t < end_time and not self._should_stop:
                frame = extractor.extract_frame(t)
                if frame is not None:
                    frames.append((t, frame))
                count += 1
                self.progress.emit(count, total)
                t += self.interval
            extractor.release()
            self.finished.emit(frames)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._should_stop = True 