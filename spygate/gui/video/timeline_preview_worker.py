import os
from typing import Dict, List

import cv2
from PyQt6.QtCore import QSize, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


class TimelinePreviewWorker(QThread):
    """
    Worker thread for extracting and rendering timeline preview thumbnails in the background.
    Optimized for performance and resource management.
    Guarantees that finished and error signals are always emitted.
    """

    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(dict)  # {frame_index: QPixmap}
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        frame_indices: list[int],
        thumbnail_size: QSize,
        parent=None,
        max_previews: int = 100,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.frame_indices = frame_indices[:max_previews]  # Limit number of previews
        self.thumbnail_size = thumbnail_size
        self._should_stop = False
        self._finished_emitted = False

    def run(self):
        try:
            if not os.path.isfile(self.video_path):
                self.error.emit(f"File not found: {self.video_path}")
                self._emit_finished({})
                return
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Failed to open video: {self.video_path}")
                self._emit_finished({})
                return
            previews: dict[int, QPixmap] = {}
            total = len(self.frame_indices)
            for i, frame_idx in enumerate(self.frame_indices, 1):
                if self._should_stop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue  # skip unreadable frames
                # Convert to RGB and resize efficiently
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                # Use INTER_AREA for downscaling
                thumb = cv2.resize(
                    frame_rgb,
                    (self.thumbnail_size.width(), self.thumbnail_size.height()),
                    interpolation=cv2.INTER_AREA,
                )
                qimg = QImage(
                    thumb.data,
                    thumb.shape[1],
                    thumb.shape[0],
                    thumb.strides[0],
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qimg)
                previews[frame_idx] = pixmap
                self.progress.emit(i, total)
            cap.release()
            self._emit_finished(previews)
        except Exception as e:
            self.error.emit(f"Worker error: {str(e)}")
            self._emit_finished({})

    def stop(self):
        self._should_stop = True

    def _emit_finished(self, previews):
        if not self._finished_emitted:
            self.finished.emit(previews)
            self._finished_emitted = True
