"""Worker for handling video transcoding in the background."""

import uuid
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from spygate.services.video_service import VideoService


class VideoTranscodeWorker(QObject):
    """Worker for handling video transcoding in the background."""

    # Signals
    progress = pyqtSignal(float)  # Progress percentage (0-100)
    error = pyqtSignal(str)  # Error message
    completed = pyqtSignal()  # Transcoding completed
    cancelled = pyqtSignal()  # Transcoding cancelled

    def __init__(self):
        """Initialize the worker."""
        super().__init__()
        self.video_service = VideoService()
        self._current_transcode_id: Optional[uuid.UUID] = None

    @pyqtSlot(uuid.UUID, int, int, float, str, int, str, bool)
    def transcode(
        self,
        clip_id: uuid.UUID,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: Optional[int] = None,
        preset: Optional[str] = None,
        has_audio: bool = True,
    ):
        """Start transcoding a video.

        Args:
            clip_id: UUID of the clip to transcode
            width: Target width
            height: Target height
            fps: Target FPS
            codec: Target codec
            crf: Optional Constant Rate Factor for quality
            preset: Optional encoding preset
            has_audio: Whether to include audio
        """
        try:
            # Create transcoded clip entry
            success, error, transcoded = self.video_service.transcode_video(
                clip_id,
                width,
                height,
                fps,
                codec,
                crf=crf,
                preset=preset,
                has_audio=has_audio,
            )

            if not success:
                self.error.emit(error or "Failed to create transcode entry")
                return

            # Store current transcode ID
            self._current_transcode_id = transcoded.id

            # Start transcoding
            success, error = self.video_service.start_transcode(
                transcoded.id, progress_callback=lambda p: self.progress.emit(p)
            )

            if success:
                self.completed.emit()
            else:
                self.error.emit(error or "Transcoding failed")

        except Exception as e:
            self.error.emit(str(e))

    @pyqtSlot()
    def cancel(self):
        """Cancel the current transcoding operation."""
        if self._current_transcode_id:
            success, error = self.video_service.cancel_transcode(self._current_transcode_id)
            if success:
                self.cancelled.emit()
            else:
                self.error.emit(error or "Failed to cancel transcoding")
            self._current_transcode_id = None
