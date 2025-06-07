"""Worker for importing videos in the background."""

import os
from typing import List, Tuple

from PyQt6.QtCore import QObject
from PyQt6.QtCore import pyqtSignal as Signal

from ...services.video_service import VideoService
from ...video.codec_validator import VideoMetadata


class VideoImportWorker(QObject):
    """Worker for importing videos in a background thread."""

    # Signals
    progress = Signal(int)  # Progress percentage
    error = Signal(str)  # Error message
    finished = Signal()  # Import finished

    def __init__(
        self,
        files: List[Tuple[str, VideoMetadata]],
        player_name: str,
        video_service: VideoService,
    ):
        """Initialize the worker.

        Args:
            files: List of tuples containing file paths and their metadata
            player_name: Name of the player ("Self" or "Opponent: Name")
            video_service: VideoService instance for database operations
        """
        super().__init__()
        self.files = files
        self.player_name = player_name
        self.video_service = video_service
        self._is_running = False

    def stop(self):
        """Stop the import process."""
        self._is_running = False

    def run(self):
        """Run the import process."""
        self._is_running = True
        total_files = len(self.files)

        try:
            for i, (file_path, metadata) in enumerate(self.files):
                if not self._is_running:
                    break

                try:
                    # Use the filename without extension as the title
                    title = os.path.splitext(os.path.basename(file_path))[0]
                    self.video_service.upload_video(
                        file_path=file_path,
                        metadata=metadata,
                        player_name=self.player_name,
                        title=title,
                    )
                    progress = int((i + 1) / total_files * 100)
                    self.progress.emit(progress)
                except Exception as e:
                    self.error.emit(
                        f"Error importing {os.path.basename(file_path)}: {str(e)}"
                    )
                    continue

            if self._is_running:
                self.progress.emit(100)
        except Exception as e:
            self.error.emit(f"Import error: {str(e)}")
        finally:
            self._is_running = False
            self.finished.emit()
