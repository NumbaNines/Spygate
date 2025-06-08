"""Worker for importing videos in the background."""

import os
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtCore import pyqtSignal as Signal

from ...services.video_service import VideoService
from ...utils.error_handler import (
    DatabaseError,
    PlayerError,
    StorageError,
    ValidationError,
    cleanup_failed_import,
    handle_import_error,
)
from ...video.codec_validator import VideoMetadata
from ...video.file_manager import VideoFileManager


class ImportStage:
    """Enumeration of import stages."""

    VALIDATION = "Validating video file"
    METADATA = "Extracting metadata"
    COPYING = "Copying file to storage"
    DATABASE = "Saving to database"
    ANALYSIS = "Running initial analysis"
    COMPLETE = "Import complete"


class ImportProgress:
    """Data class for import progress information."""

    def __init__(self, stage: str, progress: int, file_name: str):
        self.stage = stage
        self.progress = progress
        self.file_name = file_name

    @property
    def message(self) -> str:
        """Get formatted progress message."""
        return f"{self.stage}: {self.file_name}"


class VideoImportWorker(QObject):
    """Worker for importing videos in a background thread."""

    # Signals
    progress = Signal(ImportProgress)  # Import progress information
    error = Signal(str, str)  # Error title, message
    warning = Signal(str, str)  # Warning title, message
    file_started = Signal(str)  # File name
    file_finished = Signal(str, bool, str)  # File name, success, message
    all_finished = Signal()  # All imports finished

    def __init__(
        self,
        files: List[Tuple[str, VideoMetadata]],
        players: List[Dict[str, Any]],
        video_service: VideoService,
    ):
        """Initialize the worker.

        Args:
            files: List of tuples containing file paths and their metadata
            players: List of player information dictionaries
            video_service: VideoService instance for database operations
        """
        super().__init__()
        self.files = files
        self.players = players
        self.video_service = video_service
        self._is_running = False
        self._current_file: Optional[str] = None
        self._current_storage_path: Optional[str] = None
        self._current_video_id: Optional[int] = None

    def stop(self):
        """Stop the import process."""
        self._is_running = False
        if self._current_storage_path:
            cleanup_failed_import(
                storage_path=self._current_storage_path, video_id=self._current_video_id
            )

    def run(self):
        """Run the import process."""
        self._is_running = True
        total_files = len(self.files)

        for i, (file_path, metadata) in enumerate(self.files):
            if not self._is_running:
                break

            self._current_file = file_path
            file_name = os.path.basename(file_path)
            self.file_started.emit(file_name)

            try:
                # Update progress for validation stage
                self.progress.emit(
                    ImportProgress(
                        ImportStage.VALIDATION,
                        int(25 * (i + 1) / total_files),
                        file_name,
                    )
                )

                # Validate file exists and is accessible
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Video file not found: {file_path}")

                # Update progress for metadata stage
                self.progress.emit(
                    ImportProgress(
                        ImportStage.METADATA, int(50 * (i + 1) / total_files), file_name
                    )
                )

                # Import the video
                success, message = self.video_service.import_video(
                    file_path,
                    self.players,
                    lambda p: self.progress.emit(
                        ImportProgress(
                            ImportStage.COPYING if p < 75 else ImportStage.DATABASE,
                            p,
                            file_name,
                        )
                    ),
                    lambda: not self._is_running,  # Cancel check callback
                )

                if success:
                    self.file_finished.emit(file_name, True, "Import successful")
                else:
                    self.file_finished.emit(file_name, False, message)

            except Exception as e:
                # Handle the error and get user-friendly messages
                title, message = handle_import_error(e, file_path)
                self.error.emit(title, message)
                self.file_finished.emit(file_name, False, message)

            finally:
                self._current_file = None
                self._current_storage_path = None
                self._current_video_id = None

        self.all_finished.emit()
        self._is_running = False
