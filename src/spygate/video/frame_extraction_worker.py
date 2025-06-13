"""
QThread worker for frame extraction with enhanced progress tracking and error handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .frame_extractor import FrameExtractor
from .frame_preprocessor import PreprocessingConfig

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Custom exception for frame extraction errors."""

    pass


class FrameExtractionWorker(QThread):
    """
    QThread worker for extracting frames in the background.
    Features enhanced progress tracking, error handling, and cancellation support.
    """

    # Signals for progress updates and results
    progress = pyqtSignal(int, int)  # current, total
    status = pyqtSignal(str)  # status message
    error = pyqtSignal(str)  # error message
    warning = pyqtSignal(str)  # warning message
    finished = pyqtSignal(list)  # List of (timestamp, frame)

    def __init__(
        self,
        video_path: str,
        interval: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the worker."""
        super().__init__()

        # Store parameters
        self.video_path = video_path
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.preprocessing_config = preprocessing_config
        self.cache_dir = cache_dir

        # Initialize state
        self._should_stop = False
        self._extractor: Optional[FrameExtractor] = None
        self._frames: list[tuple[float, np.ndarray]] = []

        # Ensure video path exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def run(self):
        """Execute frame extraction in the background."""
        try:
            # Initialize frame extractor
            self.status.emit("Initializing frame extractor...")
            self._extractor = FrameExtractor(
                self.video_path,
                preprocessing_config=self.preprocessing_config,
                cache_dir=self.cache_dir,
            )

            # Set up progress callback
            self._extractor.set_progress_callback(self._on_progress)

            # Extract frames
            self.status.emit("Extracting frames...")
            self._frames = []
            for timestamp, frame in self._extractor.extract_frames(
                interval=self.interval,
                start_time=self.start_time,
                end_time=self.end_time,
            ):
                if self._should_stop:
                    self.status.emit("Extraction cancelled")
                    break
                self._frames.append((timestamp, frame))

            # Clean up and emit results
            if not self._should_stop:
                self.status.emit("Extraction complete")
                self.finished.emit(self._frames)

        except FileNotFoundError as e:
            logger.error(f"Video file error: {e}")
            self.error.emit(f"Video file not found: {self.video_path}")
        except Exception as e:
            logger.error(f"Frame extraction error: {e}", exc_info=True)
            self.error.emit(f"Frame extraction failed: {str(e)}")
        finally:
            self._cleanup()

    def _on_progress(self, current: int, total: int):
        """Handle progress updates from the extractor."""
        self.progress.emit(current, total)

    def _cleanup(self):
        """Clean up resources."""
        try:
            if self._extractor:
                self._extractor.release()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def stop(self):
        """
        Stop the extraction process.
        Can be called from any thread.
        """
        self._should_stop = True
        if self._extractor:
            self._extractor.cancel_extraction()

    def get_frames(self) -> list[tuple[float, np.ndarray]]:
        """
        Get the currently extracted frames.
        Thread-safe access to partial results.
        """
        return self._frames.copy()

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()
        self._cleanup()
