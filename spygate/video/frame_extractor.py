"""
Enhanced frame extractor with parallel processing and advanced caching.
"""

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .frame_preprocessor import FramePreprocessor, PreprocessingConfig

logger = logging.getLogger(__name__)


class FrameCache:
    """LRU cache for frames with disk backing."""

    def __init__(self, max_memory_frames: int = 1000, cache_dir: Optional[str] = None):
        """Initialize the frame cache."""
        self.max_memory_frames = max_memory_frames
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.memory_cache: Dict[int, np.ndarray] = {}
        self.frame_access_times: Dict[int, int] = {}
        self.access_count = 0
        self.lock = threading.Lock()

    def get(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a frame from cache."""
        with self.lock:
            # Try memory cache first
            if frame_number in self.memory_cache:
                self._update_access_time(frame_number)
                return self.memory_cache[frame_number]

            # Try disk cache if configured
            if self.cache_dir:
                frame_path = self._get_frame_path(frame_number)
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        self._add_to_memory_cache(frame_number, frame)
                        return frame

            return None

    def put(self, frame_number: int, frame: np.ndarray):
        """Add a frame to cache."""
        with self.lock:
            self._add_to_memory_cache(frame_number, frame)

            # Save to disk if configured
            if self.cache_dir:
                frame_path = self._get_frame_path(frame_number)
                cv2.imwrite(str(frame_path), frame)

    def _add_to_memory_cache(self, frame_number: int, frame: np.ndarray):
        """Add a frame to memory cache, managing size limits."""
        if len(self.memory_cache) >= self.max_memory_frames:
            # Remove least recently used frame
            lru_frame = min(self.frame_access_times.items(), key=lambda x: x[1])[0]
            del self.memory_cache[lru_frame]
            del self.frame_access_times[lru_frame]

        self.memory_cache[frame_number] = frame
        self._update_access_time(frame_number)

    def _update_access_time(self, frame_number: int):
        """Update the access time for a frame."""
        self.access_count += 1
        self.frame_access_times[frame_number] = self.access_count

    def _get_frame_path(self, frame_number: int) -> Path:
        """Get the path for a cached frame file."""
        return Path(self.cache_dir) / f"frame_{frame_number:08d}.jpg"

    def clear(self):
        """Clear all cached frames."""
        with self.lock:
            self.memory_cache.clear()
            self.frame_access_times.clear()
            self.access_count = 0

            if self.cache_dir:
                for file in Path(self.cache_dir).glob("frame_*.jpg"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file}: {e}")


class FrameExtractor:
    """Enhanced frame extractor with parallel processing and advanced caching."""

    def __init__(
        self,
        video_path: str,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the frame extractor."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize hardware-aware components
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.preprocessor = FramePreprocessor(preprocessing_config)

        # Configure parallel processing
        self.thread_count = self.optimizer.get_optimal_thread_count()
        self.batch_size = self.optimizer.get_optimal_batch_size()

        # Initialize caching
        cache_size = self.optimizer.get_optimal_cache_size()
        self.cache = FrameCache(cache_size, cache_dir)

        # Initialize queues for parallel processing
        self.frame_queue = Queue(maxsize=self.batch_size * 2)
        self.result_queue = Queue(maxsize=self.batch_size * 2)

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.thread_count)
        self.processing = False
        self.stop_event = threading.Event()

    def extract_frame(self, frame_number: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract and preprocess a specific frame.

        Args:
            frame_number: The frame number to extract

        Returns:
            Dict containing processed frame data or None if frame not found
        """
        # Check cache first
        cached_frame = self.cache.get(frame_number)
        if cached_frame is not None:
            return self.preprocessor.preprocess_frame(cached_frame)

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_number}")
            return None

        # Cache the raw frame
        self.cache.put(frame_number, frame)

        # Preprocess and return
        return self.preprocessor.preprocess_frame(frame)

    def extract_frames(
        self,
        start_frame: int,
        end_frame: int,
        callback: Optional[Callable[[int, Dict[str, np.ndarray]], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Extract a range of frames with parallel processing.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (inclusive)
            callback: Optional callback for processed frames
            progress_callback: Optional callback for progress updates

        Yields:
            Tuples of (frame_number, processed_frame_data)
        """
        self.processing = True
        self.stop_event.clear()

        try:
            # Start frame reading thread
            reader_thread = threading.Thread(
                target=self._read_frames, args=(start_frame, end_frame)
            )
            reader_thread.start()

            # Start processing threads
            processing_futures = []
            for _ in range(self.thread_count):
                future = self.thread_pool.submit(self._process_frames, callback)
                processing_futures.append(future)

            # Track progress
            processed_count = 0
            total_frames = end_frame - start_frame + 1

            while processed_count < total_frames and not self.stop_event.is_set():
                try:
                    frame_number, processed = self.result_queue.get(timeout=1.0)
                    processed_count += 1

                    if progress_callback:
                        progress_callback(processed_count, total_frames)

                    yield frame_number, processed

                except Queue.Empty:
                    continue

            # Clean up
            self.processing = False
            reader_thread.join()
            for future in processing_futures:
                future.result()  # Will raise any exceptions that occurred

        except Exception as e:
            logger.error(f"Error in frame extraction: {e}", exc_info=True)
            self.stop_event.set()
            raise

        finally:
            self.processing = False

    def _read_frames(self, start_frame: int, end_frame: int):
        """Read frames from video file into queue."""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

            while current_frame <= end_frame and not self.stop_event.is_set():
                # Check cache first
                cached_frame = self.cache.get(current_frame)
                if cached_frame is not None:
                    self.frame_queue.put((current_frame, cached_frame))
                    current_frame += 1
                    continue

                # Read from video if not cached
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {current_frame}")
                    break

                self.cache.put(current_frame, frame)
                self.frame_queue.put((current_frame, frame))
                current_frame += 1

        except Exception as e:
            logger.error(f"Error reading frames: {e}", exc_info=True)
            self.stop_event.set()
            raise

        finally:
            # Signal end of frames
            for _ in range(self.thread_count):
                self.frame_queue.put((None, None))

    def _process_frames(
        self, callback: Optional[Callable[[int, Dict[str, np.ndarray]], None]] = None
    ):
        """Process frames from queue."""
        try:
            while not self.stop_event.is_set():
                frame_number, frame = self.frame_queue.get()

                if frame_number is None:  # End signal
                    break

                processed = self.preprocessor.preprocess_frame(frame)

                if callback:
                    callback(frame_number, processed)

                self.result_queue.put((frame_number, processed))

        except Exception as e:
            logger.error(f"Error processing frames: {e}", exc_info=True)
            self.stop_event.set()
            raise

    def stop(self):
        """Stop frame extraction."""
        self.stop_event.set()
        self.processing = False

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self.cap.release()
        self.thread_pool.shutdown()
        self.preprocessor.cleanup()
        self.cache.clear()
