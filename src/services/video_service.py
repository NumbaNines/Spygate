"""Service for handling video-related operations."""

import logging
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..database.models import ImportStatus, Video, VideoMetadata
from ..database.session import Session
from ..video.codec_validator import VideoMetadata as CodecMetadata

logger = logging.getLogger(__name__)


class VideoService:
    """Service for handling video-related operations."""

    def __init__(self):
        """Initialize the service."""
        self.upload_dir = Path("data/videos")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._preview_cache: Dict[str, Dict] = {}
        self._preview_threads: Dict[str, threading.Thread] = {}
        self._preview_queues: Dict[str, Queue] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

    def upload_video(
        self,
        file_path: str,
        metadata: CodecMetadata,
        player_name: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[uuid.UUID, str]:
        """Upload a video file and create its database entry.

        Args:
            file_path: Path to the video file
            metadata: VideoMetadata object with validated metadata
            player_name: Name of the player ("Self" or "Opponent: Name")
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple[uuid.UUID, str]: The video ID and destination path

        Raises:
            OSError: If file operations fail
            Exception: If database operations fail
        """
        # Generate unique filename
        video_id = uuid.uuid4()
        ext = Path(file_path).suffix
        dest_filename = f"{video_id}{ext}"
        dest_path = self.upload_dir / dest_filename

        try:
            # Copy file with progress updates
            file_size = os.path.getsize(file_path)
            copied_size = 0

            with open(file_path, "rb") as src, open(dest_path, "wb") as dst:
                while True:
                    buf = src.read(8192)  # 8KB chunks
                    if not buf:
                        break
                    dst.write(buf)
                    copied_size += len(buf)
                    if progress_callback:
                        progress = int((copied_size / file_size) * 100)
                        progress_callback(progress)

            # Create database entries
            with Session() as session:
                # Create video entry
                video = Video(
                    id=video_id,
                    file_path=str(dest_path),
                    original_filename=os.path.basename(file_path),
                    file_size=file_size,
                    file_hash="",  # TODO: Implement file hashing
                    player_name=player_name,
                    import_status=ImportStatus.COMPLETED,
                )
                session.add(video)

                # Create metadata entry
                video_metadata = VideoMetadata(
                    video_id=video_id,
                    width=metadata.width,
                    height=metadata.height,
                    duration=metadata.duration,
                    fps=metadata.fps,
                    codec=metadata.codec,
                )
                session.add(video_metadata)

                session.commit()

            return video_id, str(dest_path)

        except Exception as e:
            # Clean up on failure
            if dest_path.exists():
                dest_path.unlink()
            raise e

    def upload_videos(
        self,
        files: List[Tuple[str, VideoMetadata]],
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[uuid.UUID, str]]:
        """Upload multiple video files.

        Args:
            files: List of tuples containing (file_path, metadata)
            progress_callback: Optional callback for progress updates

        Returns:
            List[Tuple[uuid.UUID, str]]: List of (video_id, destination_path)
        """
        results = []
        total_size = sum(os.path.getsize(path) for path, _ in files)
        uploaded_size = 0

        for file_path, metadata in files:
            try:
                # Calculate file size for progress tracking
                file_size = os.path.getsize(file_path)

                # Create a progress wrapper that updates based on total progress
                def progress_wrapper(file_progress):
                    if progress_callback:
                        total_progress = int(
                            (
                                (uploaded_size + (file_size * file_progress / 100))
                                / total_size
                            )
                            * 100
                        )
                        progress_callback(total_progress)

                # Upload the file
                video_id, dest_path = self.upload_video(
                    file_path, metadata, progress_wrapper
                )
                results.append((video_id, dest_path))
                uploaded_size += file_size

            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {str(e)}")
                # Skip this file but continue with others
                continue

        return results

    def _copy_with_progress(
        self,
        src_path: str,
        dest_path: str,
        progress_callback: Optional[callable] = None,
    ):
        """Copy a file with progress updates.

        Args:
            src_path: Source file path
            dest_path: Destination file path
            progress_callback: Optional callback for progress updates

        Raises:
            OSError: If file operations fail
        """
        file_size = os.path.getsize(src_path)
        copied = 0

        try:
            with open(src_path, "rb") as src, open(dest_path, "wb") as dest:
                while True:
                    buf = src.read(self.CHUNK_SIZE)
                    if not buf:
                        break
                    dest.write(buf)
                    copied += len(buf)
                    if progress_callback:
                        progress = int((copied / file_size) * 100)
                        progress_callback(progress)
        except OSError as e:
            # If copy fails, try to clean up the partial file
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except:
                    pass
            raise OSError(f"Failed to copy file: {str(e)}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex digest of the file hash

        Raises:
            OSError: If file operations fail
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.CHUNK_SIZE), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def get_video_duration(self, file_path: str) -> int:
        """Get the duration of a video in seconds.

        Args:
            file_path: Path to the video file

        Returns:
            int: Duration in seconds
        """
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        return int(frame_count / fps)

    def generate_thumbnail(
        self, video_path: str, timestamp: float = 0.0
    ) -> Optional[str]:
        """Generate a thumbnail from a video at the specified timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame at {timestamp}s from {video_path}")
                return None

            # Create thumbnails directory if it doesn't exist
            thumbnails_dir = Path("data/thumbnails")
            thumbnails_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            filename = f"{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            thumbnail_path = str(thumbnails_dir / filename)

            # Save thumbnail
            cv2.imwrite(thumbnail_path, frame)

            cap.release()
            return thumbnail_path

        except Exception as e:
            logger.error(f"Error generating thumbnail: {str(e)}")
            return None

    def start_preview(self, video_path: str, callback) -> None:
        """Start generating preview frames from a video"""
        if video_path in self._preview_threads:
            return

        # Create a queue for this preview
        preview_queue = Queue()
        self._preview_queues[video_path] = preview_queue

        # Start preview thread
        thread = threading.Thread(
            target=self._preview_worker,
            args=(video_path, preview_queue, callback),
            daemon=True,
        )
        self._preview_threads[video_path] = thread
        thread.start()

    def stop_preview(self, video_path: str) -> None:
        """Stop generating preview frames for a video"""
        if video_path in self._preview_queues:
            self._preview_queues[video_path].put(None)  # Signal thread to stop

            if video_path in self._preview_threads:
                self._preview_threads[video_path].join(timeout=1.0)
                del self._preview_threads[video_path]

            del self._preview_queues[video_path]

            # Clear cache
            with self._lock:
                if video_path in self._preview_cache:
                    del self._preview_cache[video_path]

    def _preview_worker(self, video_path: str, queue: Queue, callback) -> None:
        """Worker thread for generating preview frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video for preview: {video_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / 2)  # Extract 2 frames per second
            frame_count = 0

            while True:
                # Check if we should stop
                if not queue.empty():
                    stop_signal = queue.get()
                    if stop_signal is None:
                        break

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue

                frame_count += 1
                if frame_count % frame_interval == 0:
                    # Resize frame for preview
                    height = 180  # YouTube-like preview height
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    width = int(height * aspect_ratio)
                    preview_frame = cv2.resize(frame, (width, height))

                    # Convert to RGB for Qt
                    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

                    # Cache frame
                    with self._lock:
                        if video_path not in self._preview_cache:
                            self._preview_cache[video_path] = {}
                        position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        self._preview_cache[video_path][position] = preview_frame.copy()

                    # Send frame to callback
                    callback(preview_frame, position)

        except Exception as e:
            logger.error(f"Error in preview worker: {str(e)}")
        finally:
            cap.release()

    def get_cached_preview(
        self, video_path: str, timestamp: float
    ) -> Optional[np.ndarray]:
        """Get a cached preview frame near the requested timestamp"""
        with self._lock:
            if video_path not in self._preview_cache:
                return None

            cache = self._preview_cache[video_path]
            if not cache:
                return None

            # Find the closest timestamp
            closest_ts = min(cache.keys(), key=lambda x: abs(x - timestamp))
            if abs(closest_ts - timestamp) > 1.0:  # Max 1 second difference
                return None

            return cache[closest_ts]

    def extract_frame(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract a specific frame from a video at the given timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                return None

            cap.release()
            return frame

        except Exception as e:
            logger.error(f"Error extracting frame: {str(e)}")
            return None
