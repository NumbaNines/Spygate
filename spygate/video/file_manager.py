"""
Video file management system for handling file storage, copying, and duplicate detection.
"""

import hashlib
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..database.video_manager import VideoManager
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class VideoFileManager:
    """Manages video file storage, copying, and duplicate detection."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        db_manager: Optional[VideoManager] = None,
    ):
        """
        Initialize the video file manager.

        Args:
            storage_dir: Optional custom storage directory path. If not provided,
                       uses default from config.
            db_manager: Optional VideoManager instance for database operations.
        """
        config = get_config()
        self.storage_dir = Path(storage_dir or config["storage"]["videos_dir"])
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self.storage_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        self.max_temp_age = timedelta(hours=config["storage"]["max_temp_age_hours"])
        self.supported_formats = config["storage"]["supported_formats"]
        self.max_file_size = (
            config["storage"]["max_file_size_gb"] * 1024 * 1024 * 1024
        )  # Convert to bytes

        self.db_manager = db_manager

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_storage_path(self, file_hash: str, original_extension: str) -> Path:
        """
        Generate storage path for a video file.

        Args:
            file_hash: SHA-256 hash of the file
            original_extension: Original file extension

        Returns:
            Path: Storage path for the file
        """
        # Use first 2 characters of hash for subdirectory to prevent too many files in one directory
        subdir = self.storage_dir / file_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{file_hash}{original_extension.lower()}"

    def _clean_temp_files(self) -> None:
        """Remove temporary files older than max_temp_age."""
        current_time = datetime.now()
        for temp_file in self.temp_dir.glob("*"):
            if temp_file.is_file():
                file_age = current_time - datetime.fromtimestamp(
                    temp_file.stat().st_mtime
                )
                if file_age > self.max_temp_age:
                    logger.info(f"Removing old temp file: {temp_file}")
                    temp_file.unlink()

    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a video file before import.

        Args:
            file_path: Path to the video file

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not file_path.exists():
            return False, "File does not exist"

        if not file_path.is_file():
            return False, "Path is not a file"

        if file_path.suffix.lower() not in self.supported_formats:
            return (
                False,
                f"Unsupported file format. Supported formats: {', '.join(self.supported_formats)}",
            )

        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            return (
                False,
                f"File too large. Maximum size: {self.max_file_size / (1024*1024*1024):.1f}GB",
            )

        if file_size == 0:
            return False, "File is empty"

        return True, None

    def check_duplicate(self, file_path: Path) -> Optional[str]:
        """
        Check if a file is already in storage by calculating its hash.

        Args:
            file_path: Path to the file to check

        Returns:
            Optional[str]: Path to existing file if duplicate found, None otherwise
        """
        file_hash = self._calculate_file_hash(file_path)
        if self.db_manager:
            existing_video = self.db_manager.get_video_by_hash(file_hash)
            if existing_video and not existing_video.is_deleted:
                return str(existing_video.file_path)
        return None

    def import_file(
        self, source_path: Path, move: bool = False
    ) -> Tuple[Path, str, int]:
        """
        Import a video file into managed storage.

        Args:
            source_path: Path to the source video file
            move: If True, move the file instead of copying

        Returns:
            Tuple[Path, str, int]: (storage_path, file_hash, file_size)

        Raises:
            ValueError: If file validation fails
            FileExistsError: If file is a duplicate
            OSError: If file operations fail
        """
        # Validate file
        is_valid, error = self.validate_file(source_path)
        if not is_valid:
            raise ValueError(f"Invalid file: {error}")

        # Calculate hash and check for duplicates
        file_hash = self._calculate_file_hash(source_path)
        if self.db_manager:
            existing_video = self.db_manager.get_video_by_hash(file_hash)
            if existing_video and not existing_video.is_deleted:
                raise FileExistsError(f"Duplicate file: {existing_video.file_path}")

        # Generate storage path
        storage_path = self._get_storage_path(file_hash, source_path.suffix)

        # Copy or move file
        try:
            if move:
                shutil.move(str(source_path), str(storage_path))
            else:
                shutil.copy2(str(source_path), str(storage_path))
        except OSError as e:
            logger.error(f"Failed to {'move' if move else 'copy'} file: {e}")
            raise

        file_size = storage_path.stat().st_size
        return storage_path, file_hash, file_size

    def delete_file(self, file_path: Path, permanent: bool = False) -> None:
        """
        Delete a video file from storage.

        Args:
            file_path: Path to the file to delete
            permanent: If True, physically delete the file; if False, mark as deleted

        Raises:
            FileNotFoundError: If file doesn't exist
            OSError: If deletion fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if permanent:
            try:
                file_path.unlink()
                # Remove parent directory if empty
                parent_dir = file_path.parent
                if not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
            except OSError as e:
                logger.error(f"Failed to delete file: {e}")
                raise
        elif self.db_manager:
            self.db_manager.mark_video_deleted(str(file_path))

    def cleanup_storage(self) -> None:
        """
        Clean up storage by removing temporary files and empty directories.
        """
        # Clean temporary files
        self._clean_temp_files()

        # Remove empty subdirectories
        for subdir in self.storage_dir.glob("**/"):
            if subdir != self.temp_dir and subdir.is_dir():
                try:
                    if not any(subdir.iterdir()):
                        subdir.rmdir()
                except OSError:
                    continue

    def generate_file_path(self, original_path: str, player_name: str) -> Path:
        """
        Generate a unique file path for storing the video.

        Args:
            original_path: Original path of the video file
            player_name: Name of the player in the video

        Returns:
            Path: Generated storage path
        """
        # Create a timestamp-based directory structure
        now = datetime.now()
        date_dir = now.strftime("%Y/%m/%d")

        # Clean player name for file system
        safe_player = "".join(
            c for c in player_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_player = safe_player.replace(" ", "_")

        # Get original filename and extension
        original_name = Path(original_path).stem
        ext = Path(original_path).suffix

        # Generate unique filename
        timestamp = now.strftime("%H%M%S")
        new_filename = f"{safe_player}_{original_name}_{timestamp}{ext}"

        # Create full path
        full_path = self.storage_dir / date_dir / new_filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        return full_path

    def calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read (default: 8KB)

        Returns:
            str: Hex digest of file hash
        """
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)

        return sha256.hexdigest()

    def copy_file(
        self,
        source_path: str,
        player_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Path, List[str]]:
        """
        Copy a video file to storage with progress tracking.

        Args:
            source_path: Path to the source video file
            player_name: Name of the player in the video
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple[Path, List[str]]: (New file path, List of warnings)
        """
        warnings = []
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Check for duplicates
        if duplicate_path := self.check_duplicate(source_path):
            warnings.append(f"Duplicate of existing video: {duplicate_path}")
            return Path(duplicate_path), warnings

        # Generate destination path
        dest_path = self.generate_file_path(str(source_path), player_name)

        # Get file size for progress calculation
        file_size = source_path.stat().st_size
        copied_size = 0

        # Copy file with progress tracking
        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            while chunk := src.read(8192):
                dst.write(chunk)
                copied_size += len(chunk)

                if progress_callback:
                    progress = (copied_size / file_size) * 100
                    progress_callback(progress)

        logger.info(f"Successfully copied video to {dest_path}")
        return dest_path, warnings

    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """
        Clean up temporary files older than specified age.

        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        now = datetime.now()

        for file_path in self.temp_dir.glob("*"):
            if not file_path.is_file():
                continue

            file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_hours = (now - file_age).total_seconds() / 3600

            if age_hours > max_age_hours:
                try:
                    file_path.unlink()
                    logger.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temp file {file_path}: {e}")

    def delete_video(self, video_path: str) -> None:
        """
        Delete a video file and its associated assets.

        Args:
            video_path: Path to the video file to delete
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.warning(f"Video file not found for deletion: {video_path}")
            return

        try:
            # Delete main video file
            video_path.unlink()

            # Delete associated thumbnail if it exists
            thumbnail_path = self.thumbnails_dir / f"{video_path.stem}.jpg"
            if thumbnail_path.exists():
                thumbnail_path.unlink()

            # Delete preview if it exists
            preview_path = self.previews_dir / f"{video_path.stem}.gif"
            if preview_path.exists():
                preview_path.unlink()

            logger.info(
                f"Successfully deleted video and associated files: {video_path}"
            )

        except Exception as e:
            logger.error(f"Error deleting video {video_path}: {e}")
            raise

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics for the video directory.

        Returns:
            Dict: Storage statistics including total size, file count, etc.
        """
        total_size = 0
        video_count = 0

        for file_path in self.storage_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                if file_path.suffix.lower() in [".mp4", ".mov", ".avi"]:
                    video_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "video_count": video_count,
            "storage_path": str(self.storage_dir),
        }
