"""
Video service for handling video import and management.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from spygate.database.config import get_engine, get_session
from spygate.database.schema import Video
from spygate.video.codec_validator import CodecValidator

from ..database.video_manager import VideoManager
from ..video.file_manager import VideoFileManager
from ..video.metadata import VideoMetadata, extract_metadata

logger = logging.getLogger(__name__)


class VideoService:
    """Service for handling video-related operations."""

    def __init__(self, video_manager: Optional[VideoManager] = None):
        """Initialize the video service.

        Args:
            video_manager: Optional VideoManager instance
        """
        self.video_manager = video_manager or VideoManager()
        self.file_manager = VideoFileManager(db_manager=self.video_manager)
        self.codec_validator = CodecValidator()
        self.engine = get_engine()

        # Ensure video storage directory exists
        self.video_dir = Path("data/videos")
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Ensure thumbnail directory exists
        self.thumbnail_dir = Path("data/thumbnails")
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def import_video(
        self,
        file_path: str,
        players: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str]:
        """
        Import a video file into the system.

        Args:
            file_path: Path to the video file
            players: List of player information dictionaries, each containing:
                    - name: Player name
                    - team: Optional team name
                    - is_self: Whether this is the user's own gameplay
                    - gamertag: Optional gamertag
                    - is_primary: Whether this is the primary player
            progress_callback: Optional callback for progress updates (0-100)
            cancel_check: Optional callback to check if operation should be cancelled

        Returns:
            tuple: (success: bool, message: str)

        Raises:
            FileNotFoundError: If the video file doesn't exist
            RuntimeError: If there's an error during import
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")

            # Extract metadata
            if progress_callback:
                progress_callback(10)

            metadata = extract_metadata(file_path)

            if cancel_check and cancel_check():
                return False, "Import cancelled"

            if progress_callback:
                progress_callback(30)

            # Validate and import file
            is_valid, error = self.file_manager.validate_file(Path(file_path))
            if not is_valid:
                raise ValueError(f"Invalid video file: {error}")

            # Check for duplicates
            duplicate_path = self.file_manager.check_duplicate(Path(file_path))
            if duplicate_path:
                raise FileExistsError(f"Duplicate video file: {duplicate_path}")

            if progress_callback:
                progress_callback(50)

            # Import file to storage
            storage_path, file_hash, file_size = self.file_manager.import_file(
                Path(file_path)
            )

            if cancel_check and cancel_check():
                # Clean up imported file
                os.remove(storage_path)
                return False, "Import cancelled"

            if progress_callback:
                progress_callback(70)

            # Update metadata with storage info
            metadata.file_path = storage_path
            metadata.file_name = storage_path.name
            metadata.file_size = file_size
            metadata.file_hash = file_hash

            # Create database entry
            try:
                video, warnings = self.video_manager.create_video(metadata, players)

                if warnings:
                    logger.warning("Import warnings: %s", warnings)

                if progress_callback:
                    progress_callback(90)

                # Update status to completed
                self.video_manager.update_video_status(video.id, "completed")

                if progress_callback:
                    progress_callback(100)

                return True, "Video imported successfully"

            except Exception as e:
                # Clean up imported file on database error
                os.remove(storage_path)
                raise RuntimeError(f"Failed to create database entry: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing video: {e}", exc_info=True)
            return False, str(e)

    def get_recent_videos(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recently imported videos.

        Args:
            limit: Maximum number of videos to return

        Returns:
            List[Dict[str, Any]]: List of video information dictionaries
        """
        videos = self.video_manager.get_recent_videos(limit)
        return [self._format_video_info(video) for video in videos]

    def get_player_videos(self, player_name: str) -> List[Dict[str, Any]]:
        """
        Get all videos for a specific player.

        Args:
            player_name: Name of the player

        Returns:
            List[Dict[str, Any]]: List of video information dictionaries
        """
        videos = self.video_manager.get_player_videos_by_name(player_name)
        return [self._format_video_info(video) for video in videos]

    def add_video_tag(self, video_id: int, tag_name: str) -> None:
        """
        Add a tag to a video.

        Args:
            video_id: ID of the video
            tag_name: Name of the tag to add
        """
        self.video_manager.add_video_tag(video_id, tag_name)

    def remove_video_tag(self, video_id: int, tag_name: str) -> None:
        """
        Remove a tag from a video.

        Args:
            video_id: ID of the video
            tag_name: Name of the tag to remove
        """
        self.video_manager.remove_video_tag(video_id, tag_name)

    def get_video_tags(self, video_id: int) -> List[str]:
        """
        Get all tags for a video.

        Args:
            video_id: ID of the video

        Returns:
            List[str]: List of tag names
        """
        return self.video_manager.get_video_tags(video_id)

    def update_video_notes(self, video_id: int, notes: str) -> None:
        """
        Update the notes for a video.

        Args:
            video_id: ID of the video
            notes: New notes text
        """
        self.video_manager.update_video_notes(video_id, notes)

    def delete_video(self, video_id: int) -> None:
        """
        Delete a video and its associated files.

        Args:
            video_id: ID of the video
        """
        video = self.video_manager.get_video_by_id(video_id)
        if not video:
            raise ValueError(f"No video found with ID {video_id}")

        # Delete the file
        self.file_manager.delete_video(video.file_path)

        # Mark as deleted in database
        self.video_manager.mark_video_deleted(video_id)

    def _format_video_info(self, video) -> Dict[str, Any]:
        """Format video information for API responses."""
        return {
            "id": video.id,
            "filename": video.filename,
            "duration": video.duration,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "import_date": video.import_date.isoformat(),
            "players": [
                {
                    "name": player.name,
                    "team": player.team,
                    "is_self": player.is_self,
                    "gamertag": player.gamertag,
                    "is_primary": assoc.is_primary,
                }
                for player, assoc in video.players
            ],
            "tags": [tag.name for tag in video.tags],
            "notes": video.notes,
        }

    def _get_storage_path(self, original_path: str) -> str:
        """
        Get the storage path for a video file.

        Args:
            original_path: Original file path

        Returns:
            str: Storage path for the video
        """
        # Create a unique filename based on timestamp and original name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(original_path)
        filename = f"{timestamp}_{original_name}"

        # Store in videos directory under data
        return os.path.join("data", "videos", filename)

    def _generate_thumbnail(self, video_path: Path) -> Optional[Path]:
        """
        Generate a thumbnail for the video.

        Args:
            video_path: Path to the video file

        Returns:
            Optional[Path]: Path to the generated thumbnail, or None if failed
        """
        try:
            import cv2

            # Open video
            cap = cv2.VideoCapture(str(video_path))

            # Read first frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read video frame for thumbnail")
                return None

            # Generate thumbnail filename
            thumbnail_filename = f"{video_path.stem}_thumb.jpg"
            thumbnail_path = self.thumbnail_dir / thumbnail_filename

            # Save thumbnail
            cv2.imwrite(str(thumbnail_path), frame)

            # Clean up
            cap.release()

            return thumbnail_path

        except Exception as e:
            logger.error(f"Error generating thumbnail: {e}", exc_info=True)
            return None
