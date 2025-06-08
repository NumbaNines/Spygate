"""Service layer for handling video imports."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

from ..database.video_import import (
    create_video_import,
    get_video_import,
    update_video_import_status,
    update_video_thumbnail,
)
from ..exceptions import VideoImportError
from ..utils.file_storage import delete_file, save_uploaded_file
from ..video.codec_validator import VideoMetadata, validate_video_file
from ..video.thumbnail_generator import generate_thumbnail

logger = logging.getLogger(__name__)


class VideoImportService:
    """Service for handling video import operations."""

    def __init__(self, upload_dir: str, thumbnail_dir: str):
        """Initialize the video import service.

        Args:
            upload_dir: Directory where uploaded videos will be stored
            thumbnail_dir: Directory where thumbnails will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.thumbnail_dir = Path(thumbnail_dir)

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def import_video(
        self,
        file_path: str,
        player_name: str,
        title: str,
        tags: Optional[list[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> tuple[UUID, str]:
        """Import a video file into the system.

        Args:
            file_path: Path to the uploaded video file
            player_name: Name of the player in the video
            title: Title for the video clip
            tags: Optional list of tags to apply
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple[UUID, str]: Clip ID and thumbnail path

        Raises:
            VideoImportError: If import fails
        """
        try:
            # Update progress
            if progress_callback:
                progress_callback("Validating video file...", 10)

            # Validate video file
            metadata = validate_video_file(file_path)

            # Update progress
            if progress_callback:
                progress_callback("Saving video file...", 30)

            # Save file to permanent storage
            permanent_path = self._get_permanent_path(file_path)
            save_uploaded_file(file_path, permanent_path)

            # Update progress
            if progress_callback:
                progress_callback("Generating thumbnail...", 50)

            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail(permanent_path, metadata)

            # Update progress
            if progress_callback:
                progress_callback("Saving to database...", 70)

            # Create database entry
            clip = create_video_import(
                file_path=str(permanent_path),
                metadata=metadata,
                player_name=player_name,
                title=title,
                tags=tags,
            )

            # Update thumbnail path
            update_video_thumbnail(clip.id, str(thumbnail_path))

            # Update progress
            if progress_callback:
                progress_callback("Import complete!", 100)

            return clip.id, str(thumbnail_path)

        except Exception as e:
            logger.error(f"Video import failed: {str(e)}")
            # Clean up any saved files
            if "permanent_path" in locals():
                delete_file(permanent_path)
            if "thumbnail_path" in locals():
                delete_file(thumbnail_path)
            # Update database if entry was created
            if "clip" in locals():
                update_video_import_status(clip.id, is_processed=False, error_message=str(e))
            raise VideoImportError(f"Failed to import video: {str(e)}")

    def cancel_import(self, clip_id: UUID) -> bool:
        """Cancel a video import and clean up resources.

        Args:
            clip_id: UUID of the clip to cancel

        Returns:
            bool: True if cancelled successfully
        """
        try:
            # Get clip info
            clip = get_video_import(clip_id)
            if not clip:
                return False

            # Delete files
            if clip.file_path:
                delete_file(clip.file_path)
            if clip.thumbnail_path:
                delete_file(clip.thumbnail_path)

            # Update database
            update_video_import_status(
                clip_id, is_processed=False, error_message="Import cancelled by user"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to cancel import: {str(e)}")
            return False

    def _get_permanent_path(self, original_path: str) -> Path:
        """Generate a permanent path for storing the video file.

        Args:
            original_path: Original path of the uploaded file

        Returns:
            Path: Path where the file should be permanently stored
        """
        filename = os.path.basename(original_path)
        return self.upload_dir / filename

    def _generate_thumbnail(self, video_path: Path, metadata: VideoMetadata) -> Path:
        """Generate a thumbnail for the video.

        Args:
            video_path: Path to the video file
            metadata: Video metadata

        Returns:
            Path: Path to the generated thumbnail

        Raises:
            VideoImportError: If thumbnail generation fails
        """
        try:
            # Generate thumbnail filename
            thumbnail_name = f"{video_path.stem}_thumb.jpg"
            thumbnail_path = self.thumbnail_dir / thumbnail_name

            # Generate thumbnail at 1 second mark
            generate_thumbnail(
                str(video_path),
                str(thumbnail_path),
                timestamp=min(1.0, metadata.duration / 2),
            )

            return thumbnail_path

        except Exception as e:
            raise VideoImportError(f"Failed to generate thumbnail: {str(e)}")
