"""Service for handling video file operations."""

import hashlib
import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
from sqlalchemy.orm import Session

from ..database import (
    Clip,
    TranscodedClip,
    TranscodeStatus,
    create_clip,
    create_transcoded_clip,
    get_db,
    update_transcode_status,
)
from ..video.codec_validator import CodecValidator, VideoMetadata
from ..video.transcoder import TranscodeError, TranscodeOptions, Transcoder
from .analysis_service import AnalysisService

logger = logging.getLogger(__name__)


class VideoService:
    """Service for handling video file operations."""

    UPLOAD_DIR = "uploads/videos"
    TRANSCODE_DIR = "uploads/transcoded"
    CHUNK_SIZE = 8192  # 8KB chunks for file transfer

    def __init__(self):
        """Initialize the video service."""
        # Create required directories
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.TRANSCODE_DIR, exist_ok=True)
        self.transcoder = Transcoder()
        self.analysis_service = AnalysisService()

    def upload_video(
        self,
        file_path: str,
        metadata: VideoMetadata,
        player_name: str,  # "Self" or "Opponent: <gamertag>"
        title: str,
        tags: List[str] = None,
        description: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        transcode_options: Optional[TranscodeOptions] = None,
    ) -> Tuple[uuid.UUID, str]:
        """Upload a video file and create its database entry.

        Args:
            file_path: Path to the video file
            metadata: VideoMetadata object with validated metadata
            player_name: Name of the player ("Self" or "Opponent: <gamertag>")
            title: Title for the clip
            tags: Optional list of tag names
            description: Optional description for the clip
            progress_callback: Optional callback for progress updates
            transcode_options: Optional TranscodeOptions for transcoding

        Returns:
            Tuple[uuid.UUID, str]: The clip ID and destination path

        Raises:
            OSError: If file operations fail
            Exception: If database operations fail
        """
        # Generate a unique filename
        file_ext = os.path.splitext(file_path)[1]
        clip_id = uuid.uuid4()
        unique_filename = f"{clip_id}{file_ext}"
        dest_path = os.path.join(self.UPLOAD_DIR, unique_filename)

        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)

        # Create database entry
        with get_db() as session:
            try:
                # Create clip with metadata
                clip = create_clip(
                    file_path=dest_path,
                    title=title,
                    player_name=player_name,
                    session=session,
                    tags=tags or [],
                    description=description,
                )

                # Update video metadata
                clip.duration = metadata.duration
                clip.width = metadata.width
                clip.height = metadata.height
                clip.fps = metadata.fps
                clip.codec = metadata.codec
                if metadata.bit_rate is not None:
                    clip.bitrate = metadata.bit_rate
                session.commit()

                try:
                    # Copy the file with progress updates
                    self._copy_with_progress(file_path, dest_path, progress_callback)

                    # Transcode if needed
                    if transcode_options:
                        try:
                            transcode_filename = f"{clip_id}_transcoded.mp4"
                            transcode_path = os.path.join(
                                self.TRANSCODE_DIR, transcode_filename
                            )

                            # Create transcoded clip entry
                            transcoded = create_transcoded_clip(
                                clip.id,
                                transcode_path,
                                transcode_options.width,
                                transcode_options.height,
                                transcode_options.fps,
                                transcode_options.codec,
                                session=session,
                                crf=transcode_options.crf,
                                preset=transcode_options.preset,
                                has_audio=transcode_options.has_audio,
                            )

                            # Perform transcoding
                            self.transcoder.transcode(
                                dest_path,
                                transcode_path,
                                transcode_options,
                                progress_callback,
                            )

                            # Update transcoded clip status
                            update_transcode_status(
                                transcoded.id,
                                TranscodeStatus.COMPLETED,
                                session=session,
                                progress=100.0,
                            )

                        except TranscodeError as e:
                            logger.error(
                                f"Transcoding failed for {dest_path}: {str(e)}"
                            )
                            if transcoded:
                                update_transcode_status(
                                    transcoded.id,
                                    TranscodeStatus.FAILED,
                                    session=session,
                                    error_message=str(e),
                                )
                            # Continue with import even if transcoding fails

                    # Start analysis in background
                    success, error, job_id = self.analysis_service.start_analysis(
                        str(clip.id)
                    )
                    if not success:
                        logger.error(
                            f"Failed to start analysis for clip {clip.id}: {error}"
                        )
                    else:
                        logger.info(f"Started analysis job {job_id} for clip {clip.id}")

                    return clip.id, dest_path

                except OSError as e:
                    # If file copy fails, delete clip and re-raise
                    session.delete(clip)
                    session.commit()
                    logger.error(f"Failed to upload video {file_path}: {str(e)}")
                    raise

            except Exception as e:
                # If any other error occurs, rollback and re-raise
                session.rollback()
                logger.error(f"Error during video upload: {str(e)}")
                raise

    def transcode_video(
        self,
        clip_id: uuid.UUID,
        width: int,
        height: int,
        fps: float,
        codec: str,
        crf: Optional[int] = None,
        preset: Optional[str] = None,
        has_audio: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[TranscodedClip]]:
        """Transcode a video file.

        Args:
            clip_id: UUID of the clip to transcode
            width: Target width
            height: Target height
            fps: Target FPS
            codec: Target codec
            crf: Optional Constant Rate Factor for quality
            preset: Optional encoding preset
            has_audio: Whether to include audio

        Returns:
            Tuple[bool, Optional[str], Optional[TranscodedClip]]:
                (success, error message if any, transcoded clip if created)
        """
        with get_db() as session:
            # Check if this version already exists
            existing = (
                session.query(TranscodedClip)
                .filter_by(
                    original_clip_id=clip_id,
                    width=width,
                    height=height,
                    fps=fps,
                    codec=codec,
                )
                .first()
            )

            if existing and existing.status == TranscodeStatus.COMPLETED:
                return True, None, existing

            # Get original clip
            clip = session.query(Clip).filter_by(id=clip_id).first()
            if not clip:
                return False, f"No clip found with ID {clip_id}", None

            # Generate output path
            base, ext = os.path.splitext(clip.file_path)
            output_path = f"{base}_{width}x{height}_{codec}{ext}"

            # Create transcoded clip entry
            transcoded = create_transcoded_clip(
                clip_id,
                output_path,
                width,
                height,
                fps,
                codec,
                session=session,
                crf=crf,
                preset=preset,
                has_audio=has_audio,
            )

            return True, None, transcoded

    def start_transcode(
        self, transcoded_id: uuid.UUID, progress_callback=None
    ) -> Tuple[bool, Optional[str]]:
        """Start a transcoding operation.

        Args:
            transcoded_id: UUID of the transcoded clip to process
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple[bool, str]: (success, error message if any)
        """
        with get_db() as session:
            # Get transcoded clip
            transcoded = (
                session.query(TranscodedClip).filter_by(id=transcoded_id).first()
            )
            if not transcoded:
                return False, f"No transcoded clip found with ID {transcoded_id}"

            # Get original clip
            clip = transcoded.original_clip
            if not clip:
                return (
                    False,
                    f"No original clip found for transcoded ID {transcoded_id}",
                )

            # Update status to in progress
            update_transcode_status(
                transcoded_id, TranscodeStatus.IN_PROGRESS, session=session
            )

            try:
                # Create transcode options
                options = TranscodeOptions(
                    width=transcoded.width,
                    height=transcoded.height,
                    fps=transcoded.fps,
                    codec=transcoded.codec,
                    crf=transcoded.crf,
                    preset=transcoded.preset,
                    has_audio=transcoded.has_audio,
                )

                def handle_progress(percent: float):
                    """Handle progress updates."""
                    if progress_callback:
                        progress_callback(percent)
                    with get_db() as progress_session:
                        update_transcode_status(
                            transcoded_id,
                            TranscodeStatus.IN_PROGRESS,
                            session=progress_session,
                            progress=percent,
                        )

                # Start transcoding
                self.transcoder.transcode(
                    clip.file_path, transcoded.file_path, options, handle_progress
                )

                # Update status to completed
                update_transcode_status(
                    transcoded_id,
                    TranscodeStatus.COMPLETED,
                    session=session,
                    progress=100.0,
                )

                return True, None

            except TranscodeError as e:
                error_msg = str(e)
                logger.error(f"Transcoding failed: {error_msg}")
                update_transcode_status(
                    transcoded_id,
                    TranscodeStatus.FAILED,
                    session=session,
                    error_message=error_msg,
                )
                return False, error_msg

    def cancel_transcode(self, transcoded_id: uuid.UUID) -> Tuple[bool, Optional[str]]:
        """Cancel a transcoding operation.

        Args:
            transcoded_id: UUID of the transcoded clip to cancel

        Returns:
            Tuple[bool, str]: (success, error message if any)
        """
        with get_db() as session:
            # Get transcoded clip
            transcoded = (
                session.query(TranscodedClip).filter_by(id=transcoded_id).first()
            )
            if not transcoded:
                return False, f"No transcoded clip found with ID {transcoded_id}"

            # Update status to cancelled
            update_transcode_status(
                transcoded_id, TranscodeStatus.CANCELLED, session=session
            )

            # Try to stop transcoding process
            self.transcoder.cancel(transcoded.file_path)

            return True, None

    def get_transcode_status(
        self, transcoded_id: uuid.UUID
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Get the status of a transcoding operation.

        Args:
            transcoded_id: UUID of the transcoded clip

        Returns:
            Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
                (success, error message if any, status info)
        """
        with get_db() as session:
            transcoded = (
                session.query(TranscodedClip).filter_by(id=transcoded_id).first()
            )
            if not transcoded:
                return False, f"No transcoded clip found with ID {transcoded_id}", None

            return (
                True,
                None,
                {
                    "status": transcoded.status.value,
                    "progress": transcoded.progress,
                    "error_message": transcoded.error_message,
                },
            )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex digest of the hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _copy_with_progress(
        self,
        src_path: str,
        dest_path: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        """Copy a file with progress updates.

        Args:
            src_path: Source file path
            dest_path: Destination file path
            progress_callback: Optional callback for progress updates
        """
        total_size = os.path.getsize(src_path)
        copied_size = 0

        with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
            while True:
                buf = src.read(self.CHUNK_SIZE)
                if not buf:
                    break
                dst.write(buf)
                copied_size += len(buf)
                if progress_callback:
                    progress = (copied_size / total_size) * 100
                    progress_callback(progress)
