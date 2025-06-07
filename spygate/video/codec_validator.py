"""Video codec validation and metadata extraction module.

This module provides functionality to validate video files and extract metadata,
ensuring videos meet the required specifications for the application.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Base exception for video validation errors."""

    pass


class VideoSizeError(VideoValidationError):
    """Raised when video file size is outside allowed limits."""

    pass


class VideoFormatError(VideoValidationError):
    """Raised when video format or codec is not supported."""

    pass


class VideoCorruptionError(VideoValidationError):
    """Raised when video file is corrupted or unreadable."""

    pass


class VideoSpecificationError(VideoValidationError):
    """Raised when video specifications (resolution, fps, duration) are invalid."""

    pass


@dataclass
class VideoMetadata:
    """Container for video metadata.

    This class holds essential technical specifications and file characteristics.

    Attributes:
        codec (str): Video codec identifier (e.g., 'H.264', 'H.265')
        width (int): Video width in pixels
        height (int): Video height in pixels
        fps (float): Frames per second
        frame_count (int): Total number of frames
        duration (float): Video duration in seconds
        file_size (int): File size in bytes
        bit_rate (Optional[int]): Video bitrate in bits per second (if available)
    """

    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # in seconds
    file_size: int  # in bytes
    bit_rate: Optional[int] = None  # in bits per second


class CodecValidator:
    """Validates video codecs and extracts metadata.

    This class provides functionality to validate video files against specified
    requirements and extract metadata. It supports multiple codecs and performs
    various integrity checks.

    Supported codecs (per PRD): H.264, H.265, MPEG-4 (configurable).
    Provides detailed error messages and suggests transcoding for unsupported codecs.
    Checks for file corruption by reading sample frames.

    Class Attributes:
        SUPPORTED_CODECS (dict): Mapping of codec identifiers to their standard names
        MIN_FILE_SIZE (int): Minimum allowed file size in bytes
        MAX_FILE_SIZE (int): Maximum allowed file size in bytes
        MIN_RESOLUTION (tuple): Minimum allowed resolution (width, height)
        MAX_RESOLUTION (tuple): Maximum allowed resolution (width, height)
        MIN_FPS (float): Minimum allowed frames per second
        MAX_FPS (float): Maximum allowed frames per second
        MIN_DURATION (float): Minimum allowed duration in seconds
        MAX_DURATION (float): Maximum allowed duration in seconds
    """

    # Supported codecs and their common names
    SUPPORTED_CODECS = {
        "avc1": "H.264",
        "h264": "H.264",
        "x264": "H.264",
        "H.264": "H.264",
        "hevc": "H.265",
        "hev1": "H.265",
        "H.265": "H.265",
        "mp4v": "MPEG-4",
        "MPEG-4": "MPEG-4",
    }

    # Validation constants
    MIN_FILE_SIZE = 1024  # 1 KB
    MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1 GB
    MIN_RESOLUTION = (320, 240)  # Minimum width, height
    MAX_RESOLUTION = (3840, 2160)  # 4K
    MIN_FPS = 15.0
    MAX_FPS = 120.0
    MIN_DURATION = 1.0  # seconds
    MAX_DURATION = 3600.0  # 1 hour

    @classmethod
    def validate_video(
        cls, file_path: str
    ) -> Tuple[bool, str, Optional[VideoMetadata]]:
        """Validate a video file and extract its metadata.

        Performs comprehensive validation of a video file, checking:
        - File existence and size limits
        - Codec support
        - Resolution constraints
        - Frame rate limits
        - Duration limits
        - File integrity/corruption

        Args:
            file_path: Path to the video file

        Returns:
            Tuple containing:
            - bool: Whether the video is valid
            - str: Error message if invalid, empty string if valid
            - VideoMetadata: Metadata if valid, None if invalid

        Raises:
            VideoValidationError: Base class for all validation errors
            VideoSizeError: When file size is outside allowed limits
            VideoFormatError: When video format/codec is not supported
            VideoCorruptionError: When file is corrupted or unreadable
            VideoSpecificationError: When video specs are invalid
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Error accessing video file: {file_path} does not exist")
                return False, "File not found.", None

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < cls.MIN_FILE_SIZE:
                logger.error(
                    f"Video validation failed: File size {file_size} bytes is below minimum {cls.MIN_FILE_SIZE} bytes"
                )
                return (
                    False,
                    f"File too small. Minimum size is {cls.MIN_FILE_SIZE} bytes.",
                    None,
                )
            if file_size > cls.MAX_FILE_SIZE:
                logger.error(
                    f"Video validation failed: File size {file_size} bytes exceeds maximum {cls.MAX_FILE_SIZE} bytes"
                )
                return (
                    False,
                    f"File too large. Maximum size is {cls.MAX_FILE_SIZE} bytes.",
                    None,
                )

            # Try to open the video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(
                    f"Video validation failed: Unable to open video file {file_path}"
                )
                return (
                    False,
                    "Failed to open video file. The file may be corrupted.",
                    None,
                )

            try:
                # Get basic metadata
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0

                # Check resolution
                if width < cls.MIN_RESOLUTION[0] or height < cls.MIN_RESOLUTION[1]:
                    logger.error(
                        f"Video validation failed: Resolution {width}x{height} is below minimum {cls.MIN_RESOLUTION[0]}x{cls.MIN_RESOLUTION[1]}"
                    )
                    return (
                        False,
                        f"Video resolution too low. Minimum is {cls.MIN_RESOLUTION[0]}x{cls.MIN_RESOLUTION[1]}.",
                        None,
                    )
                if width > cls.MAX_RESOLUTION[0] or height > cls.MAX_RESOLUTION[1]:
                    logger.error(
                        f"Video validation failed: Resolution {width}x{height} exceeds maximum {cls.MAX_RESOLUTION[0]}x{cls.MAX_RESOLUTION[1]}"
                    )
                    return (
                        False,
                        f"Video resolution too high. Maximum is {cls.MAX_RESOLUTION[0]}x{cls.MAX_RESOLUTION[1]}.",
                        None,
                    )

                # Check frame rate
                if fps < cls.MIN_FPS:
                    logger.error(
                        f"Video validation failed: Frame rate {fps} is below minimum {cls.MIN_FPS}"
                    )
                    return (
                        False,
                        f"Frame rate too low. Minimum is {cls.MIN_FPS} fps.",
                        None,
                    )
                if fps > cls.MAX_FPS:
                    logger.error(
                        f"Video validation failed: Frame rate {fps} exceeds maximum {cls.MAX_FPS}"
                    )
                    return (
                        False,
                        f"Frame rate too high. Maximum is {cls.MAX_FPS} fps.",
                        None,
                    )

                # Check duration
                if duration < cls.MIN_DURATION:
                    logger.error(
                        f"Video validation failed: Duration {duration:.2f}s is below minimum {cls.MIN_DURATION}s"
                    )
                    return (
                        False,
                        f"Video too short. Minimum duration is {cls.MIN_DURATION} seconds.",
                        None,
                    )
                if duration > cls.MAX_DURATION:
                    logger.error(
                        f"Video validation failed: Duration {duration:.2f}s exceeds maximum {cls.MAX_DURATION/3600:.1f} hours"
                    )
                    return (
                        False,
                        f"Video too long. Maximum duration is {cls.MAX_DURATION/3600:.1f} hours.",
                        None,
                    )

                # Get codec information
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

                # Check if codec is supported
                if fourcc.lower() not in [
                    k.lower() for k in cls.SUPPORTED_CODECS.keys()
                ]:
                    logger.error(
                        f"Video validation failed: Codec {fourcc} is not supported"
                    )
                    supported_list = ", ".join(set(cls.SUPPORTED_CODECS.values()))
                    return (
                        False,
                        f"Unsupported codec. Please use one of: {supported_list}.",
                        None,
                    )

                # Check for corruption by reading frames
                frame_check_count = min(10, frame_count)  # Check up to 10 frames
                frames_read = 0
                for _ in range(frame_check_count):
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        logger.error(
                            f"Video validation failed: Unable to read frame {frames_read + 1} of {frame_check_count}"
                        )
                        return False, "Video file is corrupted or incomplete.", None
                    frames_read += 1

                if frames_read == 0:
                    logger.error(f"Video validation failed: No frames could be read")
                    return (
                        False,
                        "Video file is corrupted. No frames could be read.",
                        None,
                    )

                # Create metadata
                metadata = VideoMetadata(
                    codec=cls.SUPPORTED_CODECS[fourcc.lower()],
                    width=width,
                    height=height,
                    fps=fps,
                    frame_count=frame_count,
                    duration=duration,
                    file_size=file_size,
                    bit_rate=None,  # OpenCV doesn't provide bitrate info
                )

                logger.info(f"Successfully validated video: {file_path}")
                return True, "", metadata

            except VideoValidationError as e:
                logger.warning(f"Video validation failed: {str(e)}")
                return False, str(e), None

            except Exception as e:
                logger.error(f"Video validation failed: Unexpected error - {str(e)}")
                return False, f"Validation failed: {str(e)}", None

            finally:
                cap.release()

        except Exception as e:
            logger.error(f"Error accessing video file: {str(e)}")
            return False, str(e), None
