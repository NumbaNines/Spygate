"""
Video metadata extraction utilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Container for video metadata."""

    # File information
    file_path: str
    file_name: str
    file_size: int  # In bytes
    import_date: datetime

    # Video properties
    duration: float  # In seconds
    frame_count: int
    fps: float
    width: int
    height: int
    codec: str

    # Additional properties
    bit_rate: Optional[int] = None  # In bits per second
    has_audio: bool = False
    audio_codec: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        base_dict = {
            # File information
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "import_date": self.import_date.isoformat(),
            # Video properties
            "duration": self.duration,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            # Derived properties
            "resolution": f"{self.width}x{self.height}",
            "aspect_ratio": f"{self.width}:{self.height}",
            "file_size_mb": round(self.file_size / 1024 / 1024, 2),
        }

        # Add optional properties if available
        if self.bit_rate is not None:
            base_dict["bit_rate"] = self.bit_rate
            base_dict["bit_rate_mbps"] = round(self.bit_rate / 1_000_000, 2)

        base_dict["has_audio"] = self.has_audio
        if self.has_audio and self.audio_codec:
            base_dict["audio_codec"] = self.audio_codec

        return base_dict


def _validate_metadata(
    cap: cv2.VideoCapture, file_path: str
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validate extracted metadata values.

    Args:
        cap: OpenCV VideoCapture object
        file_path: Path to the video file

    Returns:
        Tuple[bool, Optional[str], Dict[str, Any]]:
            - Success flag
            - Error message if validation failed
            - Dictionary of validated properties
    """
    props = {}

    # Get basic properties
    props["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    props["fps"] = cap.get(cv2.CAP_PROP_FPS)
    props["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    props["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validate basic properties
    if props["frame_count"] <= 0:
        return False, "Invalid frame count", props
    if props["fps"] <= 0:
        return False, "Invalid frame rate", props
    if props["width"] <= 0 or props["height"] <= 0:
        return False, "Invalid video dimensions", props

    # Calculate and validate duration
    props["duration"] = props["frame_count"] / props["fps"] if props["fps"] > 0 else 0
    if props["duration"] <= 0:
        return False, "Invalid video duration", props

    # Get codec information
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    props["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    if not props["codec"]:
        return False, "Could not determine video codec", props

    # Get file information
    path = Path(file_path)
    props["file_size"] = path.stat().st_size
    if props["file_size"] <= 0:
        return False, "Invalid file size", props

    # Try to get bitrate
    props["bit_rate"] = int(cap.get(cv2.CAP_PROP_BITRATE))

    # Check for audio stream
    props["has_audio"] = bool(cap.get(cv2.CAP_PROP_AUDIO_STREAM))
    if props["has_audio"]:
        # Try to get audio codec (not all OpenCV builds support this)
        try:
            audio_codec = cap.get(cv2.CAP_PROP_AUDIO_CODEC_NAME)
            props["audio_codec"] = str(audio_codec) if audio_codec else None
        except:
            props["audio_codec"] = None

    return True, None, props


def extract_metadata(file_path: str) -> Optional[VideoMetadata]:
    """
    Extract metadata from a video file using OpenCV.

    Args:
        file_path: Path to the video file

    Returns:
        Optional[VideoMetadata]: Extracted metadata or None if extraction fails

    Raises:
        ValueError: If the video file cannot be opened or is invalid
    """
    try:
        # Validate file exists
        if not Path(file_path).exists():
            raise ValueError(f"Video file does not exist: {file_path}")

        # Open video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {file_path}")

        try:
            # Validate and extract metadata
            success, error_msg, props = _validate_metadata(cap, file_path)
            if not success:
                raise ValueError(f"Invalid video file: {error_msg}")

            # Create metadata object
            metadata = VideoMetadata(
                file_path=str(Path(file_path).absolute()),
                file_name=Path(file_path).name,
                file_size=props["file_size"],
                import_date=datetime.now(),
                duration=props["duration"],
                frame_count=props["frame_count"],
                fps=props["fps"],
                width=props["width"],
                height=props["height"],
                codec=props["codec"],
                bit_rate=props.get("bit_rate"),
                has_audio=props.get("has_audio", False),
                audio_codec=props.get("audio_codec"),
            )

            logger.info(f"Successfully extracted metadata from {file_path}")
            logger.debug(f"Metadata: {metadata.to_dict()}")

            return metadata

        finally:
            cap.release()

    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}", exc_info=True)
        return None


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"
