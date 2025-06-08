"""Validator for video codecs and metadata."""

from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class VideoMetadata:
    """Container for video metadata."""

    width: int
    height: int
    duration: float
    fps: float
    codec: str


class CodecValidator:
    """Validator for video codecs and metadata."""

    SUPPORTED_CODECS = ["h264", "avc1"]

    def validate(self, file_path: str) -> Optional[VideoMetadata]:
        """Validate a video file and extract its metadata.

        Args:
            file_path: Path to the video file

        Returns:
            Optional[VideoMetadata]: Metadata if valid, None if invalid

        Raises:
            Exception: If file cannot be opened or metadata cannot be read
        """
        # Open video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        try:
            # Get basic metadata
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # Get codec info
            codec_numeric = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((codec_numeric >> 8 * i) & 0xFF) for i in range(4)])

            # Validate codec
            if not any(supported in codec.lower() for supported in self.SUPPORTED_CODECS):
                return None

            return VideoMetadata(
                width=width, height=height, duration=duration, fps=fps, codec=codec
            )

        finally:
            cap.release()
