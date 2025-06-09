"""Utility for extracting video clips."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Utility class for extracting video clips."""

    @staticmethod
    def extract_segment(
        source_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        copy_codec: bool = False,
    ) -> tuple[bool, str | None]:
        """Extract a segment from a video file.

        Args:
            source_path: Path to the source video file
            output_path: Path to save the extracted clip
            start_time: Start time in seconds
            end_time: End time in seconds
            copy_codec: Whether to copy the codec instead of re-encoding

        Returns:
            tuple[bool, str | None]: Success status and error message if any
        """
        try:
            # Validate inputs
            if not os.path.exists(source_path):
                return False, "Source file not found"

            if not os.path.exists(os.path.dirname(output_path)):
                return False, "Output directory does not exist"

            if start_time < 0:
                return False, "Invalid start time"

            if end_time <= start_time:
                return False, "End time must be greater than start time"

            # Build FFmpeg command
            command = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-ss",
                str(start_time),
                "-i",
                source_path,
                "-t",
                str(end_time - start_time),
            ]

            if copy_codec:
                command.extend(["-c", "copy"])
            else:
                command.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                    ]
                )

            command.append(output_path)

            # Run FFmpeg
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"FFmpeg failed: {error_msg}")
                return False, error_msg

            return True, None

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to extract clip: {error_msg}")
            return False, error_msg
