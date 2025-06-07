"""Video transcoding module.

This module provides functionality to transcode videos into standardized formats
optimized for streaming and playback within the application.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from .codec_validator import CodecValidator, VideoMetadata

logger = logging.getLogger(__name__)


class TranscodeError(Exception):
    """Base exception for transcoding errors."""

    pass


class TranscodeStatus(Enum):
    """Status of a transcoding operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscodeOptions:
    """Options for video transcoding.

    Attributes:
        target_codec: Target codec (e.g., 'H.264', 'H.265')
        target_resolution: Target resolution as (width, height) tuple
        target_fps: Target frames per second
        target_bitrate: Target bitrate in bits per second
        target_audio_codec: Target audio codec (e.g., 'AAC')
        target_audio_bitrate: Target audio bitrate in bits per second
        target_format: Target container format (e.g., 'mp4', 'mov')
        preserve_audio: Whether to preserve audio track
        fast_start: Whether to enable fast start for web playback
        hardware_acceleration: Whether to use hardware acceleration
    """

    target_codec: str = "H.264"
    target_resolution: Tuple[int, int] = (1920, 1080)
    target_fps: float = 30.0
    target_bitrate: int = 5000000  # 5 Mbps
    target_audio_codec: str = "AAC"
    target_audio_bitrate: int = 128000  # 128 kbps
    target_format: str = "mp4"
    preserve_audio: bool = True
    fast_start: bool = True
    hardware_acceleration: bool = True


class Transcoder:
    """Handles video transcoding operations.

    This class provides functionality to transcode videos into standardized formats
    using FFmpeg. It supports various codecs, resolutions, and optimization options.
    """

    # Default transcoding options
    DEFAULT_OPTIONS = TranscodeOptions()

    # FFmpeg codec mappings
    CODEC_MAP = {"H.264": "libx264", "H.265": "libx265", "VP9": "libvpx-vp9"}

    # FFmpeg audio codec mappings
    AUDIO_CODEC_MAP = {"AAC": "aac", "MP3": "libmp3lame", "Opus": "libopus"}

    def __init__(self):
        """Initialize the transcoder."""
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> bool:
        """Verify that FFmpeg is installed and accessible.

        Returns:
            bool: True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("FFmpeg not found or not working properly")
            return False

    def transcode(
        self,
        input_path: str,
        output_path: str,
        options: Optional[TranscodeOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> VideoMetadata:
        """Transcode a video file.

        Args:
            input_path: Path to input video file
            output_path: Path to save transcoded video
            options: TranscodeOptions instance or None to use defaults
            progress_callback: Optional callback function for progress updates

        Returns:
            VideoMetadata: Metadata of the transcoded video

        Raises:
            TranscodeError: If transcoding fails
            FileNotFoundError: If input file doesn't exist
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        options = options or self.DEFAULT_OPTIONS

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(input_path, output_path, options)

            # Run FFmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Monitor progress
            duration = None
            while True:
                line = process.stderr.readline()
                if not line:
                    break

                # Extract duration if not already known
                if not duration and "Duration:" in line:
                    duration = self._parse_duration(line)

                # Extract progress
                if duration and "time=" in line:
                    current_time = self._parse_time(line)
                    if current_time and duration:
                        progress = (current_time / duration) * 100
                        if progress_callback:
                            progress_callback(progress)

            # Wait for process to complete
            process.wait()

            if process.returncode != 0:
                error_output = process.stderr.read()
                logger.error(f"FFmpeg transcoding failed: {error_output}")
                raise TranscodeError(f"Transcoding failed: {error_output}")

            # Validate transcoded file
            is_valid, error, metadata = CodecValidator.validate_video(output_path)
            if not is_valid:
                raise TranscodeError(f"Transcoded file validation failed: {error}")

            return metadata

        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg process error: {str(e)}")
            raise TranscodeError(f"FFmpeg process error: {str(e)}") from e

        except Exception as e:
            logger.error(f"Transcoding error: {str(e)}")
            raise TranscodeError(f"Transcoding error: {str(e)}") from e

    def _build_ffmpeg_command(
        self, input_path: str, output_path: str, options: TranscodeOptions
    ) -> list:
        """Build FFmpeg command with the specified options.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            options: Transcoding options

        Returns:
            list: FFmpeg command as list of arguments
        """
        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Video codec settings
        video_codec = self.CODEC_MAP.get(options.target_codec)
        if not video_codec:
            raise TranscodeError(f"Unsupported video codec: {options.target_codec}")

        cmd.extend(["-c:v", video_codec])

        # Hardware acceleration (if enabled and available)
        if options.hardware_acceleration:
            # TODO: Add platform-specific hardware acceleration options
            pass

        # Video settings
        cmd.extend(
            [
                "-b:v",
                str(options.target_bitrate),
                "-r",
                str(options.target_fps),
                "-vf",
                f"scale={options.target_resolution[0]}:{options.target_resolution[1]},format=yuv420p",  # Force yuv420p pixel format
            ]
        )

        # Codec-specific options
        if video_codec == "libx264":
            cmd.extend(
                [
                    "-preset",
                    "medium",
                    "-profile:v",
                    "main",  # Use main profile which supports yuv420p
                    "-level:v",
                    "4.1",
                ]
            )
            if options.fast_start:
                cmd.extend(["-movflags", "+faststart"])
        elif video_codec == "libx265":
            cmd.extend(["-preset", "medium", "-x265-params", "log-level=error"])

        # Audio settings
        if options.preserve_audio:
            audio_codec = self.AUDIO_CODEC_MAP.get(options.target_audio_codec)
            if not audio_codec:
                raise TranscodeError(
                    f"Unsupported audio codec: {options.target_audio_codec}"
                )
            cmd.extend(["-c:a", audio_codec, "-b:a", str(options.target_audio_bitrate)])
        else:
            cmd.extend(["-an"])  # No audio

        # Output format
        cmd.extend(["-f", options.target_format, output_path])

        # Log the command
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        return cmd

    def _parse_duration(self, line: str) -> Optional[float]:
        """Parse duration from FFmpeg output.

        Args:
            line: Line from FFmpeg output containing duration

        Returns:
            float: Duration in seconds, or None if parsing fails
        """
        try:
            time_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = time_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
        except (IndexError, ValueError):
            return None

    def _parse_time(self, line: str) -> Optional[float]:
        """Parse current time from FFmpeg output.

        Args:
            line: Line from FFmpeg output containing time

        Returns:
            float: Current time in seconds, or None if parsing fails
        """
        try:
            time_str = line.split("time=")[1].split()[0].strip()
            h, m, s = time_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
        except (IndexError, ValueError):
            return None
