import cv2
import subprocess
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    """Container for video metadata."""
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # in seconds

class CodecValidator:
    """
    Validates video codecs and extracts metadata using both OpenCV and FFmpeg for robustness.
    Supported codecs (per PRD): H.264, H.265, MPEG-4 (configurable).
    Provides detailed error messages and suggests transcoding for unsupported codecs.
    Checks for file corruption by reading sample frames.
    """
    # Supported codecs and their common names
    SUPPORTED_CODECS = {
        'avc1': 'H.264', 'h264': 'H.264', 'x264': 'H.264', 'H.264': 'H.264',
        'hevc': 'H.265', 'hev1': 'H.265', 'H.265': 'H.265',
        'mp4v': 'MPEG-4', 'mp4v.20.9': 'MPEG-4', 'mpeg4': 'MPEG-4', 'MPEG-4': 'MPEG-4',
    }
    # FFmpeg codec names for matching
    FFMPEG_CODECS = {
        'h264': 'H.264', 'hevc': 'H.265', 'mpeg4': 'MPEG-4',
    }

    @staticmethod
    def validate_video(file_path: str) -> Tuple[bool, str, Optional[VideoMetadata]]:
        """
        Validate a video file and extract its metadata using OpenCV and FFmpeg.
        Returns:
            (is_valid, error_message, metadata)
        """
        # 1. Check file existence and permissions
        if not os.path.isfile(file_path):
            return False, "File not found.", None
        if not os.access(file_path, os.R_OK):
            return False, "Permission denied: cannot read file.", None

        # 2. Try OpenCV for basic metadata
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Failed to open video file (OpenCV).", None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).lower().strip('\0')
            # Try to read a few frames to check for corruption
            readable_frames = 0
            for _ in range(min(3, frame_count)):
                ret, _ = cap.read()
                if ret:
                    readable_frames += 1
            cap.release()
            if width <= 0 or height <= 0:
                return False, "Invalid video dimensions.", None
            if fps <= 0:
                return False, "Invalid frame rate.", None
            if frame_count <= 0:
                return False, "Invalid frame count.", None
            if readable_frames == 0:
                return False, "Video file appears to be corrupted (no readable frames).", None
        except Exception as e:
            return False, f"Error reading video with OpenCV: {str(e)}", None

        # 3. Use FFmpeg for robust codec detection
        ffmpeg_codec = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames',
                '-of', 'default=noprint_wrappers=1', file_path
            ], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            ffmpeg_info = {}
            for line in lines:
                if '=' in line:
                    k, v = line.split('=', 1)
                    ffmpeg_info[k.strip()] = v.strip()
            ffmpeg_codec = ffmpeg_info.get('codec_name', '').lower()
            # Prefer FFmpeg's width/height/fps if available
            width = int(ffmpeg_info.get('width', width))
            height = int(ffmpeg_info.get('height', height))
            # r_frame_rate is like '30000/1001'
            r_frame_rate = ffmpeg_info.get('r_frame_rate', None)
            if r_frame_rate and '/' in r_frame_rate:
                num, denom = r_frame_rate.split('/')
                try:
                    fps = float(num) / float(denom)
                except Exception:
                    pass
            nb_frames = ffmpeg_info.get('nb_frames', None)
            if nb_frames and nb_frames.isdigit():
                frame_count = int(nb_frames)
                duration = frame_count / fps if fps > 0 else duration
        except FileNotFoundError:
            # ffprobe not installed
            pass
        except Exception as e:
            # ffprobe failed, fallback to OpenCV info
            pass

        # 4. Determine codec support
        codec_name = None
        if ffmpeg_codec and ffmpeg_codec in CodecValidator.FFMPEG_CODECS:
            codec_name = CodecValidator.FFMPEG_CODECS[ffmpeg_codec]
        elif codec in CodecValidator.SUPPORTED_CODECS:
            codec_name = CodecValidator.SUPPORTED_CODECS[codec]
        else:
            # Not supported
            msg = f"Unsupported codec: {ffmpeg_codec or codec}. Supported codecs: H.264, H.265, MPEG-4."
            msg += "\nConsider transcoding your video to a supported format using a tool like HandBrake or FFmpeg."
            return False, msg, None

        # 5. Return metadata
        metadata = VideoMetadata(
            codec=codec_name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration
        )
        return True, "", metadata 