import os

import cv2
import numpy as np


def create_test_video(filename, codec, width, height, fps, duration, pattern="noise"):
    """
    Create a test video with specified parameters.

    Args:
        filename: Output filename
        codec: FourCC codec code
        width: Frame width
        height: Frame height
        fps: Frames per second
        duration: Duration in seconds
        pattern: Type of pattern to generate ('noise' or 'gradient')
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    try:
        # Generate frames
        frame_count = int(fps * duration)
        for i in range(frame_count):
            if pattern == "noise":
                # Create white noise pattern
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            else:
                # Create moving gradient pattern
                x = np.linspace(0, 255, width)
                y = np.linspace(0, 255, height)
                X, Y = np.meshgrid(x, y)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                offset = (i / frame_count) * 255
                frame[..., 0] = (X + offset) % 255  # Red channel
                frame[..., 1] = (Y + offset) % 255  # Green channel
                frame[..., 2] = ((X + Y + offset) / 2) % 255  # Blue channel
                frame = frame.astype(np.uint8)

            out.write(frame)

        print(f"Created {filename} ({codec}, {width}x{height}, {fps} fps, {duration}s)")
        return True
    except Exception as e:
        print(f"Error creating {filename}: {str(e)}")
        return False
    finally:
        out.release()


def main():
    # Create test videos directory if it doesn't exist
    os.makedirs("test_videos", exist_ok=True)

    # Test video configurations
    configs = [
        # H.264 videos with different resolutions
        ("h264_720p.mp4", "avc1", 1280, 720, 30, 2, "gradient"),
        ("h264_480p.mp4", "avc1", 640, 480, 30, 2, "noise"),
        # MPEG-4 videos (using x264 codec for better compatibility)
        ("mpeg4_480p.mp4", "X264", 640, 480, 30, 2, "gradient"),
        # Invalid codec (will be rejected by the validator)
        ("invalid_codec.avi", "DIVX", 640, 480, 30, 2, "noise"),
    ]

    # Create each test video
    for filename, codec, width, height, fps, duration, pattern in configs:
        filepath = os.path.join("test_videos", filename)
        create_test_video(filepath, codec, width, height, fps, duration, pattern)


if __name__ == "__main__":
    main()
