import logging
import os
import subprocess
import unittest

from spygate.services.video_service import VideoService
from spygate.video.transcoder import TranscodeOptions, Transcoder

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestVideoProcessing(unittest.TestCase):
    def setUp(self):
        self.video_service = VideoService()
        self.transcoder = Transcoder()
        self.test_video_path = os.path.join(
            "spygate", "tests", "data", "test_video.mp4"
        )
        self.test_output_path = os.path.join(
            "spygate", "tests", "data", "test_output.mp4"
        )

    def tearDown(self):
        # Clean up output file if it exists
        if os.path.exists(self.test_output_path):
            try:
                os.remove(self.test_output_path)
            except Exception as e:
                logger.error(f"Failed to clean up test output file: {e}")

    def test_ffmpeg_available(self):
        """Test that FFmpeg is available and working"""
        # First check if ffmpeg is in PATH
        try:
            result = subprocess.run(
                ["where", "ffmpeg"],
                capture_output=True,
                text=True,
            )
            ffmpeg_path = result.stdout.strip()
            self.assertTrue(ffmpeg_path, "FFmpeg should be found in PATH")
            logger.info(f"FFmpeg found at: {ffmpeg_path}")

            # Verify FFmpeg version
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode, 0, "FFmpeg version check should succeed"
            )
            logger.info(f"FFmpeg version info: {result.stdout.split('\\n')[0]}")

            # Verify through transcoder
            self.assertTrue(
                self.transcoder._verify_ffmpeg(),
                "FFmpeg should be available through transcoder",
            )
        except Exception as e:
            self.fail(f"FFmpeg check failed: {str(e)}")

    def test_7zip_available(self):
        """Test that 7-Zip is available"""
        seven_zip_path = "C:\\Program Files\\7-Zip\\7z.exe"
        try:
            self.assertTrue(os.path.exists(seven_zip_path), "7-Zip should be installed")
            logger.info(f"7-Zip found at: {seven_zip_path}")

            # Verify 7-Zip version
            result = subprocess.run(
                [seven_zip_path, "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, "7-Zip help command should succeed")
            logger.info("7-Zip is working properly")
        except Exception as e:
            self.fail(f"7-Zip check failed: {str(e)}")

    def test_video_transcoding(self):
        """Test video transcoding functionality"""
        # Verify test video exists
        self.assertTrue(
            os.path.exists(self.test_video_path), "Test video file should exist"
        )
        logger.info(f"Test video file found: {self.test_video_path}")

        # Log input file size
        input_size = os.path.getsize(self.test_video_path)
        logger.info(f"Input video size: {input_size} bytes")

        # Create transcoding options
        options = TranscodeOptions(
            target_codec="H.264",
            target_resolution=(640, 360),  # 720p -> 360p
            target_fps=30.0,
            target_bitrate=1000000,  # 1 Mbps
            target_format="mp4",
            preserve_audio=True,
            fast_start=True,
            hardware_acceleration=False,  # Disable hardware acceleration for testing
        )

        # Track progress
        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)
            logger.info(f"Transcoding progress: {progress:.2f}%")

        # Perform transcoding
        try:
            metadata = self.transcoder.transcode(
                self.test_video_path, self.test_output_path, options, progress_callback
            )

            # Verify output file exists
            self.assertTrue(
                os.path.exists(self.test_output_path), "Output file should exist"
            )

            # Log output file size
            output_size = os.path.getsize(self.test_output_path)
            logger.info(f"Output video size: {output_size} bytes")

            # Verify output file size is non-zero
            self.assertGreater(output_size, 0, "Output file should not be empty")

            # Verify progress tracking worked
            self.assertGreater(
                len(progress_values), 0, "Progress callback should have been called"
            )
            logger.info(f"Progress tracking: {len(progress_values)} updates received")

            # Check if transcoding made significant progress (>95% is good enough)
            self.assertGreaterEqual(
                max(progress_values),
                95,
                "Transcoding should have made significant progress",
            )

            # Verify metadata
            self.assertIsNotNone(metadata, "Metadata should be returned")
            self.assertEqual(metadata.width, 640, "Output width should match target")
            self.assertEqual(metadata.height, 360, "Output height should match target")
            logger.info(f"Output video metadata: {metadata}")

        except Exception as e:
            logger.error(f"Transcoding failed with error: {str(e)}")
            self.fail(f"Transcoding failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
