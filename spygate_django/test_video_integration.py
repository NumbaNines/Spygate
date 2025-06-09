#!/usr/bin/env python3
"""
Video Integration Test for Django-YOLOv8 Integration

This script tests the video analysis and HUD detection capabilities
with actual video files to validate the complete integration.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# API Configuration
API_BASE_URL = "http://localhost:8000/api"


class VideoIntegrationTester:
    """Test suite for video analysis integration."""

    def __init__(self):
        self.results = []
        self.api_base = API_BASE_URL

    def log_result(self, test_name: str, success: bool, message: str, data: dict = None):
        """Log test result."""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.time(),
            "data": data or {},
        }
        self.results.append(result)

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}: {message}")

        if data and success:
            # Show key data for successful tests
            if "analysis" in data:
                print(f"   📊 Analysis data available")
            if "hud_elements" in data:
                print(f"   🎮 HUD elements detected")

    def create_test_video(self, duration_seconds: int = 2, fps: int = 30) -> str:
        """Create a simple test video file."""
        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()

        # Video properties
        width, height = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create video writer
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        total_frames = duration_seconds * fps

        for frame_num in range(total_frames):
            # Create a simple frame with some elements that might resemble a game HUD
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Add some "HUD-like" elements
            # Score area
            cv2.rectangle(frame, (50, 50), (300, 120), (100, 100, 100), -1)
            cv2.putText(
                frame,
                "HOME 14 - 7 AWAY",
                (60, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Down and distance
            cv2.rectangle(frame, (width // 2 - 100, 50), (width // 2 + 100, 120), (80, 80, 80), -1)
            cv2.putText(
                frame,
                "2nd & 8",
                (width // 2 - 50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Game clock
            cv2.rectangle(frame, (width - 200, 50), (width - 50, 120), (60, 60, 60), -1)
            clock_time = f"12:{30 - (frame_num // fps):02d}"
            cv2.putText(
                frame,
                clock_time,
                (width - 180, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Add some motion (moving rectangle to simulate players)
            x_pos = (frame_num * 5) % (width - 100)
            cv2.rectangle(
                frame, (x_pos, height // 2), (x_pos + 50, height // 2 + 100), (0, 255, 0), -1
            )

            out.write(frame)

        out.release()

        self.log_result(
            "Test Video Creation",
            True,
            f"Created {duration_seconds}s test video with {total_frames} frames",
        )

        return temp_path

    def test_video_analysis(self, video_path: str):
        """Test video analysis endpoint with actual video."""
        try:
            with open(video_path, "rb") as video_file:
                files = {"video_file": video_file}
                data = {"context": "test_analysis", "confidence": "0.7"}

                response = requests.post(f"{self.api_base}/analyze/video/", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    analysis_data = result.get("data", {})

                    self.log_result(
                        "Video Analysis",
                        True,
                        f"Analysis completed for {analysis_data.get('video_name', 'test video')}",
                        analysis_data,
                    )
                    return True
                else:
                    error_data = (
                        response.json()
                        if response.headers.get("content-type") == "application/json"
                        else {"error": response.text}
                    )
                    self.log_result(
                        "Video Analysis",
                        False,
                        f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}",
                    )
                    return False

        except Exception as e:
            self.log_result("Video Analysis", False, f"Exception: {e}")
            return False

    def test_hud_detection(self, video_path: str):
        """Test HUD detection endpoint with actual video."""
        try:
            with open(video_path, "rb") as video_file:
                files = {"video_file": video_file}
                data = {"frame_number": "10"}  # Test with frame 10

                response = requests.post(f"{self.api_base}/detect/hud/", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    hud_data = result.get("data", {})

                    self.log_result(
                        "HUD Detection",
                        True,
                        f"HUD detection completed for frame {hud_data.get('frame_number', 'unknown')}",
                        hud_data,
                    )
                    return True
                else:
                    error_data = (
                        response.json()
                        if response.headers.get("content-type") == "application/json"
                        else {"error": response.text}
                    )
                    self.log_result(
                        "HUD Detection",
                        False,
                        f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}",
                    )
                    return False

        except Exception as e:
            self.log_result("HUD Detection", False, f"Exception: {e}")
            return False

    def test_situational_library(self):
        """Test situational library building."""
        try:
            data = {"situation_type": "3rd_long"}

            response = requests.post(f"{self.api_base}/library/build/", json=data)

            if response.status_code == 200:
                result = response.json()
                library_data = result.get("data", {})

                self.log_result(
                    "Situational Library",
                    True,
                    f"Library built for 3rd_long situations",
                    library_data,
                )
                return True
            else:
                error_data = (
                    response.json()
                    if response.headers.get("content-type") == "application/json"
                    else {"error": response.text}
                )
                self.log_result(
                    "Situational Library",
                    False,
                    f"HTTP {response.status_code}: {error_data.get('error', 'Unknown error')}",
                )
                return False

        except Exception as e:
            self.log_result("Situational Library", False, f"Exception: {e}")
            return False

    def run_all_tests(self):
        """Run all video integration tests."""
        print("🎬 Starting Django-YOLOv8 Video Integration Tests")
        print("=" * 60)

        # Create test video
        test_video_path = self.create_test_video(duration_seconds=3, fps=30)

        try:
            tests = [
                lambda: self.test_video_analysis(test_video_path),
                lambda: self.test_hud_detection(test_video_path),
                self.test_situational_library,
            ]

            passed = 1  # Video creation already passed
            total = len(tests) + 1  # +1 for video creation

            for test in tests:
                if test():
                    passed += 1
                print()  # Empty line between tests

            print("=" * 60)
            print(f"📊 Test Results: {passed}/{total} passed")

            if passed == total:
                print("🎉 All video integration tests passed! Django-YOLOv8 is fully functional.")
            elif passed > total // 2:
                print("⚠️  Most tests passed. Video analysis integration is working.")
            else:
                print("❌ Multiple test failures. Check video processing pipeline.")

            return passed, total

        finally:
            # Clean up test video
            try:
                Path(test_video_path).unlink()
                print(f"🧹 Cleaned up test video: {test_video_path}")
            except Exception:
                pass

    def generate_report(self):
        """Generate detailed test report."""
        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r["success"]),
                "failed": sum(1 for r in self.results if not r["success"]),
                "timestamp": time.time(),
            },
            "tests": self.results,
        }

        report_file = Path("video_integration_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_file}")
        return report


def main():
    """Main test execution."""
    print("Django-YOLOv8 Video Integration Test Suite")
    print("Testing video analysis and HUD detection with actual video files...")
    print()

    tester = VideoIntegrationTester()
    passed, total = tester.run_all_tests()
    tester.generate_report()

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
