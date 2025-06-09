#!/usr/bin/env python3
"""
Comprehensive API Integration Test for Django-YOLOv8 Integration

This script tests the Django REST API endpoints to validate the YOLOv8 integration
and overall SpygateAI engine functionality.
"""

import json
import sys
import time
from pathlib import Path

import requests

# API Configuration
API_BASE_URL = "http://localhost:8000/api"
HEADERS = {"Content-Type": "application/json"}


class APIIntegrationTester:
    """Test suite for Django-YOLOv8 API integration."""

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

        if data and not success:
            print(f"   Error details: {data}")

    def test_health_check(self):
        """Test basic health check endpoint."""
        try:
            response = requests.get(f"{self.api_base}/health/")

            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "Health Check", True, f"Service status: {data['data']['status']}", data
                )
                return True
            else:
                self.log_result(
                    "Health Check",
                    False,
                    f"HTTP {response.status_code}",
                    {"status_code": response.status_code, "response": response.text},
                )
                return False

        except Exception as e:
            self.log_result("Health Check", False, f"Exception: {e}")
            return False

    def test_api_info(self):
        """Test API info endpoint."""
        try:
            response = requests.get(f"{self.api_base}/info/")

            if response.status_code == 200:
                data = response.json()
                endpoints = data.get("data", {}).get("endpoints", {})
                self.log_result(
                    "API Info",
                    True,
                    f"Found {len(endpoints)} endpoint categories",
                    {"endpoints": list(endpoints.keys())},
                )
                return True
            else:
                self.log_result("API Info", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_result("API Info", False, f"Exception: {e}")
            return False

    def test_engine_status(self):
        """Test engine status endpoint (may require auth)."""
        try:
            response = requests.get(f"{self.api_base}/engine/status/")

            if response.status_code == 200:
                data = response.json()
                engine_data = data.get("data", {})
                self.log_result(
                    "Engine Status",
                    True,
                    f"Engine initialized: {engine_data.get('initialized', False)}",
                    engine_data,
                )
                return True
            elif response.status_code == 401:
                self.log_result(
                    "Engine Status",
                    False,
                    "Authentication required - this is expected for protected endpoints",
                )
                return False
            else:
                self.log_result("Engine Status", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_result("Engine Status", False, f"Exception: {e}")
            return False

    def test_hardware_optimization(self):
        """Test hardware optimization endpoint."""
        try:
            response = requests.get(f"{self.api_base}/hardware/optimization/")

            if response.status_code == 200:
                data = response.json()
                self.log_result(
                    "Hardware Optimization",
                    True,
                    "Hardware optimization data retrieved",
                    data.get("data", {}),
                )
                return True
            elif response.status_code == 401:
                self.log_result("Hardware Optimization", False, "Authentication required")
                return False
            else:
                self.log_result("Hardware Optimization", False, f"HTTP {response.status_code}")
                return False

        except Exception as e:
            self.log_result("Hardware Optimization", False, f"Exception: {e}")
            return False

    def test_video_analysis_endpoint_structure(self):
        """Test video analysis endpoint structure (without actual file)."""
        try:
            # Test with no file to check error handling
            response = requests.post(f"{self.api_base}/analyze/video/")

            # We expect either 400 (no file) or 401 (auth required)
            if response.status_code == 400:
                data = response.json()
                if "video_file" in data.get("error", "").lower():
                    self.log_result(
                        "Video Analysis Structure",
                        True,
                        "Endpoint correctly validates file requirement",
                    )
                    return True
                else:
                    self.log_result(
                        "Video Analysis Structure",
                        False,
                        f"Unexpected error: {data.get('error', 'Unknown')}",
                    )
                    return False
            elif response.status_code == 401:
                self.log_result("Video Analysis Structure", False, "Authentication required")
                return False
            else:
                self.log_result(
                    "Video Analysis Structure", False, f"Unexpected status: {response.status_code}"
                )
                return False

        except Exception as e:
            self.log_result("Video Analysis Structure", False, f"Exception: {e}")
            return False

    def test_hud_detection_endpoint_structure(self):
        """Test HUD detection endpoint structure (without actual file)."""
        try:
            # Test with no file to check error handling
            response = requests.post(f"{self.api_base}/detect/hud/")

            # We expect either 400 (no file) or 401 (auth required)
            if response.status_code == 400:
                data = response.json()
                if "video_file" in data.get("error", "").lower():
                    self.log_result(
                        "HUD Detection Structure",
                        True,
                        "Endpoint correctly validates file requirement",
                    )
                    return True
                else:
                    self.log_result(
                        "HUD Detection Structure",
                        False,
                        f"Unexpected error: {data.get('error', 'Unknown')}",
                    )
                    return False
            elif response.status_code == 401:
                self.log_result("HUD Detection Structure", False, "Authentication required")
                return False
            else:
                self.log_result(
                    "HUD Detection Structure", False, f"Unexpected status: {response.status_code}"
                )
                return False

        except Exception as e:
            self.log_result("HUD Detection Structure", False, f"Exception: {e}")
            return False

    def check_yolov8_model_availability(self):
        """Check if YOLOv8 model files are available."""
        model_paths = [
            Path("yolov8m.pt"),
            Path("yolov8n.pt"),
            Path("spygate/yolov8m.pt"),
            Path("../yolov8m.pt"),
            Path("../yolov8n.pt"),
        ]

        found_models = []
        for model_path in model_paths:
            if model_path.exists():
                found_models.append(str(model_path))

        if found_models:
            self.log_result(
                "YOLOv8 Model Availability",
                True,
                f"Found {len(found_models)} model file(s)",
                {"models": found_models},
            )
            return True
        else:
            self.log_result("YOLOv8 Model Availability", False, "No YOLOv8 model files found")
            return False

    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Django-YOLOv8 API Integration Tests")
        print("=" * 60)

        tests = [
            self.test_health_check,
            self.test_api_info,
            self.check_yolov8_model_availability,
            self.test_engine_status,
            self.test_hardware_optimization,
            self.test_video_analysis_endpoint_structure,
            self.test_hud_detection_endpoint_structure,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            if test():
                passed += 1
            print()  # Empty line between tests

        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} passed")

        if passed == total:
            print("🎉 All tests passed! Django-YOLOv8 integration is working correctly.")
        elif passed > total // 2:
            print("⚠️  Most tests passed. Some endpoints may require authentication.")
        else:
            print("❌ Multiple test failures. Check Django server and configuration.")

        return passed, total

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

        report_file = Path("django_yolov8_integration_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Detailed report saved to: {report_file}")
        return report


def main():
    """Main test execution."""
    print("Django-YOLOv8 Integration Test Suite")
    print("Testing API endpoints and YOLOv8 integration...")
    print()

    tester = APIIntegrationTester()
    passed, total = tester.run_all_tests()
    tester.generate_report()

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
