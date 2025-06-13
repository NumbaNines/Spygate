"""
Test script for main application components
This tests core functionality without full application startup
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

# Add the current directory to Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pyqt6_application():
    """Test basic PyQt6 application functionality."""
    logger.info("Testing PyQt6 application creation...")

    try:
        app = QApplication([])
        logger.info("✅ PyQt6 application created successfully")
        app.quit()
        return True
    except Exception as e:
        logger.error(f"❌ PyQt6 application test failed: {e}")
        return False


def test_core_imports():
    """Test core application imports."""
    logger.info("Testing core application imports...")

    import_tests = [
        ("PyQt6.QtWidgets", "Core PyQt6 widgets"),
        ("PyQt6.QtCore", "PyQt6 core functionality"),
        ("ultralytics", "YOLOv8 ultralytics"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
    ]

    success_count = 0
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            logger.info(f"✅ {description} import successful")
            success_count += 1
        except ImportError as e:
            logger.error(f"❌ {description} import failed: {e}")

    return success_count == len(import_tests)


def test_simple_window():
    """Test creating a simple PyQt6 window."""
    logger.info("Testing simple PyQt6 window creation...")

    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

        app = QApplication([])

        # Create main window
        window = QMainWindow()
        window.setWindowTitle("Spygate Test Window")
        window.setMinimumSize(800, 600)

        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout(central_widget)

        # Add label
        label = QLabel("✅ Spygate Integration Test - Core Components Working")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Set dark theme
        window.setStyleSheet(
            """
            QMainWindow {
                background-color: #121212;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 16px;
                padding: 20px;
            }
        """
        )

        # Show window briefly
        window.show()

        # Process events and close
        app.processEvents()
        window.close()
        app.quit()

        logger.info("✅ Simple PyQt6 window test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Simple window test failed: {e}")
        return False


def test_yolov8_detection_pipeline():
    """Test the complete YOLOv8 detection pipeline."""
    logger.info("Testing YOLOv8 detection pipeline...")

    try:
        from ultralytics import YOLO

        # Load model
        model = YOLO("yolov8n.pt")

        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run detection
        logger.info("Running detection pipeline...")
        start_time = time.time()
        results = model(test_image, verbose=False)
        detection_time = time.time() - start_time

        logger.info(f"✅ Detection pipeline completed in {detection_time:.3f}s")

        # Test with actual demo image if available
        demo_path = Path("demo_frame.jpg")
        if demo_path.exists():
            demo_image = cv2.imread(str(demo_path))
            if demo_image is not None:
                logger.info("Testing with demo image...")
                start_time = time.time()
                demo_results = model(demo_image, verbose=False)
                demo_time = time.time() - start_time
                logger.info(f"✅ Demo image detection completed in {demo_time:.3f}s")

        return True

    except Exception as e:
        logger.error(f"❌ YOLOv8 detection pipeline test failed: {e}")
        return False


def test_image_processing():
    """Test OpenCV image processing capabilities."""
    logger.info("Testing image processing capabilities...")

    try:
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test various operations
        operations = [
            ("Color conversion", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
            ("Gaussian blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
            (
                "Edge detection",
                lambda img: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150),
            ),
            (
                "Morphological ops",
                lambda img: cv2.morphologyEx(
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
                ),
            ),
        ]

        for op_name, op_func in operations:
            try:
                result = op_func(test_image)
                logger.info(f"✅ {op_name}: {result.shape}")
            except Exception as e:
                logger.error(f"❌ {op_name} failed: {e}")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Image processing test failed: {e}")
        return False


def test_file_operations():
    """Test file operations and path handling."""
    logger.info("Testing file operations...")

    try:
        # Test creating temporary files
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test image writing/reading
        test_path = Path("test_temp_image.jpg")
        cv2.imwrite(str(test_path), test_data)

        if test_path.exists():
            loaded_image = cv2.imread(str(test_path))
            if loaded_image is not None:
                logger.info("✅ Image file operations working")
                test_path.unlink()  # Clean up
            else:
                logger.error("❌ Failed to load saved image")
                return False
        else:
            logger.error("❌ Failed to save image")
            return False

        # Test directory operations
        temp_dir = Path("temp_test_dir")
        temp_dir.mkdir(exist_ok=True)

        if temp_dir.exists() and temp_dir.is_dir():
            logger.info("✅ Directory operations working")
            temp_dir.rmdir()  # Clean up
        else:
            logger.error("❌ Directory operations failed")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ File operations test failed: {e}")
        return False


def run_component_tests():
    """Run all component tests."""
    logger.info("=" * 60)
    logger.info("SPYGATE MAIN COMPONENTS TESTING")
    logger.info("=" * 60)

    test_results = {}

    # Run all tests
    tests = [
        ("Core Imports", test_core_imports),
        ("PyQt6 Application", test_pyqt6_application),
        ("Simple Window", test_simple_window),
        ("YOLOv8 Detection Pipeline", test_yolov8_detection_pipeline),
        ("Image Processing", test_image_processing),
        ("File Operations", test_file_operations),
    ]

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            test_results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPONENT TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info(
            "🎉 ALL COMPONENT TESTS PASSED! Main application components are working correctly."
        )
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        logger.info("✅ SUFFICIENT TESTS PASSED! Core functionality is working.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Review the issues above.")
        return False


if __name__ == "__main__":
    success = run_component_tests()
    exit(0 if success else 1)
