"""
Simplified YOLOv8 integration test for SpygateAI.
Tests basic YOLOv8 functionality without complex imports.
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add the spygate directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "spygate"))

# Test basic imports
try:
    import torch
    import ultralytics
    from ultralytics import YOLO

    DEPENDENCIES_AVAILABLE = True
    print(f"✓ PyTorch {torch.__version__} available")
    print(f"✓ Ultralytics {ultralytics.__version__} available")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"✗ Dependencies not available: {e}")

logger = logging.getLogger(__name__)


def create_test_image():
    """Create a test image for detection."""
    # Create a simple test image (640x480, 3 channels)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some basic shapes to simulate objects
    cv2.rectangle(image, (50, 50), (200, 100), (255, 255, 255), -1)  # White rectangle
    cv2.rectangle(image, (300, 400), (450, 450), (128, 128, 128), -1)  # Gray rectangle
    cv2.circle(image, (500, 100), 30, (255, 255, 255), -1)  # White circle

    return image


class TestYOLOv8BasicFunctionality:
    """Basic YOLOv8 functionality tests."""

    @pytest.fixture(scope="class")
    def test_image(self):
        """Create a test image for detection (pytest fixture)."""
        return create_test_image()

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="YOLOv8 dependencies not available")
    def test_yolo_model_loading(self):
        """Test that YOLOv8 model can be loaded."""
        try:
            # Load the smallest YOLOv8 model
            model = YOLO("yolov8n.pt")
            assert model is not None
            print("✓ YOLOv8n model loaded successfully")

            # Check model properties
            assert hasattr(model, "model")
            assert hasattr(model, "predict")
            print("✓ Model has required attributes")

        except Exception as e:
            pytest.fail(f"Failed to load YOLOv8 model: {e}")

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="YOLOv8 dependencies not available")
    def test_yolo_inference(self, test_image):
        """Test basic YOLOv8 inference."""
        try:
            # Load model
            model = YOLO("yolov8n.pt")

            # Run inference
            start_time = time.time()
            results = model.predict(test_image, verbose=False)
            inference_time = time.time() - start_time

            # Check results
            assert results is not None
            assert len(results) > 0
            print(f"✓ Inference completed in {inference_time:.3f} seconds")

            # Check result structure
            result = results[0]
            assert hasattr(result, "boxes")
            print(f"✓ Result has boxes attribute")

            # Log detection count
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"✓ Detected {num_detections} objects")

        except Exception as e:
            pytest.fail(f"YOLOv8 inference failed: {e}")

    def run_yolo_inference_test(self, test_image):
        """Run YOLOv8 inference test (non-pytest version)."""
        try:
            # Load model
            model = YOLO("yolov8n.pt")

            # Run inference
            start_time = time.time()
            results = model.predict(test_image, verbose=False)
            inference_time = time.time() - start_time

            # Check results
            assert results is not None
            assert len(results) > 0
            print(f"✓ Inference completed in {inference_time:.3f} seconds")

            # Check result structure
            result = results[0]
            assert hasattr(result, "boxes")
            print(f"✓ Result has boxes attribute")

            # Log detection count
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"✓ Detected {num_detections} objects")

        except Exception as e:
            raise Exception(f"YOLOv8 inference failed: {e}")

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="YOLOv8 dependencies not available")
    def test_yolo_performance_benchmark(self, test_image):
        """Benchmark YOLOv8 performance."""
        try:
            model = YOLO("yolov8n.pt")

            # Warm-up run
            model.predict(test_image, verbose=False)

            # Benchmark runs
            num_runs = 5
            inference_times = []

            for i in range(num_runs):
                start_time = time.time()
                results = model.predict(test_image, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

            # Calculate statistics
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)

            print(f"✓ Performance Benchmark ({num_runs} runs):")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")

            # Performance assertion (should be reasonable for CPU)
            assert avg_time < 10.0, f"Average inference time too slow: {avg_time:.3f}s"

        except Exception as e:
            pytest.fail(f"Performance benchmark failed: {e}")

    def run_yolo_performance_benchmark(self, test_image):
        """Benchmark YOLOv8 performance (non-pytest version)."""
        try:
            model = YOLO("yolov8n.pt")

            # Warm-up run
            model.predict(test_image, verbose=False)

            # Benchmark runs
            num_runs = 5
            inference_times = []

            for i in range(num_runs):
                start_time = time.time()
                results = model.predict(test_image, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

            # Calculate statistics
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)

            print(f"✓ Performance Benchmark ({num_runs} runs):")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")

            # Performance assertion (should be reasonable for CPU)
            assert avg_time < 10.0, f"Average inference time too slow: {avg_time:.3f}s"

        except Exception as e:
            raise Exception(f"Performance benchmark failed: {e}")

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="YOLOv8 dependencies not available")
    def test_yolo_different_image_sizes(self):
        """Test YOLOv8 with different image sizes."""
        try:
            model = YOLO("yolov8n.pt")

            # Test different image sizes
            sizes = [
                (320, 240),  # Small
                (640, 480),  # Medium
                (1280, 720),  # Large
            ]

            for width, height in sizes:
                # Create test image
                test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

                # Run inference
                start_time = time.time()
                results = model.predict(test_img, verbose=False)
                inference_time = time.time() - start_time

                assert results is not None
                print(f"✓ {width}x{height}: {inference_time:.3f}s")

        except Exception as e:
            pytest.fail(f"Multi-size test failed: {e}")

    def run_yolo_different_image_sizes(self):
        """Test YOLOv8 with different image sizes (non-pytest version)."""
        try:
            model = YOLO("yolov8n.pt")

            # Test different image sizes
            sizes = [
                (320, 240),  # Small
                (640, 480),  # Medium
                (1280, 720),  # Large
            ]

            for width, height in sizes:
                # Create test image
                test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

                # Run inference
                start_time = time.time()
                results = model.predict(test_img, verbose=False)
                inference_time = time.time() - start_time

                assert results is not None
                print(f"✓ {width}x{height}: {inference_time:.3f}s")

        except Exception as e:
            raise Exception(f"Multi-size test failed: {e}")

    @pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="YOLOv8 dependencies not available")
    def test_yolo_device_compatibility(self):
        """Test device compatibility."""
        try:
            model = YOLO("yolov8n.pt")

            # Test CPU
            model.to("cpu")
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = model.predict(test_img, verbose=False)
            assert results is not None
            print("✓ CPU inference working")

            # Test CUDA if available
            if torch.cuda.is_available():
                model.to("cuda")
                results = model.predict(test_img, verbose=False)
                assert results is not None
                print("✓ CUDA inference working")
            else:
                print("ℹ CUDA not available, skipping GPU test")

        except Exception as e:
            pytest.fail(f"Device compatibility test failed: {e}")

    def run_yolo_device_compatibility(self):
        """Test device compatibility (non-pytest version)."""
        try:
            model = YOLO("yolov8n.pt")

            # Test CPU
            model.to("cpu")
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = model.predict(test_img, verbose=False)
            assert results is not None
            print("✓ CPU inference working")

            # Test CUDA if available
            if torch.cuda.is_available():
                model.to("cuda")
                results = model.predict(test_img, verbose=False)
                assert results is not None
                print("✓ CUDA inference working")
            else:
                print("ℹ CUDA not available, skipping GPU test")

        except Exception as e:
            raise Exception(f"Device compatibility test failed: {e}")


def test_hardware_detection_basic():
    """Test basic hardware detection without complex imports."""
    try:
        # Test basic system info
        import platform

        import psutil

        print(f"✓ Platform: {platform.system()} {platform.release()}")
        print(f"✓ CPU: {platform.processor()}")
        print(f"✓ Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

        # Test GPU detection
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
            )
        else:
            print("ℹ No CUDA GPU detected")

        assert True  # Basic hardware detection successful

    except Exception as e:
        pytest.fail(f"Hardware detection failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    print("=" * 60)
    print("SpygateAI YOLOv8 Integration Test")
    print("=" * 60)

    # Test dependencies
    if DEPENDENCIES_AVAILABLE:
        print("✓ All dependencies available")

        # Run basic tests
        test_hardware_detection_basic()

        # Create test instance
        test_instance = TestYOLOv8BasicFunctionality()

        # Create test image
        test_img = create_test_image()

        # Run tests
        test_instance.test_yolo_model_loading()
        test_instance.run_yolo_inference_test(test_img)
        test_instance.run_yolo_performance_benchmark(test_img)
        test_instance.run_yolo_different_image_sizes()
        test_instance.run_yolo_device_compatibility()

        print("=" * 60)
        print("✓ All YOLOv8 integration tests passed!")
        print("=" * 60)
    else:
        print("✗ Dependencies not available, skipping tests")
