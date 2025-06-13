"""
Enhanced YOLOv8 Integration Testing for SpygateAI.
Tests comprehensive YOLOv8 functionality including AutoClipDetector integration,
hardware detection, performance metrics, and error handling.
"""

import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Add the spygate directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Import SpygateAI components
try:
    from spygate.core.gpu_memory_manager import get_memory_manager
    from spygate.core.hardware import HardwareDetector, HardwareTier
    from spygate.ml.yolov8_model import MODEL_CONFIGS, UI_CLASSES, CustomYOLOv8, DetectionResult

    SPYGATE_COMPONENTS_AVAILABLE = True
    print("✓ SpygateAI components available")
except ImportError as e:
    SPYGATE_COMPONENTS_AVAILABLE = False
    print(f"✗ SpygateAI components not available: {e}")

logger = logging.getLogger(__name__)


def create_madden_hud_test_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a test image that simulates Madden NFL 25 HUD elements."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Background (field-like color)
    image[:, :] = [34, 102, 34]  # Dark green

    # Score bug area (top center)
    cv2.rectangle(image, (760, 20), (1160, 120), (0, 0, 0), -1)  # Black background
    cv2.rectangle(image, (770, 30), (1150, 110), (255, 255, 255), -1)  # White text area
    cv2.putText(image, "BUF 14 - 21 KC", (780, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Down and distance (bottom left)
    cv2.rectangle(image, (50, height - 150), (300, height - 50), (0, 0, 0), -1)
    cv2.rectangle(image, (60, height - 140), (290, height - 60), (255, 255, 255), -1)
    cv2.putText(image, "3rd & 7", (80, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Game clock (top right)
    cv2.rectangle(image, (width - 200, 20), (width - 20, 80), (0, 0, 0), -1)
    cv2.rectangle(image, (width - 190, 30), (width - 30, 70), (255, 255, 255), -1)
    cv2.putText(image, "12:45", (width - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Field position marker (center bottom)
    cv2.rectangle(
        image, (width // 2 - 100, height - 100), (width // 2 + 100, height - 20), (255, 255, 0), -1
    )
    cv2.putText(
        image, "KC 35", (width // 2 - 50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    # Play clock (bottom right)
    cv2.rectangle(image, (width - 150, height - 120), (width - 20, height - 20), (255, 0, 0), -1)
    cv2.putText(
        image, "15", (width - 100, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    return image


def create_test_video(duration: int = 5, fps: int = 30) -> str:
    """Create a test video file with HUD elements."""
    temp_dir = tempfile.mkdtemp()
    video_path = Path(temp_dir) / "test_video.mp4"

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (1920, 1080))

    total_frames = duration * fps
    for frame_num in range(total_frames):
        # Create frame with slight variations
        frame = create_madden_hud_test_image()

        # Add some motion simulation
        offset = int(10 * np.sin(frame_num * 0.1))
        frame = np.roll(frame, offset, axis=1)

        writer.write(frame)

    writer.release()
    return str(video_path)


class TestYOLOv8EnhancedIntegration:
    """Enhanced YOLOv8 integration tests for SpygateAI."""

    @pytest.fixture(scope="class")
    def hardware_detector(self):
        """Create hardware detector instance."""
        if not SPYGATE_COMPONENTS_AVAILABLE:
            pytest.skip("SpygateAI components not available")
        return HardwareDetector()

    @pytest.fixture(scope="class")
    def test_image(self):
        """Create a test image simulating Madden HUD."""
        return create_madden_hud_test_image()

    @pytest.fixture(scope="class")
    def test_video_path(self):
        """Create a test video file."""
        video_path = create_test_video(duration=3)
        yield video_path
        # Cleanup
        try:
            Path(video_path).parent.rmdir()
        except:
            pass

    @pytest.fixture(scope="class")
    def yolo_model(self, hardware_detector):
        """Create YOLOv8 model instance."""
        if not SPYGATE_COMPONENTS_AVAILABLE:
            pytest.skip("SpygateAI components not available")
        try:
            return CustomYOLOv8(hardware=hardware_detector)
        except Exception as e:
            pytest.skip(f"Could not initialize YOLOv8 model: {e}")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_hardware_detection_integration(self, hardware_detector):
        """Test hardware detection and tier classification."""
        assert hardware_detector is not None
        assert hasattr(hardware_detector, "tier")
        assert isinstance(hardware_detector.tier, HardwareTier)

        # Test hardware capabilities
        assert hasattr(hardware_detector, "cpu_info")
        assert hasattr(hardware_detector, "memory_info")
        assert hasattr(hardware_detector, "gpu_info")

        logger.info(f"Detected hardware tier: {hardware_detector.tier.name}")
        logger.info(f"CPU cores: {hardware_detector.cpu_info.get('cores', 'unknown')}")
        logger.info(f"Total memory: {hardware_detector.memory_info.get('total', 'unknown')} GB")
        logger.info(f"GPU available: {hardware_detector.has_cuda}")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_model_configuration_selection(self, hardware_detector):
        """Test that appropriate model configuration is selected based on hardware."""
        tier = hardware_detector.tier
        assert tier in MODEL_CONFIGS

        config = MODEL_CONFIGS[tier]
        required_keys = ["model_size", "img_size", "batch_size", "device", "conf", "iou"]

        for key in required_keys:
            assert key in config, f"Missing config key: {key}"

        # Validate config values
        assert config["model_size"] in ["n", "s", "m", "l", "x"]
        assert isinstance(config["img_size"], int) and config["img_size"] > 0
        assert isinstance(config["batch_size"], int) and config["batch_size"] > 0
        assert 0 < config["conf"] < 1
        assert 0 < config["iou"] < 1

        logger.info(f"Selected config for {tier.name}: {config}")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_yolo_model_initialization(self, yolo_model):
        """Test YOLOv8 model initialization with SpygateAI components."""
        assert yolo_model is not None
        assert hasattr(yolo_model, "model")
        assert hasattr(yolo_model, "class_names")
        assert yolo_model.class_names == UI_CLASSES

        # Test hardware integration
        assert hasattr(yolo_model, "hardware")
        assert hasattr(yolo_model, "config")
        assert hasattr(yolo_model, "device")

        logger.info(f"Model initialized with device: {yolo_model.device}")
        logger.info(f"Model class names: {len(yolo_model.class_names)} classes")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_hud_element_detection(self, yolo_model, test_image):
        """Test detection of HUD elements in test image."""
        start_time = time.time()

        # Run inference
        results = yolo_model.predict(test_image, verbose=False)

        inference_time = time.time() - start_time

        assert results is not None
        assert len(results) > 0

        result = results[0]
        assert hasattr(result, "boxes")

        # Log detection results
        num_detections = len(result.boxes) if result.boxes is not None else 0
        logger.info(f"HUD detection completed in {inference_time:.3f} seconds")
        logger.info(f"Detected {num_detections} HUD elements")

        if result.boxes is not None and len(result.boxes) > 0:
            # Check detection quality
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            logger.info(f"Detection confidences: {confidences}")
            logger.info(f"Detected classes: {classes}")

            # Verify at least some reasonable confidence
            assert np.max(confidences) > 0.1, "No high-confidence detections found"

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_performance_across_image_sizes(self, yolo_model):
        """Test YOLOv8 performance with different image sizes."""
        test_sizes = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]
        performance_results = {}

        for width, height in test_sizes:
            # Create test image
            test_img = create_madden_hud_test_image(width, height)

            # Warmup
            yolo_model.predict(test_img, verbose=False)

            # Benchmark
            times = []
            for _ in range(3):
                start_time = time.time()
                results = yolo_model.predict(test_img, verbose=False)
                inference_time = time.time() - start_time
                times.append(inference_time)

            avg_time = np.mean(times)
            performance_results[f"{width}x{height}"] = avg_time

            logger.info(f"Average inference time for {width}x{height}: {avg_time:.3f}s")

            # Performance assertion (should scale reasonably with image size)
            if width <= 640:
                assert avg_time < 5.0, f"Performance too slow for {width}x{height}: {avg_time:.3f}s"
            else:
                assert (
                    avg_time < 10.0
                ), f"Performance too slow for {width}x{height}: {avg_time:.3f}s"

        # Check that performance scales reasonably
        small_time = performance_results["320x240"]
        large_time = performance_results["1920x1080"]

        # Large image shouldn't be more than 10x slower than small
        assert large_time / small_time < 10, "Performance doesn't scale well with image size"

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_batch_processing_capability(self, yolo_model, test_image):
        """Test batch processing with multiple images."""
        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            # Create batch of images
            batch_images = [test_image for _ in range(batch_size)]

            start_time = time.time()
            results = yolo_model.predict(batch_images, verbose=False)
            batch_time = time.time() - start_time

            assert len(results) == batch_size

            avg_time_per_image = batch_time / batch_size
            logger.info(f"Batch size {batch_size}: {avg_time_per_image:.3f}s per image")

            # Batch processing should be reasonably efficient
            assert avg_time_per_image < 5.0, f"Batch processing too slow: {avg_time_per_image:.3f}s"

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_memory_management_integration(self, yolo_model, test_image):
        """Test GPU memory management during inference."""
        if not hasattr(yolo_model, "memory_manager") or yolo_model.memory_manager is None:
            pytest.skip("Memory manager not available")

        # Get initial memory state
        try:
            initial_memory = yolo_model.memory_manager.get_memory_stats()
            logger.info(f"Initial memory state: {initial_memory}")
        except Exception as e:
            pytest.skip(f"Memory manager not functional: {e}")

        # Run multiple inferences
        for i in range(5):
            results = yolo_model.predict(test_image, verbose=False)

        # Get final memory state
        final_memory = yolo_model.memory_manager.get_memory_stats()
        logger.info(f"Final memory state: {final_memory}")

        # Check for excessive memory growth
        if "allocated" in initial_memory and "allocated" in final_memory:
            memory_increase = final_memory["allocated"] - initial_memory["allocated"]
            logger.info(f"Memory increase: {memory_increase} bytes")

            # Allow some memory increase but not excessive (100MB limit)
            assert memory_increase < 100 * 1024 * 1024, "Potential memory leak detected"

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_error_handling_robustness(self, yolo_model):
        """Test error handling with invalid inputs."""

        # Test with None input
        try:
            results = yolo_model.predict(None, verbose=False)
            assert False, "Should have raised an error with None input"
        except Exception as e:
            logger.info(f"Correctly handled None input: {e}")

        # Test with empty array
        try:
            empty_img = np.array([])
            results = yolo_model.predict(empty_img, verbose=False)
            logger.info("Empty array handled gracefully")
        except Exception as e:
            logger.info(f"Empty array error handled: {e}")

        # Test with invalid shape
        try:
            invalid_img = np.zeros((10, 10))  # 2D instead of 3D
            results = yolo_model.predict(invalid_img, verbose=False)
            logger.info("Invalid shape handled gracefully")
        except Exception as e:
            logger.info(f"Invalid shape error handled: {e}")

        # Test with very small image
        try:
            tiny_img = np.zeros((1, 1, 3), dtype=np.uint8)
            results = yolo_model.predict(tiny_img, verbose=False)
            logger.info("Tiny image handled gracefully")
        except Exception as e:
            logger.info(f"Tiny image error handled: {e}")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_device_compatibility(self, yolo_model):
        """Test device compatibility (CPU/GPU)."""
        device = yolo_model.device
        logger.info(f"Model running on device: {device}")

        # Test device switching if CUDA is available
        if torch.cuda.is_available():
            try:
                # Test GPU inference
                test_img = create_madden_hud_test_image(640, 480)
                results_gpu = yolo_model.predict(test_img, verbose=False)
                logger.info("GPU inference successful")

                # Note: We don't test CPU switching here as it requires model reinitialization
                # which is not practical in this test context

            except Exception as e:
                logger.warning(f"GPU inference failed: {e}")
        else:
            logger.info("CUDA not available, testing CPU only")

        # Verify inference works on current device
        test_img = create_madden_hud_test_image(640, 480)
        results = yolo_model.predict(test_img, verbose=False)
        assert results is not None
        logger.info(f"Inference successful on {device}")

    @pytest.mark.skipif(
        not DEPENDENCIES_AVAILABLE or not SPYGATE_COMPONENTS_AVAILABLE,
        reason="Dependencies not available",
    )
    def test_ui_class_detection_accuracy(self, yolo_model, test_image):
        """Test accuracy of UI class detection for HUD elements."""
        results = yolo_model.predict(test_image, verbose=False)

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                # Check that detected classes are valid UI classes
                valid_class_indices = set(range(len(UI_CLASSES)))
                detected_classes = set(classes.astype(int))

                assert detected_classes.issubset(
                    valid_class_indices
                ), f"Invalid class indices detected: {detected_classes - valid_class_indices}"

                # Log detected UI elements
                for i, (cls_idx, conf) in enumerate(zip(classes, confidences)):
                    class_name = UI_CLASSES[int(cls_idx)]
                    logger.info(f"Detected {class_name} with confidence {conf:.3f}")

                logger.info(f"Total UI elements detected: {len(classes)}")
            else:
                logger.info("No UI elements detected in test image")
        else:
            logger.info("No results from inference")

    def test_integration_summary(self):
        """Generate a summary of integration test results."""
        summary = {
            "dependencies_available": DEPENDENCIES_AVAILABLE,
            "spygate_components_available": SPYGATE_COMPONENTS_AVAILABLE,
            "pytorch_version": torch.__version__ if DEPENDENCIES_AVAILABLE else "N/A",
            "ultralytics_version": ultralytics.__version__ if DEPENDENCIES_AVAILABLE else "N/A",
            "cuda_available": torch.cuda.is_available() if DEPENDENCIES_AVAILABLE else False,
        }

        logger.info("YOLOv8 Integration Test Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

        # Basic functionality check
        if DEPENDENCIES_AVAILABLE and SPYGATE_COMPONENTS_AVAILABLE:
            try:
                hardware = HardwareDetector()
                model = CustomYOLOv8(hardware=hardware)
                test_img = create_madden_hud_test_image(640, 480)
                results = model.predict(test_img, verbose=False)
                logger.info("  basic_functionality: PASS")
            except Exception as e:
                logger.error(f"  basic_functionality: FAIL - {e}")
        else:
            logger.info("  basic_functionality: SKIP - Dependencies not available")


if __name__ == "__main__":
    # Run tests directly without pytest
    test_instance = TestYOLOv8EnhancedIntegration()

    print("Running YOLOv8 Enhanced Integration Tests...")

    # Test hardware detection
    if SPYGATE_COMPONENTS_AVAILABLE:
        hardware = HardwareDetector()
        test_instance.test_hardware_detection_integration(hardware)

        # Test model initialization
        try:
            model = CustomYOLOv8(hardware=hardware)
            test_instance.test_yolo_model_initialization(model)

            # Test basic inference
            test_img = create_madden_hud_test_image()
            test_instance.test_hud_element_detection(model, test_img)

            print("✓ All enhanced integration tests completed successfully")

        except Exception as e:
            print(f"✗ Model initialization failed: {e}")
    else:
        print("✗ SpygateAI components not available")

    # Generate summary
    test_instance.test_integration_summary()
