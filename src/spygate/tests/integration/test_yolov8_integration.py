"""
Integration tests for YOLOv8 functionality in SpygateAI.
Tests the complete YOLOv8 pipeline including hardware detection, model loading, and inference.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pytest

# Import YOLOv8 components with proper error handling
try:
    from spygate.core.hardware import HardwareDetector, HardwareTier
    from spygate.ml.yolov8_model import MODEL_CONFIGS, UI_CLASSES, CustomYOLOv8, DetectionResult

    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    pytest.skip(f"YOLOv8 components not available: {e}", allow_module_level=True)

logger = logging.getLogger(__name__)


class TestYOLOv8Integration:
    """Integration tests for YOLOv8 functionality."""

    @pytest.fixture(scope="class")
    def hardware_detector(self):
        """Create hardware detector instance."""
        return HardwareDetector()

    @pytest.fixture(scope="class")
    def test_image(self):
        """Create a test image for detection."""
        # Create a simple test image (640x480, 3 channels)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some basic shapes to simulate HUD elements
        cv2.rectangle(image, (50, 50), (200, 100), (255, 255, 255), -1)  # Score bug area
        cv2.rectangle(image, (300, 400), (450, 450), (255, 255, 255), -1)  # Down/distance area
        cv2.circle(image, (500, 100), 30, (255, 255, 255), -1)  # Clock area

        return image

    @pytest.fixture(scope="class")
    def yolo_model(self, hardware_detector):
        """Create YOLOv8 model instance."""
        try:
            return CustomYOLOv8(hardware=hardware_detector)
        except Exception as e:
            pytest.skip(f"Could not initialize YOLOv8 model: {e}")

    def test_hardware_detection(self, hardware_detector):
        """Test that hardware detection works correctly."""
        assert hardware_detector is not None
        assert hasattr(hardware_detector, "tier")
        assert isinstance(hardware_detector.tier, HardwareTier)

        logger.info(f"Detected hardware tier: {hardware_detector.tier.name}")

        # Verify hardware capabilities
        assert hasattr(hardware_detector, "cpu_info")
        assert hasattr(hardware_detector, "memory_info")
        assert hasattr(hardware_detector, "gpu_info")

    def test_model_configuration_selection(self, hardware_detector):
        """Test that appropriate model configuration is selected based on hardware."""
        tier = hardware_detector.tier
        assert tier in MODEL_CONFIGS

        config = MODEL_CONFIGS[tier]
        assert "model_size" in config
        assert "img_size" in config
        assert "batch_size" in config
        assert "device" in config

        logger.info(f"Selected config for {tier.name}: {config}")

    def test_model_initialization(self, yolo_model):
        """Test that YOLOv8 model initializes correctly."""
        assert yolo_model is not None
        assert hasattr(yolo_model, "model")
        assert hasattr(yolo_model, "class_names")
        assert yolo_model.class_names == UI_CLASSES

        logger.info(f"Model initialized with device: {yolo_model.device}")

    def test_model_inference(self, yolo_model, test_image):
        """Test basic model inference functionality."""
        start_time = time.time()

        # Run inference
        results = yolo_model.predict(test_image, verbose=False)

        inference_time = time.time() - start_time

        assert results is not None
        assert len(results) > 0

        # Check result structure
        result = results[0]
        assert hasattr(result, "boxes")

        logger.info(f"Inference completed in {inference_time:.3f} seconds")
        logger.info(f"Detected {len(result.boxes) if result.boxes is not None else 0} objects")

    def test_detection_result_structure(self, yolo_model, test_image):
        """Test that detection results have the expected structure."""
        results = yolo_model.predict(test_image, verbose=False)

        if results and len(results) > 0:
            result = results[0]

            # Check if boxes exist and have correct structure
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Verify box data structure
                assert hasattr(boxes, "xyxy")  # Bounding box coordinates
                assert hasattr(boxes, "conf")  # Confidence scores
                assert hasattr(boxes, "cls")  # Class indices

                # Verify data types
                assert isinstance(boxes.xyxy.cpu().numpy(), np.ndarray)
                assert isinstance(boxes.conf.cpu().numpy(), np.ndarray)
                assert isinstance(boxes.cls.cpu().numpy(), np.ndarray)

    def test_performance_benchmarking(self, yolo_model, test_image):
        """Benchmark YOLOv8 performance across multiple runs."""
        num_runs = 10
        inference_times = []

        # Warm-up run
        yolo_model.predict(test_image, verbose=False)

        # Benchmark runs
        for i in range(num_runs):
            start_time = time.time()
            results = yolo_model.predict(test_image, verbose=False)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_time = np.std(inference_times)

        logger.info(f"Performance Benchmark Results ({num_runs} runs):")
        logger.info(f"  Average: {avg_time:.3f}s")
        logger.info(f"  Min: {min_time:.3f}s")
        logger.info(f"  Max: {max_time:.3f}s")
        logger.info(f"  Std Dev: {std_time:.3f}s")

        # Performance assertions
        assert avg_time < 5.0, f"Average inference time too slow: {avg_time:.3f}s"
        assert std_time < 1.0, f"Inference time too variable: {std_time:.3f}s"

    def test_memory_management(self, yolo_model, test_image):
        """Test GPU memory management during inference."""
        if not hasattr(yolo_model, "memory_manager") or yolo_model.memory_manager is None:
            pytest.skip("Memory manager not available")

        # Get initial memory state
        initial_memory = yolo_model.memory_manager.get_memory_stats()

        # Run multiple inferences
        for i in range(5):
            results = yolo_model.predict(test_image, verbose=False)

        # Get final memory state
        final_memory = yolo_model.memory_manager.get_memory_stats()

        # Check for memory leaks
        memory_increase = final_memory.get("allocated", 0) - initial_memory.get("allocated", 0)

        logger.info(f"Memory usage - Initial: {initial_memory}")
        logger.info(f"Memory usage - Final: {final_memory}")
        logger.info(f"Memory increase: {memory_increase} bytes")

        # Allow some memory increase but not excessive
        assert memory_increase < 100 * 1024 * 1024, "Potential memory leak detected"

    def test_batch_processing(self, yolo_model, test_image):
        """Test batch processing capabilities."""
        batch_size = yolo_model.optimal_batch_size

        # Create batch of images
        batch_images = [test_image for _ in range(min(batch_size, 4))]

        start_time = time.time()

        # Process batch
        results = yolo_model.predict(batch_images, verbose=False)

        batch_time = time.time() - start_time

        assert len(results) == len(batch_images)

        logger.info(f"Batch processing ({len(batch_images)} images) completed in {batch_time:.3f}s")
        logger.info(f"Average per image: {batch_time/len(batch_images):.3f}s")

    def test_ui_class_detection(self, yolo_model):
        """Test detection of specific UI classes."""
        # This would require actual game footage with HUD elements
        # For now, we'll test that the class names are properly configured

        assert yolo_model.class_names == UI_CLASSES
        assert "score_bug" in yolo_model.class_names
        assert "down_distance" in yolo_model.class_names
        assert "game_clock" in yolo_model.class_names

        logger.info(f"Configured UI classes: {yolo_model.class_names}")

    def test_error_handling(self, yolo_model):
        """Test error handling with invalid inputs."""
        # Test with None input
        with pytest.raises((ValueError, TypeError, AttributeError)):
            yolo_model.predict(None)

        # Test with invalid image shape
        invalid_image = np.zeros((10, 10), dtype=np.uint8)  # 2D instead of 3D
        try:
            results = yolo_model.predict(invalid_image, verbose=False)
            # Some implementations might handle this gracefully
            logger.info("Model handled 2D image gracefully")
        except Exception as e:
            logger.info(f"Model correctly rejected invalid input: {e}")

    def test_device_compatibility(self, yolo_model):
        """Test device compatibility (CPU/GPU)."""
        device = yolo_model.device
        assert device in ["cpu", "cuda", "mps"]

        logger.info(f"Model running on device: {device}")

        # If CUDA is available, test GPU functionality
        if device == "cuda":
            import torch

            assert torch.cuda.is_available()
            assert torch.cuda.device_count() > 0

            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")


class TestYOLOv8PerformanceMetrics:
    """Performance-focused tests for YOLOv8."""

    @pytest.fixture(scope="class")
    def performance_test_images(self):
        """Create various test images for performance testing."""
        images = {}

        # Small image
        images["small"] = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        # Medium image
        images["medium"] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Large image
        images["large"] = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # HD image
        images["hd"] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        return images

    @pytest.mark.parametrize("image_size", ["small", "medium", "large"])
    def test_performance_by_image_size(self, image_size, performance_test_images):
        """Test performance across different image sizes."""
        hardware = HardwareDetector()
        model = CustomYOLOv8(hardware=hardware)

        image = performance_test_images[image_size]

        # Warm-up
        model.predict(image, verbose=False)

        # Benchmark
        start_time = time.time()
        results = model.predict(image, verbose=False)
        inference_time = time.time() - start_time

        logger.info(f"Image size {image_size} ({image.shape}): {inference_time:.3f}s")

        # Performance expectations based on image size
        if image_size == "small":
            assert inference_time < 1.0
        elif image_size == "medium":
            assert inference_time < 2.0
        elif image_size == "large":
            assert inference_time < 3.0

    def test_hardware_tier_performance(self):
        """Test that performance scales appropriately with hardware tier."""
        hardware = HardwareDetector()
        tier = hardware.tier

        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        model = CustomYOLOv8(hardware=hardware)

        # Benchmark
        start_time = time.time()
        results = model.predict(test_image, verbose=False)
        inference_time = time.time() - start_time

        logger.info(f"Hardware tier {tier.name}: {inference_time:.3f}s")

        # Performance expectations based on hardware tier
        if tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            assert inference_time < 5.0  # More lenient for low-end hardware
        elif tier == HardwareTier.MEDIUM:
            assert inference_time < 2.0
        elif tier in [HardwareTier.HIGH, HardwareTier.ULTRA]:
            assert inference_time < 1.0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
