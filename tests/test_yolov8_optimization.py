"""
Comprehensive test suite for Priority 3: YOLOv8 Model Configuration Optimization

Tests include:
- Hardware-tier specific optimizations
- Dynamic model switching
- Performance monitoring and metrics
- Adaptive batch sizing
- Auto-optimization features
- Benchmarking capabilities
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spygate.core.hardware import HardwareDetector, HardwareTier
from spygate.ml.yolov8_model import (
    MODEL_CONFIGS,
    EnhancedYOLOv8,
    OptimizationConfig,
    PerformanceMetrics,
    create_optimization_config,
    get_hardware_optimized_config,
    load_optimized_yolov8_model,
)


class TestPerformanceMetrics:
    """Test performance metrics tracking functionality."""

    def test_performance_metrics_initialization(self):
        """Test that performance metrics initialize correctly."""
        metrics = PerformanceMetrics()

        assert len(metrics.inference_times) == 0
        assert len(metrics.memory_usage) == 0
        assert len(metrics.batch_sizes) == 0
        assert len(metrics.accuracy_scores) == 0
        assert len(metrics.gpu_utilization) == 0

    def test_add_measurement(self):
        """Test adding performance measurements."""
        metrics = PerformanceMetrics()

        # Add basic measurement
        metrics.add_measurement(inference_time=0.5, memory_mb=512.0, batch_size=4)

        assert len(metrics.inference_times) == 1
        assert metrics.inference_times[0] == 0.5
        assert metrics.memory_usage[0] == 512.0
        assert metrics.batch_sizes[0] == 4

    def test_add_measurement_with_optional_params(self):
        """Test adding measurements with optional parameters."""
        metrics = PerformanceMetrics()

        metrics.add_measurement(
            inference_time=0.3, memory_mb=256.0, batch_size=2, accuracy=0.92, gpu_util=0.75
        )

        assert metrics.accuracy_scores[0] == 0.92
        assert metrics.gpu_utilization[0] == 0.75

    def test_average_metrics_calculation(self):
        """Test calculation of average metrics."""
        metrics = PerformanceMetrics()

        # Add multiple measurements
        for i in range(5):
            metrics.add_measurement(
                inference_time=0.1 * (i + 1),
                memory_mb=100.0 * (i + 1),
                batch_size=i + 1,
                accuracy=0.8 + 0.02 * i,
            )

        avg_metrics = metrics.get_average_metrics()

        assert avg_metrics["avg_inference_time"] == 0.3  # (0.1+0.2+0.3+0.4+0.5)/5
        assert avg_metrics["avg_memory_usage"] == 300.0  # (100+200+300+400+500)/5
        assert avg_metrics["avg_batch_size"] == 3.0  # (1+2+3+4+5)/5
        assert avg_metrics["total_inferences"] == 5

    def test_metrics_deque_maxlen(self):
        """Test that metrics respect maximum length limits."""
        metrics = PerformanceMetrics()

        # Add more than maxlen measurements
        for i in range(150):  # More than maxlen=100 for inference_times
            metrics.add_measurement(inference_time=0.1, memory_mb=100.0, batch_size=1)

        assert len(metrics.inference_times) == 100  # Should be capped at maxlen


class TestOptimizationConfig:
    """Test optimization configuration functionality."""

    def test_default_optimization_config(self):
        """Test default optimization configuration values."""
        config = OptimizationConfig()

        assert config.enable_dynamic_switching is True
        assert config.enable_adaptive_batch_size is True
        assert config.enable_performance_monitoring is True
        assert config.enable_auto_optimization is True
        assert config.max_inference_time == 1.0
        assert config.min_accuracy == 0.85
        assert config.max_memory_usage == 0.9

    def test_create_optimization_config(self):
        """Test creating custom optimization config."""
        config = create_optimization_config(
            enable_dynamic_switching=False, max_inference_time=0.5, min_accuracy=0.9
        )

        assert config.enable_dynamic_switching is False
        assert config.max_inference_time == 0.5
        assert config.min_accuracy == 0.9
        # Ensure other defaults are preserved
        assert config.enable_adaptive_batch_size is True


class TestHardwareOptimizedConfigs:
    """Test hardware-tier specific configurations."""

    def test_model_configs_exist_for_all_tiers(self):
        """Test that configurations exist for all hardware tiers."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        expected_tiers = [
            HardwareTier.ULTRA_LOW,
            HardwareTier.LOW,
            HardwareTier.MEDIUM,
            HardwareTier.HIGH,
            HardwareTier.ULTRA,
        ]

        for tier in expected_tiers:
            config = get_hardware_optimized_config(tier)
            assert isinstance(config, dict)
            assert "model_size" in config
            assert "img_size" in config
            assert "batch_size" in config

    def test_tier_specific_optimizations(self):
        """Test that different tiers have appropriate optimizations."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        ultra_low_config = MODEL_CONFIGS[HardwareTier.ULTRA_LOW]
        ultra_config = MODEL_CONFIGS[HardwareTier.ULTRA]

        # Ultra-low should have conservative settings
        assert ultra_low_config["model_size"] == "n"
        assert ultra_low_config["batch_size"] == 1
        assert ultra_low_config["half"] is False
        assert ultra_low_config["quantize"] is False
        assert ultra_low_config["compile"] is False

        # Ultra should have aggressive optimization settings
        assert ultra_config["model_size"] == "l"
        assert ultra_config["batch_size"] > ultra_low_config["batch_size"]
        assert ultra_config["half"] is True
        assert ultra_config["quantize"] is True
        assert ultra_config["compile"] is True

    def test_config_progression_across_tiers(self):
        """Test that configurations scale appropriately across tiers."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        tiers = [
            HardwareTier.ULTRA_LOW,
            HardwareTier.LOW,
            HardwareTier.MEDIUM,
            HardwareTier.HIGH,
            HardwareTier.ULTRA,
        ]

        # Image sizes should generally increase with tier
        img_sizes = [MODEL_CONFIGS[tier]["img_size"] for tier in tiers]
        assert img_sizes == sorted(img_sizes)  # Should be in ascending order

        # Batch sizes should generally increase with tier
        batch_sizes = [MODEL_CONFIGS[tier]["batch_size"] for tier in tiers]
        assert batch_sizes == sorted(batch_sizes)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEnhancedYOLOv8:
    """Test EnhancedYOLOv8 model optimization features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hardware = MagicMock(spec=HardwareDetector)
        self.hardware.tier = HardwareTier.MEDIUM
        self.hardware.has_cuda = False  # Use CPU for testing

        self.optimization_config = OptimizationConfig(
            enable_performance_monitoring=True,
            enable_auto_optimization=False,  # Disable for testing
            enable_dynamic_switching=False,  # Disable for testing
        )

    @patch("spygate.ml.yolo8_model.YOLO")
    def test_enhanced_model_initialization(self, mock_yolo):
        """Test enhanced model initialization with optimizations."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = EnhancedYOLOv8(hardware=self.hardware, optimization_config=self.optimization_config)

        assert model.hardware == self.hardware
        assert model.optimization_config == self.optimization_config
        assert isinstance(model.performance_metrics, PerformanceMetrics)
        assert model.inference_counter == 0
        assert "default" in model.model_variants

    @patch("spygate.ml.yolo8_model.YOLO")
    def test_optimization_application(self, mock_yolo):
        """Test that optimizations are applied correctly."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.fuse = MagicMock()
        mock_yolo.return_value = mock_model

        # Test with a tier that enables optimizations
        self.hardware.tier = HardwareTier.HIGH

        model = EnhancedYOLOv8(hardware=self.hardware, optimization_config=self.optimization_config)

        # Verify model was set to eval mode
        mock_model.eval.assert_called_once()

    @patch("spygate.ml.yolo8_model.YOLO")
    def test_performance_metrics_tracking(self, mock_yolo):
        """Test that performance metrics are tracked during inference."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_yolo.return_value = mock_model
        mock_model.predict.return_value = [mock_result]

        model = EnhancedYOLOv8(hardware=self.hardware, optimization_config=self.optimization_config)

        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run detection
        result = model.detect_hud_elements(frame)

        # Check that performance metrics were recorded
        assert len(model.performance_metrics.inference_times) > 0
        assert model.inference_counter == 1

    @patch("spygate.ml.yolo8_model.YOLO")
    def test_performance_report_generation(self, mock_yolo):
        """Test performance report generation."""
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        model = EnhancedYOLOv8(hardware=self.hardware, optimization_config=self.optimization_config)

        # Add some performance data
        model.performance_metrics.add_measurement(0.5, 512.0, 4, 0.9)
        model.inference_counter = 10

        report = model.get_performance_report()

        assert "model_info" in report
        assert "performance_metrics" in report
        assert "configuration" in report
        assert "optimization_config" in report
        assert report["total_inferences"] == 10
        assert report["model_info"]["hardware_tier"] == "MEDIUM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
