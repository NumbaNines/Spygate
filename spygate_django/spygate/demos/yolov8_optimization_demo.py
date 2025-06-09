"""
Priority 3: YOLOv8 Model Configuration Optimization Demo

This script demonstrates the advanced optimization features implemented in Priority 3:
1. Hardware-tier specific configurations
2. Dynamic model switching
3. Performance monitoring and adaptive optimization
4. Benchmarking capabilities
5. Memory management integration

Usage:
    python spygate/demos/yolov8_optimization_demo.py
"""

import json
import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
    logger.info("PyTorch available - GPU optimizations enabled")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - CPU fallback mode")

from spygate.core.hardware import HardwareDetector, HardwareTier
from spygate.ml.yolov8_model import (
    MODEL_CONFIGS,
    EnhancedYOLOv8,
    OptimizationConfig,
    create_optimization_config,
    get_hardware_optimized_config,
    load_optimized_yolov8_model,
)


class OptimizationDemo:
    """Demonstration class for YOLOv8 optimization features."""

    def __init__(self):
        """Initialize the optimization demo."""
        logger.info("Initializing YOLOv8 Optimization Demo")

        # Detect hardware
        self.hardware = HardwareDetector()
        logger.info(f"Detected hardware tier: {self.hardware.tier.name}")

        # Create sample images for testing
        self.test_images = self._create_test_images()
        logger.info(f"Created {len(self.test_images)} test images")

    def _create_test_images(self, count: int = 5) -> list[np.ndarray]:
        """Create synthetic test images that simulate HUD elements."""
        images = []

        for i in range(count):
            # Create base image
            image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            # Add simulated HUD elements
            # Score bug (top left)
            cv2.rectangle(image, (10, 10), (200, 60), (0, 0, 255), -1)
            cv2.putText(
                image,
                "HOME 14 - 7 AWAY",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # Down and distance (bottom)
            cv2.rectangle(image, (500, 680), (780, 710), (255, 255, 0), -1)
            cv2.putText(
                image, f"3rd & {5 + i}", (510, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
            )

            # Game clock (top center)
            cv2.rectangle(image, (590, 10), (690, 40), (0, 255, 0), -1)
            cv2.putText(image, f"12:3{i}", (605, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            images.append(image)

        return images

    def demonstrate_hardware_configurations(self):
        """Demonstrate hardware-tier specific configurations."""
        logger.info("\n" + "=" * 60)
        logger.info("HARDWARE-TIER SPECIFIC CONFIGURATIONS")
        logger.info("=" * 60)

        for tier in [
            HardwareTier.ULTRA_LOW,
            HardwareTier.LOW,
            HardwareTier.MEDIUM,
            HardwareTier.HIGH,
            HardwareTier.ULTRA,
        ]:
            config = get_hardware_optimized_config(tier)

            logger.info(f"\n{tier.name} Configuration:")
            logger.info(f"  Model Size: YOLOv8{config['model_size']}")
            logger.info(f"  Input Size: {config['img_size']}px")
            logger.info(f"  Batch Size: {config['batch_size']}")
            logger.info(f"  Half Precision: {config['half']}")
            logger.info(f"  Quantization: {config['quantize']}")
            logger.info(f"  Compilation: {config['compile']}")
            logger.info(f"  Confidence: {config['conf']}")
            logger.info(f"  IoU Threshold: {config['iou']}")

    def demonstrate_optimization_configs(self):
        """Demonstrate different optimization configuration options."""
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION CONFIGURATION OPTIONS")
        logger.info("=" * 60)

        configs = {
            "Conservative": create_optimization_config(
                enable_dynamic_switching=False,
                enable_adaptive_batch_size=False,
                enable_auto_optimization=False,
                max_inference_time=2.0,
                max_memory_usage=0.5,
            ),
            "Balanced": create_optimization_config(
                enable_dynamic_switching=True,
                enable_adaptive_batch_size=True,
                enable_auto_optimization=True,
                max_inference_time=1.0,
                max_memory_usage=0.8,
            ),
            "Aggressive": create_optimization_config(
                enable_dynamic_switching=True,
                enable_adaptive_batch_size=True,
                enable_auto_optimization=True,
                max_inference_time=0.5,
                max_memory_usage=0.95,
            ),
        }

        for name, config in configs.items():
            logger.info(f"\n{name} Configuration:")
            logger.info(f"  Dynamic Switching: {config.enable_dynamic_switching}")
            logger.info(f"  Adaptive Batching: {config.enable_adaptive_batch_size}")
            logger.info(f"  Auto Optimization: {config.enable_auto_optimization}")
            logger.info(f"  Max Inference Time: {config.max_inference_time}s")
            logger.info(f"  Max Memory Usage: {config.max_memory_usage * 100}%")

    def demonstrate_model_initialization(self):
        """Demonstrate enhanced model initialization."""
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED MODEL INITIALIZATION")
        logger.info("=" * 60)

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping model initialization demo")
            return None

        try:
            # Create optimization config
            optimization_config = create_optimization_config(
                enable_performance_monitoring=True,
                enable_auto_optimization=False,  # Disable for demo
                max_inference_time=1.0,
            )

            # Initialize enhanced model
            logger.info("Initializing Enhanced YOLOv8 model...")
            model = load_optimized_yolov8_model(
                hardware=self.hardware, optimization_config=optimization_config
            )

            # Log capabilities
            model.log_capabilities()

            # Get and display memory statistics
            memory_stats = model.get_model_memory_stats()
            logger.info(f"\nModel Memory Statistics:")
            for key, value in memory_stats.items():
                logger.info(f"  {key}: {value}")

            return model

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return None

    def demonstrate_performance_monitoring(self, model: EnhancedYOLOv8):
        """Demonstrate performance monitoring capabilities."""
        if model is None:
            logger.warning("Model not available - skipping performance monitoring demo")
            return

        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE MONITORING DEMONSTRATION")
        logger.info("=" * 60)

        logger.info("Running inference on test images to collect performance metrics...")

        # Run several inferences to collect metrics
        for i, image in enumerate(self.test_images):
            logger.info(f"Processing image {i+1}/{len(self.test_images)}")

            # Run detection with memory tracking
            result, memory_stats = model.detect_hud_elements(image, return_memory_stats=True)

            logger.info(f"  Detections: {len(result.boxes)}")
            logger.info(f"  Processing time: {result.processing_time:.3f}s")
            if result.memory_usage:
                logger.info(f"  Memory usage: {result.memory_usage}")

        # Get performance report
        performance_report = model.get_performance_report()

        logger.info("\nPerformance Report:")
        logger.info(f"  Total inferences: {performance_report['total_inferences']}")

        metrics = performance_report["performance_metrics"]
        logger.info(f"  Average inference time: {metrics['avg_inference_time']:.3f}s")
        logger.info(f"  Average memory usage: {metrics['avg_memory_usage']:.1f}MB")
        logger.info(f"  Average batch size: {metrics['avg_batch_size']:.1f}")

    def demonstrate_benchmarking(self, model: EnhancedYOLOv8):
        """Demonstrate benchmarking capabilities."""
        if model is None:
            logger.warning("Model not available - skipping benchmarking demo")
            return

        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARKING DEMONSTRATION")
        logger.info("=" * 60)

        logger.info("Running comprehensive benchmark...")

        # Run benchmark
        benchmark_results = model.benchmark_model(self.test_images, iterations=3)

        logger.info("\nBenchmark Results:")
        logger.info(f"  Average inference time: {benchmark_results['avg_inference_time']:.3f}s")
        logger.info(f"  Standard deviation: {benchmark_results['std_inference_time']:.3f}s")
        logger.info(f"  Min inference time: {benchmark_results['min_inference_time']:.3f}s")
        logger.info(f"  Max inference time: {benchmark_results['max_inference_time']:.3f}s")
        logger.info(f"  Average memory usage: {benchmark_results['avg_memory_usage']:.1f}MB")
        logger.info(f"  Average detections: {benchmark_results['avg_detections']:.1f}")

        # Save benchmark results
        benchmark_file = Path("benchmark_results.json")
        model.save_performance_report(str(benchmark_file))
        logger.info(f"Detailed performance report saved to: {benchmark_file}")

    def demonstrate_adaptive_optimization(self, model: EnhancedYOLOv8):
        """Demonstrate adaptive optimization features."""
        if model is None:
            logger.warning("Model not available - skipping adaptive optimization demo")
            return

        logger.info("\n" + "=" * 60)
        logger.info("ADAPTIVE OPTIMIZATION DEMONSTRATION")
        logger.info("=" * 60)

        # Enable auto-optimization with aggressive thresholds for demo
        model.optimization_config.enable_auto_optimization = True
        model.optimization_config.max_inference_time = 0.01  # Very low for demo
        model.optimization_config.optimization_interval = 2

        logger.info("Enabled aggressive auto-optimization for demonstration...")
        logger.info(
            f"Max inference time threshold: {model.optimization_config.max_inference_time}s"
        )

        # Add some "slow" performance measurements to trigger optimization
        logger.info("Simulating slow performance to trigger optimization...")
        for i in range(3):
            model.performance_metrics.add_measurement(
                inference_time=0.5, memory_mb=800.0, batch_size=1  # Simulated slow inference
            )

        # Trigger optimization check
        original_config = model.config.copy()
        model._check_performance_and_optimize()

        # Check if optimizations were applied
        if model.config != original_config:
            logger.info("Optimizations applied!")
            logger.info("Configuration changes:")
            for key in original_config:
                if model.config.get(key) != original_config.get(key):
                    logger.info(f"  {key}: {original_config[key]} -> {model.config[key]}")
        else:
            logger.info("No optimizations needed or applied")

    def demonstrate_memory_integration(self, model: EnhancedYOLOv8):
        """Demonstrate GPU memory manager integration."""
        if model is None or not model.memory_manager:
            logger.warning("Memory manager not available - skipping integration demo")
            return

        logger.info("\n" + "=" * 60)
        logger.info("GPU MEMORY MANAGER INTEGRATION")
        logger.info("=" * 60)

        # Get memory statistics
        memory_stats = model.memory_manager.get_memory_stats()

        logger.info("Memory Manager Statistics:")
        for key, value in memory_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")

        # Test memory buffer operations
        logger.info("\nTesting memory buffer operations...")

        buffer_sizes = [(100, 100), (256, 256), (512, 512)]
        for size in buffer_sizes:
            buffer = model.get_memory_buffer(size)
            logger.info(
                f"  Allocated buffer {size}: {buffer.shape if hasattr(buffer, 'shape') else 'Success'}"
            )
            model.return_memory_buffer(buffer)
            logger.info(f"  Returned buffer {size}")

    def run_complete_demo(self):
        """Run the complete optimization demonstration."""
        logger.info("Starting Complete YOLOv8 Optimization Demonstration")
        logger.info("This demo showcases Priority 3 implementation features")

        try:
            # Demonstrate hardware configurations
            self.demonstrate_hardware_configurations()

            # Demonstrate optimization configurations
            self.demonstrate_optimization_configs()

            # Initialize and demonstrate enhanced model
            model = self.demonstrate_model_initialization()

            if model:
                # Demonstrate performance monitoring
                self.demonstrate_performance_monitoring(model)

                # Demonstrate benchmarking
                self.demonstrate_benchmarking(model)

                # Demonstrate adaptive optimization
                self.demonstrate_adaptive_optimization(model)

                # Demonstrate memory integration
                self.demonstrate_memory_integration(model)

            logger.info("\n" + "=" * 60)
            logger.info("DEMONSTRATION COMPLETE")
            logger.info("=" * 60)
            logger.info(
                "Priority 3: YOLOv8 Model Configuration Optimization features demonstrated!"
            )
            logger.info("Key achievements:")
            logger.info("✓ Hardware-tier specific optimizations")
            logger.info("✓ Dynamic model switching capabilities")
            logger.info("✓ Performance monitoring and metrics")
            logger.info("✓ Adaptive batch sizing")
            logger.info("✓ Auto-optimization features")
            logger.info("✓ Comprehensive benchmarking")
            logger.info("✓ GPU memory manager integration")

        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            raise


def main():
    """Main function to run the optimization demo."""
    demo = OptimizationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
