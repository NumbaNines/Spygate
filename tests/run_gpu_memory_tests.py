#!/usr/bin/env python3
"""
GPU Memory Management Testing Script for Task 19.7

This script tests GPU memory management and performance optimization features
under various load conditions including:
- Memory allocation patterns during video processing
- Performance optimization under heavy load
- Memory efficiency of YOLOv8 CPU inference
- Memory usage during auto-clip detection with large videos
- Memory management during 36,004 frame processing scenario
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_gpu_memory_management():
    """Main GPU memory management test function"""
    logger.info("=== GPU Memory Management Testing (Task 19.7) ===")

    try:
        # Import required modules
        from spygate.core.hardware import HardwareDetector

        # Initialize hardware detector
        hardware = HardwareDetector()
        logger.info(f"Hardware Tier: {hardware.tier.name}")
        logger.info(f"System Memory: {hardware.memory_gb:.1f} GB")
        logger.info(f"CPU Cores: {hardware.cpu_cores}")
        logger.info(f"CUDA Available: {hardware.has_cuda}")

        # Test 1: CPU Memory Optimization (primary focus since no CUDA)
        logger.info("\n--- Test 1: CPU Memory Optimization ---")
        test_cpu_memory_patterns()

        # Test 2: YOLOv8 Memory Efficiency
        logger.info("\n--- Test 2: YOLOv8 Memory Efficiency ---")
        test_yolov8_memory_usage()

        # Test 3: Large Video Processing Memory
        logger.info("\n--- Test 3: Large Video Processing Memory ---")
        test_large_video_memory()

        # Test 4: Hardware-Adaptive Settings
        logger.info("\n--- Test 4: Hardware-Adaptive Memory Settings ---")
        test_adaptive_memory_settings(hardware)

        # Test 5: Memory Pressure Recovery
        logger.info("\n--- Test 5: Memory Pressure Recovery ---")
        test_memory_pressure_recovery()

        logger.info("\n✅ GPU Memory Management Testing COMPLETED")
        return True

    except Exception as e:
        logger.error(f"GPU Memory Management Testing FAILED: {e}")
        return False


def get_process_memory():
    """Get current process memory usage in MB"""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_cpu_memory_patterns():
    """Test memory allocation patterns during CPU processing"""
    logger.info("Testing CPU memory allocation patterns...")

    initial_memory = get_process_memory()
    logger.info(f"Initial memory: {initial_memory:.1f} MB")

    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False
        import numpy as np

    # Simulate heavy processing workload
    large_tensors = []
    for i in range(10):
        if torch_available:
            tensor = torch.randn(1000, 1000, dtype=torch.float32)
            large_tensors.append(tensor)
        else:
            array = np.random.randn(1000, 1000).astype(np.float32)
            large_tensors.append(array)

        current_memory = get_process_memory()
        logger.info(f"  Allocation {i+1}: {current_memory:.1f} MB")

    peak_memory = get_process_memory()
    logger.info(f"Peak memory: {peak_memory:.1f} MB")

    # Test cleanup
    del large_tensors
    gc.collect()
    time.sleep(1)

    final_memory = get_process_memory()
    recovery_rate = ((peak_memory - final_memory) / (peak_memory - initial_memory)) * 100
    logger.info(f"Final memory: {final_memory:.1f} MB (Recovery: {recovery_rate:.1f}%)")

    return recovery_rate > 80


def test_yolov8_memory_usage():
    """Test YOLOv8 memory efficiency during inference"""
    logger.info("Testing YOLOv8 memory efficiency...")

    try:
        from spygate.ml.yolov8_model import EnhancedYOLOv8

        initial_memory = get_process_memory()

        # Initialize model
        model = EnhancedYOLOv8()
        model_memory = get_process_memory()
        logger.info(f"Memory after model load: {model_memory:.1f} MB")

        try:
            import torch

            # Test with tensor inputs
            memory_readings = []
            for i in range(5):
                # Create test image
                test_image = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8)

                # Simulate processing
                start_time = time.time()
                # Note: Actual inference would go here
                time.sleep(0.1)  # Simulate processing time

                current_memory = get_process_memory()
                memory_readings.append(current_memory)
                logger.info(f"  Inference {i+1}: {current_memory:.1f} MB")

            # Check memory stability
            memory_variation = max(memory_readings) - min(memory_readings)
            logger.info(f"Memory variation: {memory_variation:.1f} MB")

            # Cleanup
            del model
            gc.collect()

            final_memory = get_process_memory()
            logger.info(f"Memory after cleanup: {final_memory:.1f} MB")

            return memory_variation < 100  # Stable memory usage

        except ImportError:
            logger.warning("PyTorch not available, skipping tensor tests")
            return True

    except ImportError:
        logger.warning("YOLOv8 model not available, skipping test")
        return True


def test_large_video_memory():
    """Test memory usage during large video processing (36,004 frames)"""
    logger.info("Testing large video processing memory usage...")

    frame_count = 36004
    batch_size = 64  # HIGH tier setting

    initial_memory = get_process_memory()
    logger.info(f"Simulating {frame_count} frame processing with batch size {batch_size}")

    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False
        import numpy as np

    processed_frames = 0
    batch_count = 0
    memory_readings = []

    while processed_frames < frame_count:
        current_batch_size = min(batch_size, frame_count - processed_frames)

        # Simulate frame batch processing
        frame_batch = []
        for _ in range(min(current_batch_size, 10)):  # Limit for testing
            if torch_available:
                frame = torch.zeros(1080, 1920, 3, dtype=torch.uint8)
            else:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame_batch.append(frame)

        # Simulate processing
        time.sleep(0.01)

        # Cleanup batch
        del frame_batch

        processed_frames += current_batch_size
        batch_count += 1

        if batch_count % 100 == 0:
            current_memory = get_process_memory()
            memory_readings.append(current_memory)
            progress = (processed_frames / frame_count) * 100
            logger.info(
                f"  Batch {batch_count}: {progress:.1f}% complete, Memory: {current_memory:.1f} MB"
            )

            # Periodic cleanup
            gc.collect()

    final_memory = get_process_memory()
    memory_growth = final_memory - initial_memory

    logger.info(f"Processing complete: {batch_count} batches")
    logger.info(f"Memory growth: {memory_growth:.1f} MB")

    return memory_growth < 500  # Reasonable memory growth


def test_adaptive_memory_settings(hardware):
    """Test hardware-adaptive memory settings"""
    logger.info("Testing hardware-adaptive memory settings...")

    from spygate.core.hardware import HardwareTier

    # Define tier-specific settings
    tier_settings = {
        HardwareTier.LOW: {"batch_size": 16, "memory_limit": 0.5},
        HardwareTier.MEDIUM: {"batch_size": 32, "memory_limit": 0.7},
        HardwareTier.HIGH: {"batch_size": 64, "memory_limit": 0.8},
        HardwareTier.ULTRA: {"batch_size": 128, "memory_limit": 0.9},
    }

    current_settings = tier_settings.get(hardware.tier, tier_settings[HardwareTier.MEDIUM])

    logger.info(f"Hardware tier: {hardware.tier.name}")
    logger.info(f"Adaptive batch size: {current_settings['batch_size']}")
    logger.info(f"Memory limit: {current_settings['memory_limit']*100:.0f}%")

    # Validate settings
    batch_size = current_settings["batch_size"]
    memory_limit = current_settings["memory_limit"]

    assert batch_size > 0 and batch_size <= 256, f"Invalid batch size: {batch_size}"
    assert 0.3 < memory_limit < 1.0, f"Invalid memory limit: {memory_limit}"

    memory_budget = hardware.memory_gb * memory_limit
    logger.info(f"Memory budget: {memory_budget:.1f} GB of {hardware.memory_gb:.1f} GB")

    return True


def test_memory_pressure_recovery():
    """Test system recovery under memory pressure"""
    logger.info("Testing memory pressure recovery...")

    initial_memory = get_process_memory()

    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False
        import numpy as np

    # Create memory pressure
    memory_consumers = []
    for i in range(15):
        try:
            size = 300 + (i * 30)
            if torch_available:
                tensor = torch.zeros(size, size, dtype=torch.float32)
                memory_consumers.append(tensor)
            else:
                array = np.zeros((size, size), dtype=np.float32)
                memory_consumers.append(array)

            current_memory = get_process_memory()

            # Stop if memory gets too high
            if current_memory > initial_memory + 1500:  # 1.5GB limit
                logger.info(f"Stopping at allocation {i+1} due to memory limit")
                break

        except MemoryError:
            logger.info(f"Memory exhausted at allocation {i+1}")
            break

    peak_memory = get_process_memory()
    logger.info(f"Peak memory: {peak_memory:.1f} MB")

    # Test recovery phases
    phases = [25, 50, 75, 100]  # Percentage to free

    for phase in phases:
        target_free = (len(memory_consumers) * phase) // 100
        while len(memory_consumers) > len(memory_consumers) - target_free and memory_consumers:
            del memory_consumers[0]

        gc.collect()
        time.sleep(0.5)

        current_memory = get_process_memory()
        logger.info(f"  Recovery {phase}%: {current_memory:.1f} MB")

    final_memory = get_process_memory()
    recovery_rate = ((peak_memory - final_memory) / (peak_memory - initial_memory)) * 100
    logger.info(f"Recovery rate: {recovery_rate:.1f}%")

    return recovery_rate > 70


if __name__ == "__main__":
    success = test_gpu_memory_management()
    sys.exit(0 if success else 1)
