"""
Comprehensive GPU Memory Management Testing Suite for SpygateAI

This test suite validates:
1. GPU memory pool management and buffer reuse
2. Adaptive batch sizing under memory constraints
3. Memory fragmentation prevention and cleanup
4. Hardware-tier specific optimization strategies
5. Performance under various load conditions
6. Memory leak detection and emergency cleanup
7. Cross-platform compatibility (CUDA available/unavailable)
"""

import gc
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add the spygate module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spygate.core.gpu_memory_manager import (
    AdvancedGPUMemoryManager,
    GPUMemoryPool,
    MemoryBuffer,
    MemoryPoolConfig,
    MemoryStrategy,
    get_memory_manager,
    initialize_memory_manager,
    shutdown_memory_manager,
)
from spygate.core.hardware import HardwareDetector, HardwareTier


class TestMemoryPoolConfig(unittest.TestCase):
    """Test memory pool configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryPoolConfig()

        # Pool configuration
        self.assertEqual(config.initial_pool_size, 0.5)
        self.assertEqual(config.max_pool_size, 0.8)
        self.assertEqual(config.min_pool_size, 0.2)

        # Buffer management
        self.assertEqual(config.buffer_growth_factor, 1.5)
        self.assertEqual(config.max_buffer_count, 100)
        self.assertEqual(config.buffer_timeout, 300.0)

        # Memory monitoring
        self.assertEqual(config.cleanup_threshold, 0.85)
        self.assertEqual(config.warning_threshold, 0.75)
        self.assertEqual(config.monitor_interval, 10.0)

        # Fragmentation management
        self.assertEqual(config.defrag_threshold, 0.3)
        self.assertEqual(config.defrag_interval, 600.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MemoryPoolConfig(
            initial_pool_size=0.3, max_pool_size=0.7, cleanup_threshold=0.9, monitor_interval=5.0
        )

        self.assertEqual(config.initial_pool_size, 0.3)
        self.assertEqual(config.max_pool_size, 0.7)
        self.assertEqual(config.cleanup_threshold, 0.9)
        self.assertEqual(config.monitor_interval, 5.0)


class TestMemoryStrategy(unittest.TestCase):
    """Test memory strategy enumeration."""

    def test_strategy_values(self):
        """Test that all strategies have expected values."""
        self.assertEqual(MemoryStrategy.ULTRA_LOW.value, "conservative")
        self.assertEqual(MemoryStrategy.LOW.value, "balanced")
        self.assertEqual(MemoryStrategy.MEDIUM.value, "adaptive")
        self.assertEqual(MemoryStrategy.HIGH.value, "performance")
        self.assertEqual(MemoryStrategy.ULTRA.value, "unlimited")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestGPUMemoryPool(unittest.TestCase):
    """Test GPU memory pool functionality."""

    def setUp(self):
        """Set up test environment."""
        self.config = MemoryPoolConfig(
            buffer_timeout=1.0, max_buffer_count=10  # Short timeout for testing
        )
        self.pool = GPUMemoryPool(self.config)

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "pool"):
            self.pool._emergency_cleanup()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_buffer_allocation_and_reuse(self):
        """Test buffer allocation and reuse functionality."""
        size = (100, 100)
        dtype = torch.float32

        # Get first buffer
        buffer1 = self.pool.get_buffer(size, dtype)
        self.assertIsInstance(buffer1, torch.Tensor)
        self.assertEqual(buffer1.shape, size)
        self.assertEqual(buffer1.dtype, dtype)
        self.assertTrue(buffer1.is_cuda)

        # Return buffer to pool
        self.pool.return_buffer(buffer1)

        # Get buffer again - should reuse the same buffer
        buffer2 = self.pool.get_buffer(size, dtype)
        self.assertEqual(buffer1.data_ptr(), buffer2.data_ptr())

        # Verify statistics
        stats = self.pool.get_statistics()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["allocations"], 1)

    def test_buffer_key_generation(self):
        """Test buffer key generation for different sizes and types."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        key1 = self.pool._get_buffer_key((100, 100), torch.float32)
        key2 = self.pool._get_buffer_key((100, 100), torch.float16)
        key3 = self.pool._get_buffer_key((200, 200), torch.float32)

        self.assertNotEqual(key1, key2)  # Different dtypes
        self.assertNotEqual(key1, key3)  # Different sizes
        self.assertNotEqual(key2, key3)  # Different sizes and dtypes

    def test_buffer_cleanup(self):
        """Test expired buffer cleanup."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        size = (50, 50)

        # Allocate and return buffer
        buffer = self.pool.get_buffer(size)
        self.pool.return_buffer(buffer)

        # Wait for buffer to expire
        time.sleep(1.1)

        # Clean up expired buffers
        self.pool.cleanup_expired_buffers()

        # Verify cleanup
        stats = self.pool.get_statistics()
        self.assertEqual(stats["deallocations"], 1)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_emergency_cleanup(self):
        """Test emergency cleanup functionality."""
        size = (100, 100)

        # Allocate multiple buffers
        buffers = []
        for _ in range(5):
            buffer = self.pool.get_buffer(size)
            buffers.append(buffer)

        # Return some buffers
        for buffer in buffers[:3]:
            self.pool.return_buffer(buffer)

        # Perform emergency cleanup
        initial_stats = self.pool.get_statistics()
        self.pool._emergency_cleanup()
        final_stats = self.pool.get_statistics()

        # Verify cleanup occurred
        self.assertGreater(final_stats["deallocations"], initial_stats["deallocations"])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_statistics_tracking(self):
        """Test memory pool statistics tracking."""
        size = (50, 50)

        # Initial statistics
        initial_stats = self.pool.get_statistics()
        self.assertEqual(initial_stats["allocations"], 0)
        self.assertEqual(initial_stats["cache_hits"], 0)
        self.assertEqual(initial_stats["cache_misses"], 0)

        # Allocate buffer
        buffer = self.pool.get_buffer(size)
        stats_after_alloc = self.pool.get_statistics()
        self.assertEqual(stats_after_alloc["allocations"], 1)
        self.assertEqual(stats_after_alloc["cache_misses"], 1)

        # Return and reuse buffer
        self.pool.return_buffer(buffer)
        buffer2 = self.pool.get_buffer(size)
        stats_after_reuse = self.pool.get_statistics()
        self.assertEqual(stats_after_reuse["cache_hits"], 1)

        # Check hit rate calculation
        self.assertEqual(stats_after_reuse["hit_rate"], 50.0)  # 1 hit out of 2 requests


class TestAdvancedGPUMemoryManager(unittest.TestCase):
    """Test advanced GPU memory manager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.mock_hardware = Mock(spec=HardwareDetector)
        self.mock_hardware.tier = HardwareTier.HIGH
        self.mock_hardware.gpu_memory_gb = 8.0
        self.mock_hardware.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

        self.config = MemoryPoolConfig(monitor_interval=0.1)  # Fast monitoring for tests

    def tearDown(self):
        """Clean up after tests."""
        # Ensure any created managers are properly shut down
        try:
            shutdown_memory_manager()
        except:
            pass

    def test_initialization_with_hardware_tiers(self):
        """Test manager initialization with different hardware tiers."""
        # Test different hardware tiers
        tiers_and_strategies = [
            (HardwareTier.ULTRA_LOW, MemoryStrategy.ULTRA_LOW),
            (HardwareTier.LOW, MemoryStrategy.LOW),
            (HardwareTier.MEDIUM, MemoryStrategy.MEDIUM),
            (HardwareTier.HIGH, MemoryStrategy.HIGH),
            (HardwareTier.ULTRA, MemoryStrategy.ULTRA),
        ]

        for tier, expected_strategy in tiers_and_strategies:
            with self.subTest(tier=tier):
                mock_hw = Mock(spec=HardwareDetector)
                mock_hw.tier = tier
                mock_hw.gpu_memory_gb = 8.0
                mock_hw.has_cuda = False  # Avoid CUDA operations in test

                manager = AdvancedGPUMemoryManager(mock_hw, self.config)
                self.assertEqual(manager.strategy, expected_strategy)
                manager.shutdown()

    def test_memory_strategy_assignment(self):
        """Test that memory strategies are correctly assigned based on hardware."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Should use HIGH strategy for HIGH tier hardware
        self.assertEqual(manager.strategy, MemoryStrategy.HIGH)
        manager.shutdown()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_memory_monitoring_thread(self):
        """Test memory monitoring thread functionality."""
        self.mock_hardware.has_cuda = True
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Wait for monitoring thread to start
        time.sleep(0.2)

        # Verify monitoring thread is running
        self.assertTrue(manager.monitoring_thread.is_alive())

        manager.shutdown()

        # Verify monitoring thread stopped
        time.sleep(0.2)
        self.assertFalse(manager.monitoring_thread.is_alive())

    def test_memory_stats_collection(self):
        """Test memory statistics collection."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        stats = manager.get_memory_stats()

        # Verify basic stats are present
        self.assertIn("strategy", stats)
        self.assertIn("hardware_tier", stats)
        self.assertEqual(stats["strategy"], MemoryStrategy.HIGH.value)
        self.assertEqual(stats["hardware_tier"], HardwareTier.HIGH.name)

        manager.shutdown()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_memory_stats(self):
        """Test CUDA-specific memory statistics."""
        self.mock_hardware.has_cuda = True
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        stats = manager.get_memory_stats()

        # Should include CUDA-specific stats
        self.assertIn("total_memory_gb", stats)
        self.assertIn("allocated_memory_gb", stats)
        self.assertIn("usage_percentage", stats)
        self.assertIn("fragmentation", stats)

        manager.shutdown()

    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Test batch size calculation
        batch_size = manager.get_optimal_batch_size(model_memory_usage=1.0)
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)

        manager.shutdown()

    def test_batch_performance_recording(self):
        """Test batch performance recording."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Record successful batch
        manager.record_batch_performance(batch_size=32, processing_time=1.5, success=True)

        # Record failed batch
        manager.record_batch_performance(batch_size=64, processing_time=3.0, success=False)

        # Get updated batch size (should adapt based on performance)
        new_batch_size = manager.get_optimal_batch_size()
        self.assertIsInstance(new_batch_size, int)

        manager.shutdown()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_buffer_management(self):
        """Test buffer get/return functionality."""
        self.mock_hardware.has_cuda = True
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Get buffer
        size = (100, 100)
        buffer = manager.get_buffer(size)
        self.assertIsInstance(buffer, torch.Tensor)
        self.assertEqual(buffer.shape, size)

        # Return buffer
        manager.return_buffer(buffer)

        manager.shutdown()

    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA is not available."""
        self.mock_hardware.has_cuda = False
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Should work without CUDA
        self.assertIsNotNone(manager)
        self.assertIsNone(manager.memory_pool)  # No GPU pool

        manager.shutdown()


class TestMemoryManagerSingleton(unittest.TestCase):
    """Test memory manager singleton functionality."""

    def tearDown(self):
        """Clean up singleton state."""
        shutdown_memory_manager()

    def test_global_memory_manager(self):
        """Test global memory manager creation and access."""
        # Get manager (should create new instance)
        manager1 = get_memory_manager()
        self.assertIsInstance(manager1, AdvancedGPUMemoryManager)

        # Get manager again (should return same instance)
        manager2 = get_memory_manager()
        self.assertIs(manager1, manager2)

    def test_initialize_custom_memory_manager(self):
        """Test initializing memory manager with custom configuration."""
        mock_hardware = Mock(spec=HardwareDetector)
        mock_hardware.tier = HardwareTier.MEDIUM
        mock_hardware.gpu_memory_gb = 4.0
        mock_hardware.has_cuda = False

        custom_config = MemoryPoolConfig(cleanup_threshold=0.9)

        manager = initialize_memory_manager(mock_hardware, custom_config)
        self.assertEqual(manager.strategy, MemoryStrategy.MEDIUM)
        self.assertEqual(manager.config.cleanup_threshold, 0.9)

    def test_shutdown_memory_manager(self):
        """Test memory manager shutdown."""
        # Create manager
        manager = get_memory_manager()
        self.assertIsNotNone(manager)

        # Shutdown
        shutdown_memory_manager()

        # Should create new instance on next access
        new_manager = get_memory_manager()
        self.assertIsNot(manager, new_manager)


class TestMemoryManagerStressTests(unittest.TestCase):
    """Stress tests for memory manager under high load."""

    def setUp(self):
        """Set up stress test environment."""
        self.mock_hardware = Mock(spec=HardwareDetector)
        self.mock_hardware.tier = HardwareTier.HIGH
        self.mock_hardware.gpu_memory_gb = 8.0
        self.mock_hardware.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

        self.config = MemoryPoolConfig(
            buffer_timeout=0.5, max_buffer_count=50  # Quick cleanup for stress tests
        )

    def tearDown(self):
        """Clean up after stress tests."""
        try:
            shutdown_memory_manager()
        except:
            pass
        gc.collect()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_concurrent_buffer_access(self):
        """Test concurrent buffer allocation and deallocation."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)
        errors = []
        buffers = []

        def allocate_buffers():
            try:
                local_buffers = []
                for i in range(10):
                    size = (50 + i * 10, 50 + i * 10)
                    buffer = manager.get_buffer(size)
                    local_buffers.append(buffer)
                    time.sleep(0.01)  # Small delay

                # Return buffers
                for buffer in local_buffers:
                    manager.return_buffer(buffer)

            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=allocate_buffers)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)

        # Check for errors
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

        manager.shutdown()

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        initial_stats = manager.get_memory_stats()

        # Perform many allocations and deallocations
        for cycle in range(10):
            buffers = []
            for i in range(20):
                try:
                    if manager.memory_pool:
                        size = (100, 100)
                        buffer = manager.get_buffer(size)
                        buffers.append(buffer)
                except:
                    pass  # Handle CUDA out of memory gracefully

            # Return buffers
            for buffer in buffers:
                try:
                    manager.return_buffer(buffer)
                except:
                    pass

            # Force cleanup periodically
            if cycle % 3 == 0:
                if manager.memory_pool:
                    manager.memory_pool.cleanup_expired_buffers()

        final_stats = manager.get_memory_stats()

        # Verify no excessive memory growth
        if "total_buffers" in initial_stats and "total_buffers" in final_stats:
            buffer_growth = final_stats["total_buffers"] - initial_stats["total_buffers"]
            self.assertLess(buffer_growth, 50, "Excessive buffer growth detected")

        manager.shutdown()

    def test_performance_under_load(self):
        """Test performance characteristics under high load."""
        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.config)

        # Measure allocation performance
        start_time = time.time()
        allocation_times = []

        for i in range(100):
            alloc_start = time.time()

            if manager.memory_pool and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    size = (50, 50)
                    buffer = manager.get_buffer(size)
                    manager.return_buffer(buffer)
                except:
                    pass  # Handle CUDA limitations gracefully

            alloc_time = time.time() - alloc_start
            allocation_times.append(alloc_time)

        total_time = time.time() - start_time
        avg_allocation_time = sum(allocation_times) / len(allocation_times)

        # Performance assertions (reasonable thresholds)
        self.assertLess(total_time, 30.0, "Total test time too long")
        self.assertLess(avg_allocation_time, 0.1, "Average allocation time too long")

        manager.shutdown()


class TestMemoryManagerIntegration(unittest.TestCase):
    """Integration tests for memory manager with other components."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test environment."""
        try:
            shutdown_memory_manager()
        except:
            pass

        # Clean up temp files
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_integration_with_hardware_detector(self):
        """Test integration with real HardwareDetector."""
        try:
            from spygate.core.hardware import HardwareDetector

            # Create real hardware detector
            hardware = HardwareDetector()

            # Initialize memory manager
            manager = AdvancedGPUMemoryManager(hardware)

            # Verify it works with real hardware info
            self.assertIsNotNone(manager.strategy)
            self.assertIn(manager.strategy, MemoryStrategy)

            # Get stats
            stats = manager.get_memory_stats()
            self.assertIn("hardware_tier", stats)

            manager.shutdown()

        except ImportError:
            self.skipTest("HardwareDetector not available")

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
    def test_yolov8_integration(self):
        """Test integration with YOLOv8 model loading."""
        manager = AdvancedGPUMemoryManager(
            Mock(tier=HardwareTier.HIGH, gpu_memory_gb=8.0, has_cuda=torch.cuda.is_available())
        )

        # Simulate model memory requirements
        model_memory = 2.0  # GB
        batch_size = manager.get_optimal_batch_size(model_memory)

        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        self.assertLess(batch_size, 1000)  # Reasonable upper bound

        manager.shutdown()


def run_gpu_memory_tests():
    """Run all GPU memory management tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMemoryPoolConfig,
        TestMemoryStrategy,
        TestGPUMemoryPool,
        TestAdvancedGPUMemoryManager,
        TestMemoryManagerSingleton,
        TestMemoryManagerStressTests,
        TestMemoryManagerIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Starting GPU Memory Management Testing Suite...")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current Device: {torch.cuda.current_device()}")

    result = run_gpu_memory_tests()

    # Print summary
    print(f"\n{'='*50}")
    print("GPU Memory Management Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest suite {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)
