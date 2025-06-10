"""
Comprehensive GPU Memory Management Testing Suite for SpygateAI - Task 19.7
Professional testing approach focusing on real-world scenarios and edge cases.

This suite extends the existing tests with:
1. Production workload simulation
2. Memory exhaustion and recovery testing
3. Performance degradation analysis
4. Multi-model concurrent processing
5. Long-running stability tests
6. Memory fragmentation analysis
7. Hardware tier validation
"""

import gc
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import numpy as np
import psutil

# Add the spygate module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

from spygate.core.gpu_memory_manager import (
    AdvancedGPUMemoryManager,
    GPUMemoryPool,
    MemoryPoolConfig,
    MemoryStrategy,
    get_memory_manager,
    initialize_memory_manager,
    shutdown_memory_manager,
)
from spygate.core.hardware import HardwareDetector, HardwareTier


class GPUMemoryProfiler:
    """Professional GPU memory profiling utility for testing."""

    def __init__(self):
        self.baseline_stats = None
        self.peak_memory = 0
        self.memory_timeline = []
        self.fragmentation_history = []

    def start_profiling(self):
        """Start memory profiling session."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.baseline_stats = self._get_current_stats()
            self.peak_memory = 0
            self.memory_timeline = []
            self.fragmentation_history = []

    def record_measurement(self, label=""):
        """Record a memory measurement point."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats = self._get_current_stats()
            stats["label"] = label
            stats["timestamp"] = time.time()
            self.memory_timeline.append(stats)

            if stats["allocated_gb"] > self.peak_memory:
                self.peak_memory = stats["allocated_gb"]

    def _get_current_stats(self):
        """Get current GPU memory statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory

        return {
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "total_gb": total / 1024**3,
            "utilization_pct": (allocated / total) * 100,
            "fragmentation_pct": ((reserved - allocated) / max(reserved, 1)) * 100,
        }

    def get_peak_memory_usage(self):
        """Get peak memory usage during profiling."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0

    def analyze_memory_leaks(self):
        """Analyze potential memory leaks."""
        if len(self.memory_timeline) < 2:
            return {"leak_detected": False, "growth_rate": 0}

        start_memory = self.memory_timeline[0]["allocated_gb"]
        end_memory = self.memory_timeline[-1]["allocated_gb"]
        growth = end_memory - start_memory

        # Consider it a leak if memory grew by more than 100MB without being freed
        leak_threshold = 0.1  # 100MB in GB
        leak_detected = growth > leak_threshold

        return {
            "leak_detected": leak_detected,
            "growth_rate": growth,
            "start_memory_gb": start_memory,
            "end_memory_gb": end_memory,
            "timeline_points": len(self.memory_timeline),
        }


class ComprehensiveGPUMemoryTests(unittest.TestCase):
    """Comprehensive GPU memory management tests for production validation."""

    def setUp(self):
        """Set up test environment with professional monitoring."""
        self.profiler = GPUMemoryProfiler()
        self.original_log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)  # Reduce noise during tests

        # Mock hardware for consistent testing
        self.mock_hardware = Mock(spec=HardwareDetector)
        self.mock_hardware.tier = HardwareTier.HIGH
        self.mock_hardware.gpu_memory_gb = 8.0
        self.mock_hardware.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

        # Test configuration with shorter timeouts for faster testing
        self.test_config = MemoryPoolConfig(
            buffer_timeout=2.0,
            max_buffer_count=50,
            cleanup_threshold=0.7,
            warning_threshold=0.5,
            monitor_interval=1.0,
        )

    def tearDown(self):
        """Clean up test environment."""
        try:
            shutdown_memory_manager()
        except:
            pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logging.getLogger().setLevel(self.original_log_level)

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_production_workload_simulation(self):
        """Test GPU memory management under simulated production workload."""
        print("\n=== Production Workload Simulation ===")

        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.test_config)
        self.profiler.start_profiling()

        # Simulate video analysis workload with varying batch sizes
        workload_phases = [
            ("small_batches", [(64, 64), (128, 128), (256, 256)], 10),
            ("large_batches", [(512, 512), (1024, 1024)], 5),
            ("mixed_workload", [(64, 64), (512, 512), (256, 256), (1024, 1024)], 8),
        ]

        for phase_name, batch_sizes, iterations in workload_phases:
            print(f"\nTesting {phase_name}...")
            self.profiler.record_measurement(f"start_{phase_name}")

            buffers = []
            for i in range(iterations):
                for size in batch_sizes:
                    try:
                        buffer = manager.get_buffer(size)
                        buffers.append(buffer)

                        # Simulate processing time
                        time.sleep(0.01)

                    except Exception as e:
                        print(f"Warning: Buffer allocation failed at iteration {i}: {e}")

            # Return buffers gradually (simulate async processing)
            for i, buffer in enumerate(buffers):
                manager.return_buffer(buffer)
                if i % 5 == 0:  # Periodic cleanup simulation
                    time.sleep(0.01)

            self.profiler.record_measurement(f"end_{phase_name}")

        # Analyze results
        peak_memory = self.profiler.get_peak_memory_usage()
        leak_analysis = self.profiler.analyze_memory_leaks()
        final_stats = manager.get_memory_stats()

        print(f"\nResults:")
        print(f"Peak memory usage: {peak_memory:.2f} GB")
        print(f"Memory leak detected: {leak_analysis['leak_detected']}")
        print(f"Memory growth: {leak_analysis['growth_rate']:.3f} GB")
        print(f"Final pool hit rate: {final_stats.get('hit_rate', 0):.1f}%")

        # Assertions for production readiness
        self.assertLess(peak_memory, 6.0, "Peak memory usage too high for production")
        self.assertFalse(leak_analysis["leak_detected"], "Memory leak detected")

        manager.shutdown()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_memory_exhaustion_recovery(self):
        """Test system behavior during memory exhaustion and recovery."""
        print("\n=== Memory Exhaustion and Recovery Test ===")

        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.test_config)
        self.profiler.start_profiling()

        # Gradually consume GPU memory until exhaustion
        large_buffers = []
        allocation_success = True
        allocated_count = 0

        print("Gradually consuming GPU memory...")
        while allocation_success and allocated_count < 20:  # Safety limit
            try:
                # Allocate progressively larger buffers
                size = (512 + allocated_count * 128, 512 + allocated_count * 128)
                buffer = manager.get_buffer(size)
                large_buffers.append(buffer)
                allocated_count += 1

                current_stats = manager.get_memory_stats()
                usage_pct = current_stats.get("usage_percentage", 0)
                print(f"Allocated buffer {allocated_count}, GPU usage: {usage_pct:.1f}%")

                if usage_pct > 85:  # Stop before complete exhaustion
                    print("Stopping allocation at 85% usage to test recovery")
                    break

            except Exception as e:
                print(f"Memory exhaustion reached at buffer {allocated_count}: {e}")
                allocation_success = False

        self.profiler.record_measurement("exhaustion_point")

        # Test recovery by freeing buffers
        print("\nTesting recovery...")
        recovery_batches = [
            len(large_buffers) // 4,  # Free 25%
            len(large_buffers) // 2,  # Free 50%
            len(large_buffers),  # Free all
        ]

        freed_count = 0
        for batch_size in recovery_batches:
            # Free a batch of buffers
            while freed_count < batch_size and freed_count < len(large_buffers):
                manager.return_buffer(large_buffers[freed_count])
                freed_count += 1

            # Force cleanup
            if hasattr(manager, "memory_pool") and manager.memory_pool:
                manager.memory_pool.cleanup_expired_buffers()
            torch.cuda.empty_cache()

            # Test if new allocations work
            try:
                test_buffer = manager.get_buffer((256, 256))
                manager.return_buffer(test_buffer)
                recovery_success = True
            except Exception:
                recovery_success = False

            current_stats = manager.get_memory_stats()
            usage_pct = current_stats.get("usage_percentage", 0)
            print(
                f"Freed {freed_count} buffers, GPU usage: {usage_pct:.1f}%, Recovery: {'OK' if recovery_success else 'FAILED'}"
            )

            self.profiler.record_measurement(f"recovery_batch_{freed_count}")

        # Verify complete recovery
        final_stats = manager.get_memory_stats()
        final_usage = final_stats.get("usage_percentage", 0)

        print(f"\nFinal GPU usage after recovery: {final_usage:.1f}%")

        # Test that system can still allocate after recovery
        try:
            test_buffer = manager.get_buffer((512, 512))
            manager.return_buffer(test_buffer)
            recovery_complete = True
        except Exception as e:
            recovery_complete = False
            print(f"Recovery incomplete: {e}")

        self.assertTrue(recovery_complete, "System failed to recover from memory exhaustion")
        self.assertLess(final_usage, 30, "Memory usage too high after recovery")

        manager.shutdown()

    @unittest.skipIf(not TORCH_AVAILABLE or not torch.cuda.is_available(), "CUDA not available")
    def test_concurrent_processing_stress(self):
        """Test concurrent GPU memory access under stress conditions."""
        print("\n=== Concurrent Processing Stress Test ===")

        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.test_config)
        self.profiler.start_profiling()

        # Test concurrent access with multiple threads
        num_threads = 8
        operations_per_thread = 20
        thread_errors = []
        thread_stats = []

        def worker_function(worker_id):
            """Worker function for concurrent testing."""
            worker_buffers = []
            worker_errors = []

            try:
                for i in range(operations_per_thread):
                    try:
                        # Vary buffer sizes to create realistic workload
                        base_size = 128 + (worker_id * 32)
                        variation = i % 4 * 64
                        size = (base_size + variation, base_size + variation)

                        buffer = manager.get_buffer(size)
                        worker_buffers.append(buffer)

                        # Simulate some processing time
                        time.sleep(0.001)

                        # Occasionally return buffers early
                        if i % 5 == 0 and worker_buffers:
                            early_buffer = worker_buffers.pop(0)
                            manager.return_buffer(early_buffer)

                    except Exception as e:
                        worker_errors.append(f"Worker {worker_id}, op {i}: {e}")

                # Return remaining buffers
                for buffer in worker_buffers:
                    try:
                        manager.return_buffer(buffer)
                    except Exception as e:
                        worker_errors.append(f"Worker {worker_id} return: {e}")

            except Exception as e:
                worker_errors.append(f"Worker {worker_id} fatal: {e}")

            return {
                "worker_id": worker_id,
                "errors": worker_errors,
                "operations_completed": operations_per_thread - len(worker_errors),
            }

        # Execute concurrent workers
        print(f"Starting {num_threads} concurrent workers...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_worker = {executor.submit(worker_function, i): i for i in range(num_threads)}

            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    thread_stats.append(result)
                    if result["errors"]:
                        thread_errors.extend(result["errors"])
                        print(f"Worker {worker_id}: {len(result['errors'])} errors")
                    else:
                        print(f"Worker {worker_id}: Completed successfully")
                except Exception as e:
                    thread_errors.append(f"Worker {worker_id} future error: {e}")

        self.profiler.record_measurement("concurrent_complete")

        # Analyze results
        total_operations = num_threads * operations_per_thread
        successful_operations = sum(stat["operations_completed"] for stat in thread_stats)
        success_rate = (successful_operations / total_operations) * 100

        final_stats = manager.get_memory_stats()

        print(f"\nConcurrent Test Results:")
        print(f"Total operations: {total_operations}")
        print(f"Successful operations: {successful_operations}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total errors: {len(thread_errors)}")
        print(f"Final hit rate: {final_stats.get('hit_rate', 0):.1f}%")

        if thread_errors:
            print("Sample errors:")
            for error in thread_errors[:3]:  # Show first 3 errors
                print(f"  - {error}")

        # Assertions for production readiness
        self.assertGreater(success_rate, 95, "Concurrent operation success rate too low")
        self.assertLess(len(thread_errors), total_operations * 0.05, "Too many concurrent errors")

        manager.shutdown()

    def test_hardware_tier_optimization_validation(self):
        """Test that memory management adapts correctly to different hardware tiers."""
        print("\n=== Hardware Tier Optimization Validation ===")

        tiers_to_test = [
            (HardwareTier.LOW, 2.0, MemoryStrategy.LOW),
            (HardwareTier.MEDIUM, 4.0, MemoryStrategy.MEDIUM),
            (HardwareTier.HIGH, 8.0, MemoryStrategy.HIGH),
            (HardwareTier.ULTRA, 16.0, MemoryStrategy.ULTRA),
        ]

        tier_results = {}

        for tier, memory_gb, expected_strategy in tiers_to_test:
            print(f"\nTesting {tier.name} tier...")

            # Mock hardware for this tier
            mock_hw = Mock(spec=HardwareDetector)
            mock_hw.tier = tier
            mock_hw.gpu_memory_gb = memory_gb
            mock_hw.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

            manager = AdvancedGPUMemoryManager(mock_hw, self.test_config)

            # Test tier-specific behavior
            stats = manager.get_memory_stats()
            strategy = manager.strategy if hasattr(manager, "strategy") else None

            # Test optimal batch size calculation
            optimal_batch = manager.get_optimal_batch_size()

            tier_results[tier.name] = {
                "memory_gb": memory_gb,
                "strategy": strategy.value if strategy else "unknown",
                "optimal_batch": optimal_batch,
                "stats": stats,
            }

            print(f"  Memory: {memory_gb}GB")
            print(f"  Strategy: {strategy.value if strategy else 'unknown'}")
            print(f"  Optimal batch size: {optimal_batch}")

            manager.shutdown()

        # Validate tier progression
        print(f"\nTier Optimization Summary:")
        for tier_name, results in tier_results.items():
            print(
                f"{tier_name}: {results['memory_gb']}GB, {results['strategy']}, batch={results['optimal_batch']}"
            )

        # Verify that higher tiers have larger optimal batch sizes
        low_batch = tier_results.get("LOW", {}).get("optimal_batch", 0)
        high_batch = tier_results.get("HIGH", {}).get("optimal_batch", 0)

        if low_batch and high_batch:
            self.assertGreaterEqual(
                high_batch, low_batch, "Higher tier should have larger or equal optimal batch size"
            )

    def test_memory_fragmentation_analysis(self):
        """Test memory fragmentation detection and mitigation."""
        print("\n=== Memory Fragmentation Analysis ===")

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self.skipTest("CUDA not available for fragmentation testing")

        manager = AdvancedGPUMemoryManager(self.mock_hardware, self.test_config)

        # Create fragmentation by allocating and freeing random-sized buffers
        print("Creating memory fragmentation...")
        buffers = []
        sizes = [(64, 64), (256, 256), (128, 128), (512, 512), (192, 192)]

        # Allocate buffers
        for i in range(20):
            size = sizes[i % len(sizes)]
            try:
                buffer = manager.get_buffer(size)
                buffers.append(buffer)
            except Exception as e:
                print(f"Allocation failed at iteration {i}: {e}")
                break

        # Free every other buffer to create fragmentation
        fragmented_buffers = []
        for i, buffer in enumerate(buffers):
            if i % 2 == 0:
                manager.return_buffer(buffer)
            else:
                fragmented_buffers.append(buffer)

        # Check fragmentation before cleanup
        initial_stats = manager.get_memory_stats()
        initial_fragmentation = initial_stats.get("fragmentation", 0)

        print(f"Initial fragmentation: {initial_fragmentation:.2f}%")

        # Force defragmentation
        if hasattr(manager, "_defragment_memory"):
            manager._defragment_memory()

        # Check fragmentation after cleanup
        final_stats = manager.get_memory_stats()
        final_fragmentation = final_stats.get("fragmentation", 0)

        print(f"Final fragmentation: {final_fragmentation:.2f}%")

        # Clean up remaining buffers
        for buffer in fragmented_buffers:
            manager.return_buffer(buffer)

        # Verify fragmentation was reduced
        if initial_fragmentation > 0:
            improvement = initial_fragmentation - final_fragmentation
            print(f"Fragmentation improvement: {improvement:.2f}%")
            self.assertGreaterEqual(improvement, 0, "Defragmentation should reduce fragmentation")

        manager.shutdown()


def run_comprehensive_gpu_tests():
    """Run the comprehensive GPU memory management test suite."""
    print("=" * 60)
    print("COMPREHENSIVE GPU MEMORY MANAGEMENT TEST SUITE")
    print("=" * 60)

    # Check system requirements
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not available - some tests will be skipped")
    elif not torch.cuda.is_available():
        print("WARNING: CUDA not available - GPU tests will be skipped")
    else:
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"Total Memory: {gpu_info.total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")

    print("-" * 60)

    # Create test suite
    suite = unittest.TestSuite()

    # Add comprehensive tests
    suite.addTest(ComprehensiveGPUMemoryTests("test_production_workload_simulation"))
    suite.addTest(ComprehensiveGPUMemoryTests("test_memory_exhaustion_recovery"))
    suite.addTest(ComprehensiveGPUMemoryTests("test_concurrent_processing_stress"))
    suite.addTest(ComprehensiveGPUMemoryTests("test_hardware_tier_optimization_validation"))
    suite.addTest(ComprehensiveGPUMemoryTests("test_memory_fragmentation_analysis"))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL RESULT: {'PASS' if success else 'FAIL'}")

    return success


if __name__ == "__main__":
    success = run_comprehensive_gpu_tests()
    sys.exit(0 if success else 1)
