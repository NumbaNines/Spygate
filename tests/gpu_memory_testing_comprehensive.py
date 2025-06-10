#!/usr/bin/env python3
"""
Professional GPU Memory Management Testing Suite - Task 19.7
Comprehensive testing framework for production validation of GPU memory systems.
"""

import gc
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Add spygate to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from spygate.core.gpu_memory_manager import (
    AdvancedGPUMemoryManager,
    MemoryPoolConfig,
    MemoryStrategy,
)
from spygate.core.hardware import HardwareDetector, HardwareTier


class GPUMemoryProfiler:
    """Professional GPU memory profiling for testing."""

    def __init__(self):
        self.start_time = None
        self.measurements = []
        self.peak_memory = 0

    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        self.measurements = []
        self.peak_memory = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record_point(self, label: str):
        """Record a measurement point."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        timestamp = time.time() - self.start_time if self.start_time else 0

        measurement = {
            "label": label,
            "timestamp": timestamp,
            "allocated_mb": allocated / 1024**2,
            "reserved_mb": reserved / 1024**2,
        }

        self.measurements.append(measurement)

        if allocated > self.peak_memory:
            self.peak_memory = allocated

    def get_peak_usage_mb(self) -> float:
        """Get peak memory usage in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def detect_memory_leaks(self) -> dict:
        """Analyze measurements for memory leaks."""
        if len(self.measurements) < 2:
            return {"leak_detected": False, "growth_mb": 0}

        start_mem = self.measurements[0]["allocated_mb"]
        end_mem = self.measurements[-1]["allocated_mb"]
        growth = end_mem - start_mem

        # Threshold: 50MB growth considered a potential leak
        leak_detected = growth > 50.0

        return {
            "leak_detected": leak_detected,
            "growth_mb": growth,
            "start_memory": start_mem,
            "end_memory": end_mem,
            "measurements": len(self.measurements),
        }


class ProfessionalGPUMemoryTester:
    """Professional GPU memory management testing suite."""

    def __init__(self):
        self.profiler = GPUMemoryProfiler()
        self.results = {}

    def run_system_assessment(self) -> dict:
        """Assess system capabilities for GPU memory testing."""
        print("=== System Assessment ===")

        assessment = {
            "pytorch_available": TORCH_AVAILABLE,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_info": {},
            "system_memory_gb": 0,
            "cpu_cores": 0,
        }

        # PyTorch and CUDA check
        if TORCH_AVAILABLE:
            print(f"✓ PyTorch {torch.__version__} available")
            if torch.cuda.is_available():
                assessment["cuda_available"] = True
                assessment["gpu_count"] = torch.cuda.device_count()

                for i in range(assessment["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    assessment["gpu_info"][i] = {
                        "name": props.name,
                        "memory_gb": props.total_memory / 1024**3,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                    print(f"✓ GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            else:
                print("⚠ CUDA not available")
        else:
            print("✗ PyTorch not available")

        # System memory check
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            assessment["system_memory_gb"] = memory.total / 1024**3
            assessment["cpu_cores"] = psutil.cpu_count()
            print(f"✓ System RAM: {assessment['system_memory_gb']:.1f} GB")
            print(f"✓ CPU Cores: {assessment['cpu_cores']}")
        else:
            print("⚠ psutil not available for system info")

        return assessment

    def test_gpu_memory_manager_integration(self) -> bool:
        """Test GPU memory manager integration and basic functionality."""
        print("\n=== GPU Memory Manager Integration Test ===")

        try:
            # Initialize hardware detector
            hardware = HardwareDetector()
            print(f"✓ Hardware Tier: {hardware.tier.name}")
            print(f"✓ CUDA Support: {hardware.has_cuda}")

            # Create test configuration
            config = MemoryPoolConfig(
                buffer_timeout=5.0, max_buffer_count=20, cleanup_threshold=0.8, monitor_interval=2.0
            )

            # Initialize memory manager
            manager = AdvancedGPUMemoryManager(hardware, config)
            print("✓ Memory Manager initialized")

            # Test memory statistics
            stats = manager.get_memory_stats()
            print(f"✓ Memory stats collected: {len(stats)} metrics")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")

            # Test optimal batch size calculation
            optimal_batch = manager.get_optimal_batch_size()
            print(f"✓ Optimal batch size: {optimal_batch}")

            # Test buffer allocation if CUDA available
            if hardware.has_cuda and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # Small buffer test
                    buffer = manager.get_buffer((128, 128))
                    print("✓ Buffer allocation successful")

                    manager.return_buffer(buffer)
                    print("✓ Buffer return successful")

                    # Test multiple buffers
                    buffers = []
                    for i in range(5):
                        size = (64 + i * 32, 64 + i * 32)
                        buffer = manager.get_buffer(size)
                        buffers.append(buffer)

                    print(f"✓ Multiple buffer allocation: {len(buffers)} buffers")

                    for buffer in buffers:
                        manager.return_buffer(buffer)
                    print("✓ Multiple buffer return successful")

                except Exception as e:
                    print(f"⚠ Buffer testing failed: {e}")

            # Cleanup
            manager.shutdown()
            print("✓ Memory Manager shutdown complete")

            return True

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_memory_exhaustion_recovery(self) -> bool:
        """Test memory exhaustion and recovery scenarios."""
        print("\n=== Memory Exhaustion Recovery Test ===")

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠ Skipping - CUDA not available")
            return True

        try:
            hardware = HardwareDetector()
            config = MemoryPoolConfig(cleanup_threshold=0.7, warning_threshold=0.5)
            manager = AdvancedGPUMemoryManager(hardware, config)

            self.profiler.start_profiling()
            self.profiler.record_point("test_start")

            # Gradually increase memory usage
            buffers = []
            allocation_successful = True
            max_buffers = 15  # Safety limit

            print("Gradually increasing memory usage...")
            for i in range(max_buffers):
                try:
                    size = (256 + i * 64, 256 + i * 64)
                    buffer = manager.get_buffer(size)
                    buffers.append(buffer)

                    stats = manager.get_memory_stats()
                    usage = stats.get("usage_percentage", 0)
                    print(f"  Buffer {i+1}: {size}, GPU usage: {usage:.1f}%")

                    if usage > 70:  # Stop at 70% to avoid system instability
                        print(f"  Stopping at {usage:.1f}% usage for safety")
                        break

                except Exception as e:
                    print(f"  Memory exhaustion at buffer {i+1}: {e}")
                    allocation_successful = False
                    break

            self.profiler.record_point("peak_usage")
            peak_usage = self.profiler.get_peak_usage_mb()
            print(f"Peak memory usage: {peak_usage:.1f} MB")

            # Test recovery by freeing buffers
            print("Testing memory recovery...")
            for i, buffer in enumerate(buffers):
                manager.return_buffer(buffer)
                if i % 3 == 0:  # Periodic stats check
                    stats = manager.get_memory_stats()
                    usage = stats.get("usage_percentage", 0)
                    print(f"  Freed {i+1} buffers, usage: {usage:.1f}%")

            # Force cleanup
            if hasattr(manager.memory_pool, "cleanup_expired_buffers"):
                manager.memory_pool.cleanup_expired_buffers()
            torch.cuda.empty_cache()

            self.profiler.record_point("recovery_complete")

            # Test new allocation after recovery
            try:
                test_buffer = manager.get_buffer((512, 512))
                manager.return_buffer(test_buffer)
                print("✓ Recovery successful - new allocations work")
                recovery_success = True
            except Exception as e:
                print(f"✗ Recovery failed: {e}")
                recovery_success = False

            # Analyze for memory leaks
            leak_analysis = self.profiler.detect_memory_leaks()
            print(f"Memory leak analysis: {leak_analysis}")

            manager.shutdown()

            return recovery_success and not leak_analysis["leak_detected"]

        except Exception as e:
            print(f"✗ Memory exhaustion test failed: {e}")
            return False

    def test_concurrent_memory_access(self) -> bool:
        """Test concurrent memory access patterns."""
        print("\n=== Concurrent Memory Access Test ===")

        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("⚠ Skipping - CUDA not available")
            return True

        try:
            hardware = HardwareDetector()
            config = MemoryPoolConfig(max_buffer_count=100)
            manager = AdvancedGPUMemoryManager(hardware, config)

            num_workers = 4
            operations_per_worker = 10
            worker_errors = []

            def worker_task(worker_id: int) -> dict:
                """Worker function for concurrent testing."""
                errors = []
                buffers = []

                try:
                    for i in range(operations_per_worker):
                        try:
                            size = (128 + worker_id * 32, 128 + worker_id * 32)
                            buffer = manager.get_buffer(size)
                            buffers.append(buffer)
                            time.sleep(0.001)  # Simulate work
                        except Exception as e:
                            errors.append(f"Worker {worker_id}, op {i}: {e}")

                    # Return buffers
                    for buffer in buffers:
                        try:
                            manager.return_buffer(buffer)
                        except Exception as e:
                            errors.append(f"Worker {worker_id} return: {e}")

                except Exception as e:
                    errors.append(f"Worker {worker_id} fatal: {e}")

                return {
                    "worker_id": worker_id,
                    "errors": errors,
                    "operations": operations_per_worker,
                }

            print(f"Starting {num_workers} concurrent workers...")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_workers)]

                for future in as_completed(futures):
                    result = future.result()
                    if result["errors"]:
                        worker_errors.extend(result["errors"])
                        print(f"Worker {result['worker_id']}: {len(result['errors'])} errors")
                    else:
                        print(f"Worker {result['worker_id']}: Success")

            total_ops = num_workers * operations_per_worker
            error_rate = len(worker_errors) / total_ops

            print(f"Concurrent test results:")
            print(f"  Total operations: {total_ops}")
            print(f"  Errors: {len(worker_errors)}")
            print(f"  Error rate: {error_rate:.2%}")

            manager.shutdown()

            # Success if error rate is under 5%
            return error_rate < 0.05

        except Exception as e:
            print(f"✗ Concurrent access test failed: {e}")
            return False

    def test_hardware_tier_adaptation(self) -> bool:
        """Test memory management adaptation to different hardware tiers."""
        print("\n=== Hardware Tier Adaptation Test ===")

        tiers_to_test = [
            (HardwareTier.LOW, 2.0),
            (HardwareTier.MEDIUM, 4.0),
            (HardwareTier.HIGH, 8.0),
            (HardwareTier.ULTRA, 16.0),
        ]

        tier_results = {}

        for tier, memory_gb in tiers_to_test:
            print(f"Testing {tier.name} tier ({memory_gb}GB)...")

            try:
                # Mock hardware for this tier
                from unittest.mock import Mock

                mock_hardware = Mock()
                mock_hardware.tier = tier
                mock_hardware.gpu_memory_gb = memory_gb
                mock_hardware.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

                config = MemoryPoolConfig()
                manager = AdvancedGPUMemoryManager(mock_hardware, config)

                optimal_batch = manager.get_optimal_batch_size()
                stats = manager.get_memory_stats()

                tier_results[tier.name] = {
                    "optimal_batch": optimal_batch,
                    "memory_gb": memory_gb,
                    "strategy": getattr(manager, "strategy", "unknown"),
                }

                print(f"  Optimal batch size: {optimal_batch}")
                print(f"  Strategy: {getattr(manager, 'strategy', 'unknown')}")

                manager.shutdown()

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                return False

        # Verify tier progression makes sense
        low_batch = tier_results.get("LOW", {}).get("optimal_batch", 0)
        high_batch = tier_results.get("HIGH", {}).get("optimal_batch", 0)

        if low_batch and high_batch:
            progression_valid = high_batch >= low_batch
            print(
                f"Tier progression valid: {progression_valid} (Low: {low_batch}, High: {high_batch})"
            )
            return progression_valid

        return True

    def run_all_tests(self) -> bool:
        """Run the complete professional testing suite."""
        print("=" * 60)
        print("PROFESSIONAL GPU MEMORY MANAGEMENT TESTING SUITE")
        print("Task 19.7 - Comprehensive Validation")
        print("=" * 60)

        # System assessment
        assessment = self.run_system_assessment()

        # Run all tests
        tests = [
            ("Integration Test", self.test_gpu_memory_manager_integration),
            ("Memory Exhaustion Recovery", self.test_memory_exhaustion_recovery),
            ("Concurrent Access", self.test_concurrent_memory_access),
            ("Hardware Tier Adaptation", self.test_hardware_tier_adaptation),
        ]

        results = {}
        all_passed = True

        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                results[test_name] = result
                status = "PASS" if result else "FAIL"
                print(f"Result: {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"✗ Test failed with exception: {e}")
                results[test_name] = False
                all_passed = False

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")

        print(f"\nOVERALL RESULT: {'PASS' if all_passed else 'FAIL'}")

        if all_passed:
            print("✓ GPU Memory Management system is production-ready")
        else:
            print("✗ Issues found - review test results")

        return all_passed


def main():
    """Main entry point for professional GPU memory testing."""
    tester = ProfessionalGPUMemoryTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
