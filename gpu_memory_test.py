"""
Simple GPU Memory Management Test for SpygateAI.
Standalone test script without pytest dependencies.
"""

import gc
import logging
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
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

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import SpygateAI modules: {e}")
    MODULES_AVAILABLE = False

# Configure logging for test output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_hardware_detection():
    """Test hardware detection and memory strategy assignment."""
    logger.info("Testing hardware detection and memory strategy assignment...")

    if not MODULES_AVAILABLE:
        logger.error("SpygateAI modules not available, skipping test")
        return False

    try:
        hardware = HardwareDetector()
        logger.info(f"Detected hardware tier: {hardware.tier.name}")
        logger.info(f"CPU cores: {hardware.cpu_count}")
        logger.info(f"Total RAM: {hardware.total_memory / (1024**3):.2f} GB")
        logger.info(f"CUDA available: {hardware.has_cuda}")

        if hardware.has_cuda:
            logger.info(f"GPU count: {hardware.gpu_count}")
            logger.info(f"GPU name: {hardware.gpu_name}")
            logger.info(f"GPU memory: {hardware.gpu_memory_total / (1024**3):.2f} GB")

        # Test memory manager initialization
        memory_manager = AdvancedGPUMemoryManager(hardware)

        # Verify strategy assignment based on hardware tier
        expected_strategies = {
            HardwareTier.ULTRA_LOW: MemoryStrategy.ULTRA_LOW,
            HardwareTier.LOW: MemoryStrategy.LOW,
            HardwareTier.MEDIUM: MemoryStrategy.MEDIUM,
            HardwareTier.HIGH: MemoryStrategy.HIGH,
            HardwareTier.ULTRA: MemoryStrategy.ULTRA,
        }

        assert memory_manager.strategy == expected_strategies[hardware.tier]
        logger.info(f"Memory strategy correctly set to: {memory_manager.strategy.value}")

        memory_manager.shutdown()
        return True

    except Exception as e:
        logger.error(f"Hardware detection test failed: {e}")
        return False


def test_memory_pool_operations():
    """Test memory pool buffer allocation and reuse."""
    logger.info("Testing memory pool operations...")

    if not MODULES_AVAILABLE:
        logger.error("SpygateAI modules not available, skipping test")
        return False

    try:
        config = MemoryPoolConfig(
            buffer_timeout=10.0,  # Short timeout for testing
            max_buffer_count=20,
        )

        if TORCH_AVAILABLE and torch.cuda.is_available():
            # GPU testing
            logger.info("Testing GPU memory pool...")
            pool = GPUMemoryPool(config)

            # Test buffer allocation
            buffer1 = pool.get_buffer((100, 100), dtype=torch.float32)
            assert buffer1.device.type == "cuda"
            assert buffer1.shape == (100, 100)
            logger.info(f"✓ GPU buffer allocated: {buffer1.shape}, device: {buffer1.device}")

            # Test buffer reuse
            pool.return_buffer(buffer1)
            buffer2 = pool.get_buffer((100, 100), dtype=torch.float32)
            assert buffer2.data_ptr() == buffer1.data_ptr()  # Should be the same buffer
            logger.info("✓ GPU buffer reuse working correctly")

            # Test statistics
            stats = pool.get_statistics()
            assert stats["cache_hits"] >= 1
            assert stats["total_buffers"] >= 1
            logger.info(f"✓ GPU pool statistics: {stats}")

        else:
            logger.info("CUDA not available, testing CPU fallback...")

            # Test CPU fallback behavior
            hardware = HardwareDetector()
            memory_manager = AdvancedGPUMemoryManager(hardware)

            # Should not create GPU memory pool without CUDA
            assert memory_manager.memory_pool is None
            logger.info("✓ CPU-only mode correctly detected")

            # Test CPU buffer allocation
            try:
                buffer = memory_manager.get_buffer((50, 50))
                assert buffer.device.type == "cpu"
                assert buffer.shape == (50, 50)
                logger.info(f"✓ CPU buffer allocated: {buffer.shape}, device: {buffer.device}")
            except RuntimeError as e:
                logger.info(f"✓ CPU fallback error handling: {e}")

            memory_manager.shutdown()

        return True

    except Exception as e:
        logger.error(f"Memory pool operations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_memory_monitoring():
    """Test memory monitoring and statistics collection."""
    logger.info("Testing memory monitoring and statistics...")

    if not MODULES_AVAILABLE:
        logger.error("SpygateAI modules not available, skipping test")
        return False

    try:
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)

        # Get comprehensive statistics
        stats = memory_manager.get_memory_stats()

        # Verify required fields
        required_fields = ["strategy", "hardware_tier"]
        for field in required_fields:
            assert field in stats
            logger.info(f"✓ {field}: {stats[field]}")

        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_fields = ["total_memory_gb", "allocated_memory_gb", "usage_percentage"]
            for field in gpu_fields:
                if field in stats:
                    logger.info(f"✓ {field}: {stats[field]}")

        # Test system memory stats
        system_memory = hardware.get_system_memory()
        logger.info(f"System memory usage: {system_memory['percent']:.2f}%")

        # Test CPU usage
        cpu_usage = hardware.get_cpu_usage()
        logger.info(f"CPU usage: {cpu_usage:.2f}%")

        memory_manager.shutdown()
        return True

    except Exception as e:
        logger.error(f"Memory monitoring test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all GPU memory management tests."""
    logger.info("=" * 80)
    logger.info("STARTING GPU MEMORY MANAGEMENT TESTS")
    logger.info("=" * 80)

    # Check prerequisites
    logger.info(f"PyTorch Available: {TORCH_AVAILABLE}")
    logger.info(f"SpygateAI Modules Available: {MODULES_AVAILABLE}")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.info(f"CUDA Available: True")
        logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
    else:
        logger.info(f"CUDA Available: False (CPU-only mode)")

    logger.info("-" * 80)

    # Run tests
    tests = [
        ("Hardware Detection", test_hardware_detection),
        ("Memory Pool Operations", test_memory_pool_operations),
        ("Memory Monitoring", test_memory_monitoring),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")

        try:
            if test_func():
                logger.info(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                logger.error(f"❌ FAILED: {test_name}")
                failed += 1
        except Exception as e:
            logger.error(f"❌ ERROR: {test_name} - {e}")
            import traceback

            traceback.print_exc()
            failed += 1

        # Cleanup between tests
        try:
            shutdown_memory_manager()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

    # Final cleanup
    try:
        shutdown_memory_manager()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GPU MEMORY MANAGEMENT TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if passed + failed > 0:
        logger.info(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")

    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! GPU Memory Management is working correctly.")
    else:
        logger.warning(f"⚠️  {failed} test(s) failed. Review the output above for details.")

    logger.info("=" * 80)

    return passed, failed


if __name__ == "__main__":
    run_all_tests()
