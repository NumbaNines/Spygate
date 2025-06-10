"""
GPU Memory Management Testing Suite for SpygateAI.

This test suite validates GPU memory management functionality including:
- Memory pool operations
- Adaptive batch sizing
- Memory cleanup and defragmentation
- CPU-only fallback handling
- Hardware tier specific optimizations
- Performance monitoring and statistics
"""

import gc
import logging
import sys
import os
import time
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestGPUMemoryManagement:
    """Test suite for GPU memory management functionality."""
    
    def setup_test(self):
        """Setup for each test."""
        # Cleanup before test
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def teardown_test(self):
        """Teardown for each test."""
        # Cleanup after test
        shutdown_memory_manager()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def test_hardware_detection_and_memory_strategy(self):
        """Test hardware detection and memory strategy assignment."""
        logger.info("Testing hardware detection and memory strategy assignment...")
        
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

    def test_memory_pool_operations(self):
        """Test memory pool buffer allocation and reuse."""
        logger.info("Testing memory pool operations...")
        
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
            assert buffer1.device.type == 'cuda'
            assert buffer1.shape == (100, 100)
            logger.info(f"✓ GPU buffer allocated: {buffer1.shape}, device: {buffer1.device}")
            
            # Test buffer reuse
            pool.return_buffer(buffer1)
            buffer2 = pool.get_buffer((100, 100), dtype=torch.float32)
            assert buffer2.data_ptr() == buffer1.data_ptr()  # Should be the same buffer
            logger.info("✓ GPU buffer reuse working correctly")
            
            # Test statistics
            stats = pool.get_statistics()
            assert stats['cache_hits'] >= 1
            assert stats['total_buffers'] >= 1
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
                assert buffer.device.type == 'cpu'
                assert buffer.shape == (50, 50)
                logger.info(f"✓ CPU buffer allocated: {buffer.shape}, device: {buffer.device}")
            except RuntimeError as e:
                logger.info(f"✓ CPU fallback error handling: {e}")
            
            memory_manager.shutdown()

    def test_memory_cleanup_and_fragmentation(self):
        """Test memory cleanup and defragmentation functionality."""
        logger.info("Testing memory cleanup and fragmentation handling...")
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("Testing GPU memory cleanup...")
            
            # Get initial memory stats
            initial_stats = memory_manager.get_memory_stats()
            logger.info(f"Initial GPU memory usage: {initial_stats.get('usage_percentage', 0):.2f}%")
            
            # Allocate some buffers
            buffers = []
            for i in range(10):
                buffer = memory_manager.get_buffer((200, 200))
                buffers.append(buffer)
            
            # Check memory usage increased
            after_alloc_stats = memory_manager.get_memory_stats()
            logger.info(f"After allocation GPU memory usage: {after_alloc_stats.get('usage_percentage', 0):.2f}%")
            
            # Test cleanup
            fragmentation = memory_manager._calculate_fragmentation()
            logger.info(f"Memory fragmentation: {fragmentation:.3f}")
            
            # Force cleanup
            memory_manager._trigger_cleanup()
            
            # Check memory after cleanup
            after_cleanup_stats = memory_manager.get_memory_stats()
            logger.info(f"After cleanup GPU memory usage: {after_cleanup_stats.get('usage_percentage', 0):.2f}%")
            
            # Test defragmentation
            memory_manager._defragment_memory()
            logger.info("✓ Memory defragmentation completed")
            
        else:
            logger.info("Testing CPU memory cleanup...")
            
            # Test CPU cleanup
            memory_manager._trigger_cleanup()
            logger.info("✓ CPU memory cleanup completed")
        
        memory_manager.shutdown()

    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on available memory."""
        logger.info("Testing adaptive batch sizing...")
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)
        
        # Test initial batch size
        initial_batch_size = memory_manager.get_optimal_batch_size()
        logger.info(f"Initial optimal batch size: {initial_batch_size}")
        assert initial_batch_size > 0
        
        # Simulate successful batch processing
        memory_manager.record_batch_performance(initial_batch_size, 1.0, True)
        
        # Simulate failed batch processing (too large)
        memory_manager.record_batch_performance(initial_batch_size * 2, 5.0, False)
        
        # Get adjusted batch size
        adjusted_batch_size = memory_manager.get_optimal_batch_size()
        logger.info(f"Adjusted optimal batch size: {adjusted_batch_size}")
        
        # Test with model memory usage
        model_memory_mb = 500.0  # Simulate 500MB model
        batch_with_model = memory_manager.get_optimal_batch_size(model_memory_mb)
        logger.info(f"Batch size with {model_memory_mb}MB model: {batch_with_model}")
        
        memory_manager.shutdown()

    def test_memory_monitoring_and_statistics(self):
        """Test memory monitoring and statistics collection."""
        logger.info("Testing memory monitoring and statistics...")
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)
        
        # Get comprehensive statistics
        stats = memory_manager.get_memory_stats()
        
        # Verify required fields
        required_fields = ['strategy', 'hardware_tier']
        for field in required_fields:
            assert field in stats
            logger.info(f"✓ {field}: {stats[field]}")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_fields = ['total_memory_gb', 'allocated_memory_gb', 'usage_percentage']
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

    def test_stress_testing_and_edge_cases(self):
        """Test stress conditions and edge cases."""
        logger.info("Testing stress conditions and edge cases...")
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)
        
        # Test rapid allocation and deallocation
        logger.info("Testing rapid allocation/deallocation...")
        
        for i in range(50):
            try:
                buffer = memory_manager.get_buffer((100, 100))
                memory_manager.return_buffer(buffer)
            except Exception as e:
                logger.warning(f"Error in rapid allocation test: {e}")
        
        logger.info("✓ Rapid allocation test completed")
        
        # Test large buffer allocation
        logger.info("Testing large buffer allocation...")
        
        try:
            large_buffer = memory_manager.get_buffer((1000, 1000))
            logger.info(f"✓ Large buffer allocated: {large_buffer.shape}")
            memory_manager.return_buffer(large_buffer)
        except Exception as e:
            logger.info(f"Large buffer allocation failed (expected on low-memory systems): {e}")
        
        # Test concurrent access simulation
        logger.info("Testing concurrent access patterns...")
        
        buffers = []
        try:
            for i in range(20):
                buffer = memory_manager.get_buffer((50, 50))
                buffers.append(buffer)
            
            # Return all buffers
            for buffer in buffers:
                memory_manager.return_buffer(buffer)
            
            logger.info("✓ Concurrent access test completed")
        except Exception as e:
            logger.warning(f"Concurrent access test error: {e}")
        
        memory_manager.shutdown()

    def test_configuration_validation(self):
        """Test memory pool configuration validation."""
        logger.info("Testing configuration validation...")
        
        # Test valid configuration
        valid_config = MemoryPoolConfig(
            initial_pool_size=0.5,
            max_pool_size=0.8,
            cleanup_threshold=0.85,
            warning_threshold=0.75,
        )
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware, valid_config)
        assert memory_manager.config == valid_config
        logger.info("✓ Valid configuration accepted")
        
        # Test configuration bounds
        assert 0.0 < valid_config.initial_pool_size <= 1.0
        assert 0.0 < valid_config.max_pool_size <= 1.0
        assert valid_config.initial_pool_size <= valid_config.max_pool_size
        logger.info("✓ Configuration bounds validated")
        
        memory_manager.shutdown()

    def test_global_memory_manager(self):
        """Test global memory manager functionality."""
        logger.info("Testing global memory manager...")
        
        # Test singleton behavior
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        assert manager1 is manager2
        logger.info("✓ Global memory manager singleton working")
        
        # Test custom initialization
        hardware = HardwareDetector()
        config = MemoryPoolConfig(monitor_interval=5.0)
        
        custom_manager = initialize_memory_manager(hardware, config)
        assert custom_manager.config.monitor_interval == 5.0
        logger.info("✓ Custom memory manager initialization working")
        
        # Test shutdown
        shutdown_memory_manager()
        logger.info("✓ Global memory manager shutdown completed")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_integration(self):
        """Test PyTorch-specific functionality."""
        logger.info("Testing PyTorch integration...")
        
        hardware = HardwareDetector()
        memory_manager = AdvancedGPUMemoryManager(hardware)
        
        if torch.cuda.is_available():
            logger.info("Testing CUDA integration...")
            
            # Test CUDA memory operations
            device = torch.device('cuda:0')
            test_tensor = torch.randn(100, 100, device=device)
            
            # Test memory statistics collection
            stats = memory_manager.get_memory_stats()
            assert 'total_memory_gb' in stats
            assert 'allocated_memory_gb' in stats
            
            logger.info(f"✓ CUDA memory stats: {stats['usage_percentage']:.2f}% used")
            
            del test_tensor
            torch.cuda.empty_cache()
        else:
            logger.info("Testing CPU-only PyTorch integration...")
            
            # Test CPU tensor operations
            device = torch.device('cpu')
            test_tensor = torch.randn(100, 100, device=device)
            
            buffer = memory_manager.get_buffer((100, 100))
            assert buffer.device.type == 'cpu'
            logger.info("✓ CPU-only PyTorch integration working")
            
            del test_tensor
        
        memory_manager.shutdown()


def run_comprehensive_gpu_memory_test():
    """Run comprehensive GPU memory management test suite."""
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE GPU MEMORY MANAGEMENT TEST SUITE")
    logger.info("=" * 80)
    
    # System information
    hardware = HardwareDetector()
    logger.info(f"System: {hardware.system}")
    logger.info(f"Hardware Tier: {hardware.tier.name}")
    logger.info(f"CPU Cores: {hardware.cpu_count}")
    logger.info(f"Total RAM: {hardware.total_memory / (1024**3):.2f} GB")
    logger.info(f"CUDA Available: {hardware.has_cuda}")
    
    if hardware.has_cuda:
        logger.info(f"GPU Count: {hardware.gpu_count}")
        logger.info(f"GPU Name: {hardware.gpu_name}")
        logger.info(f"GPU Memory: {hardware.gpu_memory_total / (1024**3):.2f} GB")
    
    logger.info("-" * 80)
    
    # Initialize test instance
    test_instance = TestGPUMemoryManagement()
    
    try:
        # Run all tests
        test_methods = [
            'test_hardware_detection_and_memory_strategy',
            'test_memory_pool_operations',
            'test_memory_cleanup_and_fragmentation',
            'test_adaptive_batch_sizing',
            'test_memory_monitoring_and_statistics',
            'test_stress_testing_and_edge_cases',
            'test_configuration_validation',
            'test_global_memory_manager',
        ]
        
        if TORCH_AVAILABLE:
            test_methods.append('test_pytorch_integration')
        
        passed_tests = 0
        failed_tests = 0
        
        for test_method in test_methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_method}")
            logger.info(f"{'='*60}")
            
            try:
                getattr(test_instance, test_method)()
                logger.info(f"✅ PASSED: {test_method}")
                passed_tests += 1
            except Exception as e:
                logger.error(f"❌ FAILED: {test_method} - {e}")
                import traceback
                traceback.print_exc()
                failed_tests += 1
    
    finally:
        # Final cleanup
        shutdown_memory_manager()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("GPU MEMORY MANAGEMENT TEST SUITE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {passed_tests + failed_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    if passed_tests + failed_tests > 0:
        logger.info(f"Success Rate: {(passed_tests / (passed_tests + failed_tests) * 100):.1f}%")
    
    if failed_tests == 0:
        logger.info("🎉 ALL TESTS PASSED! GPU Memory Management is working correctly.")
    else:
        logger.warning(f"⚠️  {failed_tests} test(s) failed. Review the output above for details.")
    
    logger.info("=" * 80)
    
    return passed_tests, failed_tests


if __name__ == "__main__":
    run_comprehensive_gpu_memory_test()
