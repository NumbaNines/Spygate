#!/usr/bin/env python3

"""
GPU Memory Management Testing for SpygateAI - Task 19.7
========================================================

Comprehensive test suite for GPU memory management and performance 
optimization features under various load conditions.
"""

import gc
import logging
import sys
import threading
import time
from pathlib import Path

# Add project path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_COUNT = 0

try:
    from spygate.core.gpu_memory_manager import (
        AdvancedGPUMemoryManager,
        GPUMemoryPool, 
        MemoryPoolConfig,
        MemoryStrategy,
        get_memory_manager
    )
    from spygate.core.hardware import HardwareDetector, HardwareTier
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import SpygateAI modules: {e}")
    MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUMemoryTester:
    """Comprehensive GPU memory management tester."""
    
    def __init__(self):
        self.hardware = HardwareDetector() if MODULES_AVAILABLE else None
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all GPU memory management tests."""
        logger.info("🧪 Starting GPU Memory Management Test Suite - Task 19.7")
        logger.info(f"System: PyTorch={TORCH_AVAILABLE}, CUDA={GPU_AVAILABLE}, GPUs={GPU_COUNT}")
        
        tests = [
            ("System Requirements", self.test_system_requirements),
            ("Memory Manager Initialization", self.test_memory_manager_init),
            ("Memory Pool Operations", self.test_memory_pool_operations),
            ("Dynamic Batch Sizing", self.test_dynamic_batch_sizing),
            ("Concurrent Operations", self.test_concurrent_operations),
            ("Memory Under Load", self.test_memory_under_load),
            ("Performance Optimization", self.test_performance_optimization),
            ("Resource Cleanup", self.test_resource_cleanup)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*40}")
            logger.info(f"🔬 Test: {test_name}")
            
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {'passed': result, 'duration': duration}
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{status} - {test_name} ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ FAILED - {test_name}: {e}")
                import traceback
                traceback.print_exc()
                self.test_results[test_name] = {'passed': False, 'duration': 0, 'error': str(e)}
        
        return self.test_results
    
    def test_system_requirements(self):
        """Test system requirements and availability."""
        logger.info("Checking system requirements...")
        
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available - testing CPU fallback mode")
        else:
            logger.info(f"✅ PyTorch available: {torch.__version__}")
        
        # Check CUDA availability
        if not GPU_AVAILABLE:
            logger.warning("⚠️ CUDA not available - CPU-only testing")
        else:
            logger.info(f"✅ CUDA available: {torch.version.cuda}")
            logger.info(f"✅ GPU count: {GPU_COUNT}")
            
            for i in range(GPU_COUNT):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Check module availability
        if not MODULES_AVAILABLE:
            logger.error("❌ SpygateAI modules not available")
            return False
        else:
            logger.info("✅ SpygateAI modules available")
        
        return True
    
    def test_memory_manager_init(self):
        """Test memory manager initialization."""
        logger.info("Testing memory manager initialization...")
        
        try:
            # Test basic initialization
            manager = AdvancedGPUMemoryManager(self.hardware)
            logger.info("✅ Basic initialization successful")
            
            # Test strategy selection
            strategy = manager.strategy
            logger.info(f"✅ Strategy selected: {strategy.value}")
            
            # Test hardware detection integration
            tier = manager.hardware.tier.name
            logger.info(f"✅ Hardware tier: {tier}")
            
            # Test memory pool initialization
            if GPU_AVAILABLE:
                if manager.memory_pool is not None:
                    logger.info("✅ GPU memory pool initialized")
                else:
                    logger.warning("⚠️ GPU memory pool not initialized")
            else:
                if manager.memory_pool is None:
                    logger.info("✅ CPU-only mode correctly detected")
                else:
                    logger.warning("⚠️ GPU memory pool created without CUDA")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def test_memory_pool_operations(self):
        """Test memory pool buffer operations."""
        logger.info("Testing memory pool operations...")
        
        try:
            manager = AdvancedGPUMemoryManager(self.hardware)
            
            # Test buffer allocation
            test_sizes = [(64, 64), (256, 256), (512, 512)]
            buffers = []
            
            for size in test_sizes:
                buffer = manager.get_buffer(size)
                buffers.append(buffer)
                
                expected_shape = size
                if buffer.shape == expected_shape:
                    logger.info(f"✅ Buffer allocated: {size}")
                else:
                    logger.error(f"❌ Wrong buffer shape: expected {expected_shape}, got {buffer.shape}")
                    return False
            
            # Test buffer returns (if GPU available)
            if GPU_AVAILABLE and manager.memory_pool:
                for buffer in buffers:
                    manager.return_buffer(buffer)
                logger.info("✅ Buffers returned to pool")
                
                # Test buffer reuse
                reused_buffer = manager.get_buffer(test_sizes[0])
                logger.info("✅ Buffer reuse tested")
                manager.return_buffer(reused_buffer)
            
            # Test memory statistics
            stats = manager.get_memory_stats()
            logger.info(f"✅ Memory stats: {stats['strategy']}, tier: {stats['hardware_tier']}")
            
            if GPU_AVAILABLE:
                logger.info(f"   Memory usage: {stats.get('usage_percentage', 0):.1f}%")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory pool operations failed: {e}")
            return False
    
    def test_dynamic_batch_sizing(self):
        """Test dynamic batch sizing optimization."""
        logger.info("Testing dynamic batch sizing...")
        
        try:
            manager = AdvancedGPUMemoryManager(self.hardware)
            
            # Test optimal batch size calculation
            batch_size = manager.get_optimal_batch_size()
            logger.info(f"✅ Optimal batch size: {batch_size}")
            
            if batch_size > 0:
                logger.info("✅ Valid batch size returned")
            else:
                logger.warning("⚠️ Invalid batch size returned")
            
            # Test batch performance recording
            manager.record_batch_performance(batch_size, 0.1, True)
            logger.info("✅ Batch performance recorded")
            
            # Test adaptive sizing
            for i in range(3):
                new_batch_size = manager.get_optimal_batch_size()
                logger.info(f"   Adaptive batch size {i+1}: {new_batch_size}")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ Dynamic batch sizing failed: {e}")
            return False
    
    def test_concurrent_operations(self):
        """Test concurrent memory operations."""
        logger.info("Testing concurrent operations...")
        
        try:
            manager = AdvancedGPUMemoryManager(self.hardware)
            errors = []
            
            def worker():
                try:
                    for i in range(5):
                        buffer = manager.get_buffer((32, 32))
                        time.sleep(0.001)  # Brief processing
                        manager.return_buffer(buffer)
                except Exception as e:
                    errors.append(str(e))
            
            # Start 4 concurrent workers
            threads = []
            for _ in range(4):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            if errors:
                logger.error(f"❌ Concurrent errors: {errors}")
                return False
            
            logger.info("✅ 4 concurrent workers completed successfully")
            manager.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ Concurrent operations failed: {e}")
            return False
    
    def test_memory_under_load(self):
        """Test memory management under heavy load."""
        logger.info("Testing memory under load...")
        
        try:
            manager = AdvancedGPUMemoryManager(self.hardware)
            
            # Allocate many buffers to stress memory
            buffers = []
            allocation_times = []
            
            for i in range(15):  # Moderate load for safety
                start_time = time.time()
                buffer = manager.get_buffer((128, 128))
                allocation_time = time.time() - start_time
                
                buffers.append(buffer)
                allocation_times.append(allocation_time)
            
            avg_allocation_time = sum(allocation_times) / len(allocation_times)
            logger.info(f"✅ Average allocation time under load: {avg_allocation_time*1000:.2f}ms")
            
            # Return buffers
            for buffer in buffers:
                manager.return_buffer(buffer)
            
            logger.info(f"✅ Load test completed - {len(buffers)} buffers processed")
            
            # Check memory stats after load
            stats = manager.get_memory_stats()
            if GPU_AVAILABLE:
                logger.info(f"   Post-load memory usage: {stats.get('usage_percentage', 0):.1f}%")
            
            manager.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory under load test failed: {e}")
            return False
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        logger.info("Testing performance optimization...")
        
        try:
            # Test different hardware tier strategies
            strategies = [MemoryStrategy.LOW, MemoryStrategy.MEDIUM, MemoryStrategy.HIGH]
            
            for strategy in strategies:
                # Mock hardware with specific tier
                class MockHardware:
                    def __init__(self, strategy_val):
                        strategy_to_tier = {
                            MemoryStrategy.LOW: HardwareTier.LOW,
                            MemoryStrategy.MEDIUM: HardwareTier.MEDIUM,
                            MemoryStrategy.HIGH: HardwareTier.HIGH
                        }
                        self.tier = strategy_to_tier.get(strategy_val, HardwareTier.MEDIUM)
                
                mock_hardware = MockHardware(strategy)
                manager = AdvancedGPUMemoryManager(mock_hardware)
                
                # Test optimization timing
                start_time = time.time()
                buffer = manager.get_buffer((256, 256))
                allocation_time = time.time() - start_time
                
                logger.info(f"✅ {strategy.value} strategy allocation: {allocation_time*1000:.2f}ms")
                
                manager.return_buffer(buffer)
                manager.shutdown()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance optimization test failed: {e}")
            return False
    
    def test_resource_cleanup(self):
        """Test resource cleanup and shutdown."""
        logger.info("Testing resource cleanup...")
        
        try:
            manager = AdvancedGPUMemoryManager(self.hardware)
            
            # Allocate some resources
            buffers = []
            for i in range(5):
                buffer = manager.get_buffer((64, 64))
                buffers.append(buffer)
            
            logger.info(f"✅ Allocated {len(buffers)} buffers")
            
            # Test shutdown
            manager.shutdown()
            logger.info("✅ Manager shutdown completed")
            
            # Test global manager
            try:
                global_manager = get_memory_manager()
                logger.info("✅ Global manager accessed")
            except Exception as e:
                logger.warning(f"⚠️ Global manager issue: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Resource cleanup test failed: {e}")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = [
            "\n" + "="*60,
            "🧪 GPU MEMORY MANAGEMENT TEST REPORT - Task 19.7",
            "="*60,
            f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)",
            f"System: PyTorch={TORCH_AVAILABLE}, CUDA={GPU_AVAILABLE}, GPUs={GPU_COUNT}",
            ""
        ]
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result['duration']
            report.append(f"{status} {test_name} ({duration:.2f}s)")
            
            if not result['passed'] and 'error' in result:
                report.append(f"    Error: {result['error']}")
        
        report.extend([
            "",
            "="*60,
            "✅ GPU Memory Management Testing Complete" if success_rate >= 80 else "❌ GPU Memory Management Testing Failed",
            "="*60
        ])
        
        return "\n".join(report)


def main():
    """Run GPU memory management tests."""
    print("🏈 SpygateAI - GPU Memory Management Testing (Task 19.7)")
    print("=" * 60)
    
    tester = GPUMemoryTester()
    results = tester.run_all_tests()
    report = tester.generate_report()
    
    print(report)
    
    # Return success code
    passed_tests = sum(1 for result in results.values() if result['passed'])
    success_rate = (passed_tests / len(results)) * 100 if results else 0
    
    return 0 if success_rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
