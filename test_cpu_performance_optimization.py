#!/usr/bin/env python3

"""
CPU Performance Optimization Testing for SpygateAI - Task 19.8
==============================================================

Comprehensive test suite for CPU-only performance optimization
focusing on YOLOv8 inference and video processing efficiency.

Tests CPU-specific optimizations:
- YOLOv8 CPU inference optimization 
- Multi-threading strategies
- Memory usage optimization
- Hardware-adaptive CPU settings
- Frame processing pipeline efficiency
"""

import gc
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import cv2
    import numpy as np
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    TORCH_AVAILABLE = False
    CV2_AVAILABLE = False

try:
    from spygate.ml.yolov8_model import EnhancedYOLOv8, SpygateYOLO
    from spygate.core.hardware_adaptive import HardwareTierDetector
    from spygate.core.performance_optimizer import PerformanceOptimizer
    SPYGATE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SpygateAI modules not available: {e}")
    SPYGATE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPUPerformanceOptimizer:
    """CPU-specific performance optimization and testing"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.optimal_threads = min(self.cpu_count, 8)  # Cap at 8 for most efficient usage
        self.test_results = {}
        
    def optimize_torch_cpu_settings(self):
        """Configure PyTorch for optimal CPU performance"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            # Set optimal CPU threads
            torch.set_num_threads(self.optimal_threads)
            torch.set_num_interop_threads(self.optimal_threads)
            
            # Enable CPU optimizations
            torch.backends.mkldnn.enabled = True
            torch.backends.mkldnn.benchmark = True
            
            # Disable GPU usage for CPU testing
            torch.cuda.set_device(-1) if torch.cuda.is_available() else None
            
            logger.info(f"✅ Torch CPU optimized: {self.optimal_threads} threads, MKL-DNN enabled")
            return True
            
        except Exception as e:
            logger.error(f"❌ Torch CPU optimization failed: {e}")
            return False
    
    def test_yolov8_cpu_performance(self) -> Dict:
        """Test YOLOv8 performance with CPU optimizations"""
        if not SPYGATE_AVAILABLE:
            return {"status": "skipped", "reason": "SpygateAI modules not available"}
            
        try:
            # Initialize YOLOv8 with CPU-specific optimizations
            model = EnhancedYOLOv8(device='cpu')
            
            # Create test images of different sizes
            test_images = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),    # Standard
                np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),  # HD
                np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8),    # Mobile
            ]
            
            results = {}
            
            for i, img in enumerate(test_images):
                size_name = ["640x640", "1080x1920", "480x854"][i]
                
                # Warm up
                for _ in range(3):
                    _ = model.predict(img)
                
                # Benchmark
                times = []
                for _ in range(10):
                    start_time = time.time()
                    detections = model.predict(img)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                results[size_name] = {
                    "avg_inference_time": round(avg_time, 4),
                    "fps": round(fps, 2),
                    "detections": len(detections) if detections else 0
                }
                
                logger.info(f"✅ YOLOv8 {size_name}: {avg_time:.4f}s ({fps:.2f} FPS)")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 CPU performance test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_multithreaded_processing(self) -> Dict:
        """Test multi-threaded video frame processing"""
        try:
            def process_frame(frame_data):
                """Simulate frame processing with CPU-intensive operations"""
                frame_id, frame = frame_data
                
                # Simulate typical frame processing operations
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else frame
                
                # CPU-intensive operations
                for _ in range(100):
                    _ = np.mean(gray) if CV2_AVAILABLE else np.mean(frame)
                
                return frame_id, time.time()
            
            # Create test frames
            test_frames = [
                (i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                for i in range(50)
            ]
            
            # Test different thread counts
            thread_counts = [1, 2, 4, 8, min(16, self.cpu_count)]
            results = {}
            
            for thread_count in thread_counts:
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    processed = list(executor.map(process_frame, test_frames))
                
                end_time = time.time()
                total_time = end_time - start_time
                fps = len(test_frames) / total_time
                
                results[f"{thread_count}_threads"] = {
                    "total_time": round(total_time, 3),
                    "fps": round(fps, 2),
                    "frames_processed": len(processed)
                }
                
                logger.info(f"✅ {thread_count} threads: {total_time:.3f}s ({fps:.2f} FPS)")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ Multi-threaded processing test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_memory_optimization(self) -> Dict:
        """Test memory usage optimization for CPU processing"""
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Test memory-intensive operations
            large_arrays = []
            
            # Create and process large arrays to test memory management
            for i in range(10):
                # Create large array
                array = np.random.random((1000, 1000, 3))
                large_arrays.append(array)
                
                # Process array
                processed = np.mean(array, axis=2)
                
                # Clear every few iterations to test garbage collection
                if i % 3 == 0:
                    large_arrays.clear()
                    gc.collect()
            
            # Final cleanup
            large_arrays.clear()
            gc.collect()
            
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_diff = final_memory - initial_memory
            
            result = {
                "initial_memory_mb": round(initial_memory, 2),
                "final_memory_mb": round(final_memory, 2),
                "memory_difference_mb": round(memory_diff, 2),
                "memory_efficiency": "good" if memory_diff < 100 else "needs_optimization"
            }
            
            logger.info(f"✅ Memory test: {initial_memory:.2f}MB → {final_memory:.2f}MB (Δ{memory_diff:+.2f}MB)")
            
            return {"status": "passed", "results": result}
            
        except ImportError:
            return {"status": "skipped", "reason": "psutil not available"}
        except Exception as e:
            logger.error(f"❌ Memory optimization test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_hardware_adaptive_cpu_settings(self) -> Dict:
        """Test hardware-adaptive settings for CPU optimization"""
        if not SPYGATE_AVAILABLE:
            return {"status": "skipped", "reason": "SpygateAI modules not available"}
            
        try:
            detector = HardwareTierDetector()
            tier = detector.detect_tier()
            
            # CPU-specific adaptive settings based on hardware tier
            cpu_settings = {
                "LOW": {
                    "threads": min(2, self.cpu_count),
                    "batch_size": 1,
                    "resolution_scale": 0.5,
                    "frame_skip": 4
                },
                "MEDIUM": {
                    "threads": min(4, self.cpu_count),
                    "batch_size": 2,
                    "resolution_scale": 0.7,
                    "frame_skip": 2
                },
                "HIGH": {
                    "threads": min(6, self.cpu_count),
                    "batch_size": 4,
                    "resolution_scale": 0.85,
                    "frame_skip": 1
                },
                "ULTRA": {
                    "threads": min(8, self.cpu_count),
                    "batch_size": 8,
                    "resolution_scale": 1.0,
                    "frame_skip": 1
                }
            }
            
            settings = cpu_settings.get(tier, cpu_settings["MEDIUM"])
            
            # Test these settings
            torch.set_num_threads(settings["threads"])
            
            # Create test with adaptive settings
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Scale resolution based on tier
            if settings["resolution_scale"] != 1.0:
                h, w = test_image.shape[:2]
                new_h = int(h * settings["resolution_scale"])
                new_w = int(w * settings["resolution_scale"])
                test_image = cv2.resize(test_image, (new_w, new_h)) if CV2_AVAILABLE else test_image
            
            # Time processing with adaptive settings
            start_time = time.time()
            for _ in range(settings["batch_size"]):
                # Simulate processing
                _ = np.mean(test_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            result = {
                "detected_tier": tier,
                "applied_settings": settings,
                "processing_time": round(processing_time, 4),
                "cpu_count": self.cpu_count
            }
            
            logger.info(f"✅ Hardware-adaptive CPU ({tier}): {settings['threads']} threads, {processing_time:.4f}s")
            
            return {"status": "passed", "results": result}
            
        except Exception as e:
            logger.error(f"❌ Hardware-adaptive CPU settings test failed: {e}")
            return {"status": "failed", "error": str(e)}

def run_cpu_performance_tests():
    """Run comprehensive CPU performance optimization tests"""
    print("🚀 Starting CPU Performance Optimization Tests - Task 19.8")
    print("=" * 60)
    
    optimizer = CPUPerformanceOptimizer()
    results = {}
    
    # System info
    print(f"💻 System: {optimizer.cpu_count} CPU cores")
    print(f"🔧 Optimal threads: {optimizer.optimal_threads}")
    print(f"📊 PyTorch available: {TORCH_AVAILABLE}")
    print(f"📷 OpenCV available: {CV2_AVAILABLE}")
    print(f"🎯 SpygateAI available: {SPYGATE_AVAILABLE}")
    print()
    
    tests = [
        ("CPU Settings Optimization", optimizer.optimize_torch_cpu_settings),
        ("YOLOv8 CPU Performance", optimizer.test_yolov8_cpu_performance),
        ("Multi-threaded Processing", optimizer.test_multithreaded_processing),
        ("Memory Optimization", optimizer.test_memory_optimization),
        ("Hardware-Adaptive CPU Settings", optimizer.test_hardware_adaptive_cpu_settings),
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
                for test_name, test_func in tests:
        print(f"🧪 Running {test_name}...")
        try:
            if callable(test_func):
                result = test_func()
                if not isinstance(result, dict):
                    result = {"status": "passed" if result else "failed"}
            else:
                result = {"status": "passed" if test_func else "failed"}
                
            results[test_name] = result
            
            if result.get("status") == "passed":
                passed += 1
                print(f"✅ {test_name} - PASSED")
            elif result.get("status") == "skipped":
                print(f"⏭️ {test_name} - SKIPPED ({result.get('reason', 'Unknown')})")
            else:
                print(f"❌ {test_name} - FAILED ({result.get('error', 'Unknown error')})")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = {"status": "error", "error": str(e)}
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("=" * 60)
    print(f"🎯 CPU Performance Optimization Test Results")
    print(f"📊 Tests passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print("=" * 60)
    
    return results, passed, total, total_time

if __name__ == "__main__":
    results, passed, total, execution_time = run_cpu_performance_tests()
    
    # Exit with appropriate code
    if passed == total:
        print("🎉 All CPU performance optimization tests completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️ {total - passed} test(s) failed or were skipped")
        sys.exit(1) 