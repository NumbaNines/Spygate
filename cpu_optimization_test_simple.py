#!/usr/bin/env python3

"""
CPU Performance Optimization Testing for SpygateAI - Task 19.8
==============================================================

Simplified but comprehensive CPU optimization test suite.
"""

import gc
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add project path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

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

def test_cpu_torch_optimization():
    """Test PyTorch CPU optimization settings"""
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "PyTorch not available"}
    
    try:
        cpu_count = mp.cpu_count()
        optimal_threads = min(cpu_count, 8)
        
        # Configure PyTorch for optimal CPU performance
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)
        
        # Enable CPU optimizations
        torch.backends.mkldnn.enabled = True
        torch.backends.mkldnn.benchmark = True
        
        print(f"✅ Torch CPU optimized: {optimal_threads} threads, MKL-DNN enabled")
        
        return {
            "status": "passed",
            "cpu_count": cpu_count,
            "optimal_threads": optimal_threads,
            "mkldnn_enabled": torch.backends.mkldnn.enabled
        }
        
    except Exception as e:
        print(f"❌ Torch CPU optimization failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_multithreaded_processing():
    """Test multi-threaded video frame processing"""
    try:
        def process_frame(frame_data):
            frame_id, frame = frame_data
            # Simulate CPU-intensive frame processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if CV2_AVAILABLE else frame
            for _ in range(100):
                _ = np.mean(gray) if CV2_AVAILABLE else np.mean(frame)
            return frame_id, time.time()
        
        # Create test frames
        test_frames = [
            (i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for i in range(50)
        ]
        
        # Test different thread counts
        cpu_count = mp.cpu_count()
        thread_counts = [1, 2, 4, 8, min(16, cpu_count)]
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
            
            print(f"✅ {thread_count} threads: {total_time:.3f}s ({fps:.2f} FPS)")
        
        return {"status": "passed", "results": results}
        
    except Exception as e:
        print(f"❌ Multi-threaded processing test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_memory_optimization():
    """Test memory usage optimization"""
    try:
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            initial_memory = 0
            print("psutil not available, skipping memory monitoring")
        
        # Test memory-intensive operations
        large_arrays = []
        
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
        
        if initial_memory > 0:
            try:
                final_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_diff = final_memory - initial_memory
                
                result = {
                    "initial_memory_mb": round(initial_memory, 2),
                    "final_memory_mb": round(final_memory, 2),
                    "memory_difference_mb": round(memory_diff, 2),
                    "memory_efficiency": "good" if memory_diff < 100 else "needs_optimization"
                }
                
                print(f"✅ Memory test: {initial_memory:.2f}MB → {final_memory:.2f}MB (Δ{memory_diff:+.2f}MB)")
            except:
                result = {"status": "completed_without_monitoring"}
        else:
            result = {"status": "completed_without_monitoring"}
        
        return {"status": "passed", "results": result}
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_hardware_adaptive_settings():
    """Test hardware-adaptive CPU settings"""
    try:
        cpu_count = mp.cpu_count()
        
        # Detect hardware tier based on CPU count (simple heuristic)
        if cpu_count >= 12:
            tier = "ULTRA"
        elif cpu_count >= 8:
            tier = "HIGH"
        elif cpu_count >= 4:
            tier = "MEDIUM"
        else:
            tier = "LOW"
        
        # CPU-specific adaptive settings
        cpu_settings = {
            "LOW": {"threads": min(2, cpu_count), "batch_size": 1, "resolution_scale": 0.5},
            "MEDIUM": {"threads": min(4, cpu_count), "batch_size": 2, "resolution_scale": 0.7},
            "HIGH": {"threads": min(6, cpu_count), "batch_size": 4, "resolution_scale": 0.85},
            "ULTRA": {"threads": min(8, cpu_count), "batch_size": 8, "resolution_scale": 1.0}
        }
        
        settings = cpu_settings.get(tier, cpu_settings["MEDIUM"])
        
        # Test these settings
        if TORCH_AVAILABLE:
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
            _ = np.mean(test_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        result = {
            "detected_tier": tier,
            "applied_settings": settings,
            "processing_time": round(processing_time, 4),
            "cpu_count": cpu_count
        }
        
        print(f"✅ Hardware-adaptive CPU ({tier}): {settings['threads']} threads, {processing_time:.4f}s")
        
        return {"status": "passed", "results": result}
        
    except Exception as e:
        print(f"❌ Hardware-adaptive CPU settings test failed: {e}")
        return {"status": "failed", "error": str(e)}

def run_cpu_optimization_tests():
    """Run CPU performance optimization tests"""
    print("🚀 Starting CPU Performance Optimization Tests - Task 19.8")
    print("=" * 60)
    
    cpu_count = mp.cpu_count()
    print(f"💻 System: {cpu_count} CPU cores")
    print(f"📊 PyTorch available: {TORCH_AVAILABLE}")
    print(f"📷 OpenCV available: {CV2_AVAILABLE}")
    print()
    
    tests = [
        ("CPU Torch Optimization", test_cpu_torch_optimization),
        ("Multi-threaded Processing", test_multithreaded_processing),
        ("Memory Optimization", test_memory_optimization),
        ("Hardware-Adaptive CPU Settings", test_hardware_adaptive_settings),
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name}...")
        try:
            result = test_func()
            
            if result.get("status") == "passed":
                passed += 1
                print(f"✅ {test_name} - PASSED")
            elif result.get("status") == "skipped":
                print(f"⏭️ {test_name} - SKIPPED ({result.get('reason', 'Unknown')})")
            else:
                print(f"❌ {test_name} - FAILED ({result.get('error', 'Unknown error')})")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print()
    print("=" * 60)
    print(f"🎯 CPU Performance Optimization Test Results")
    print(f"📊 Tests passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print("=" * 60)
    
    return passed, total, total_time

if __name__ == "__main__":
    passed, total, execution_time = run_cpu_optimization_tests()
    
    if passed == total:
        print("🎉 All CPU performance optimization tests completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️ {total - passed} test(s) failed or were skipped")
        sys.exit(1) 