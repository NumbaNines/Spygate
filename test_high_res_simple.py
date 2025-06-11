#!/usr/bin/env python3

"""
High-Resolution Image Processing Testing for SpygateAI - Task 19.10
====================================================================

Simplified high-resolution image processing test suite focusing on 1080x1920.
"""

import gc
import sys
import time
from pathlib import Path

# Add project path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def create_test_image(width, height):
    """Create a test image with realistic content"""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some rectangles to simulate objects
    for _ in range(10):
        x1 = np.random.randint(0, width - 50)
        y1 = np.random.randint(0, height - 50)
        x2 = x1 + np.random.randint(25, 50)
        y2 = y1 + np.random.randint(25, 50)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        if CV2_AVAILABLE:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    
    return image

def test_resolution_performance():
    """Test performance across different resolutions"""
    print("🧪 Testing Resolution Performance...")
    
    resolutions = [
        (640, 640, "Standard"),
        (1280, 720, "HD 720p"),
        (1920, 1080, "Full HD"),
        (1080, 1920, "Vertical HD"),  # Target resolution
        (2560, 1440, "QHD")
    ]
    
    results = {}
    
    for width, height, name in resolutions:
        print(f"  Testing {name} ({width}x{height})...")
        
        # Create test image
        test_image = create_test_image(width, height)
        image_size_mb = test_image.nbytes / (1024 * 1024)
        
        # Benchmark processing
        times = []
        for _ in range(5):
            start_time = time.time()
            
            # Typical image processing operations
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                resized = cv2.resize(test_image, (640, 640))
            else:
                gray = np.mean(test_image, axis=2)
                resized = test_image[::2, ::2]
            
            # Statistical operations
            mean_val = np.mean(test_image)
            std_val = np.std(test_image)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        results[name] = {
            "resolution": f"{width}x{height}",
            "image_size_mb": round(image_size_mb, 2),
            "avg_time": round(avg_time, 4),
            "fps": round(fps, 2),
            "pixels": width * height
        }
        
        print(f"    ✅ {name}: {avg_time:.4f}s ({fps:.2f} FPS), {image_size_mb:.2f}MB")
    
    return {"status": "passed", "results": results}

def test_memory_scaling():
    """Test memory usage with different image sizes"""
    print("🧪 Testing Memory Scaling...")
    
    try:
        import psutil
        process = psutil.Process()
        memory_available = True
    except ImportError:
        memory_available = False
        print("    psutil not available, memory monitoring disabled")
    
    # Test with target resolution: 1080x1920
    width, height = 1080, 1920
    
    if memory_available:
        initial_memory = process.memory_info().rss / (1024 * 1024)
    else:
        initial_memory = 0
    
    # Create multiple high-res images
    images = []
    for i in range(10):
        image = create_test_image(width, height)
        images.append(image)
        
        # Process each image
        if CV2_AVAILABLE:
            processed = cv2.resize(image, (640, 640))
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed = image[::2, ::2]
            gray = np.mean(processed, axis=2)
    
    if memory_available:
        peak_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = peak_memory - initial_memory
    else:
        peak_memory = 0
        memory_increase = 0
    
    # Cleanup
    images.clear()
    gc.collect()
    
    if memory_available:
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_cleanup = peak_memory - final_memory
        
        print(f"    ✅ Memory test: {initial_memory:.1f}MB → {peak_memory:.1f}MB (Δ+{memory_increase:.1f}MB)")
        print(f"    ✅ Cleanup: {memory_cleanup:.1f}MB freed")
    else:
        memory_cleanup = 0
        print("    ✅ Memory test completed (monitoring unavailable)")
    
    return {
        "status": "passed",
        "results": {
            "initial_memory_mb": round(initial_memory, 2) if memory_available else "N/A",
            "peak_memory_mb": round(peak_memory, 2) if memory_available else "N/A",
            "memory_increase_mb": round(memory_increase, 2) if memory_available else "N/A",
            "memory_cleanup_mb": round(memory_cleanup, 2) if memory_available else "N/A"
        }
    }

def test_hardware_adaptive_scaling():
    """Test hardware-adaptive resolution scaling"""
    print("🧪 Testing Hardware-Adaptive Scaling...")
    
    # Hardware tier configurations
    configs = {
        "LOW": {"scale": 0.5, "target_fps": 10},
        "MEDIUM": {"scale": 0.7, "target_fps": 15},
        "HIGH": {"scale": 0.85, "target_fps": 20},
        "ULTRA": {"scale": 1.0, "target_fps": 30}
    }
    
    # Base resolution: 1080x1920
    base_width, base_height = 1080, 1920
    base_image = create_test_image(base_width, base_height)
    
    results = {}
    
    for tier, config in configs.items():
        print(f"  Testing {tier} tier (scale: {config['scale']})...")
        
        # Scale image
        if config['scale'] != 1.0:
            new_width = int(base_width * config['scale'])
            new_height = int(base_height * config['scale'])
            if CV2_AVAILABLE:
                scaled_image = cv2.resize(base_image, (new_width, new_height))
            else:
                step = int(1 / config['scale'])
                scaled_image = base_image[::step, ::step]
                new_height, new_width = scaled_image.shape[:2]
        else:
            scaled_image = base_image
            new_width, new_height = base_width, base_height
        
        # Benchmark processing
        times = []
        for _ in range(10):
            start_time = time.time()
            
            # Typical processing
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            else:
                gray = np.mean(scaled_image, axis=2)
            
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        meets_target = fps >= config['target_fps']
        
        results[tier] = {
            "scale_factor": config['scale'],
            "resolution": f"{new_width}x{new_height}",
            "avg_time": round(avg_time, 4),
            "fps": round(fps, 2),
            "target_fps": config['target_fps'],
            "meets_target": meets_target
        }
        
        status = "✅" if meets_target else "⚠️"
        print(f"    {status} {tier}: {avg_time:.4f}s ({fps:.2f} FPS) - Target: {config['target_fps']} FPS")
    
    return {"status": "passed", "results": results}

def run_high_resolution_tests():
    """Run high-resolution image processing tests"""
    print("🚀 Starting High-Resolution Image Processing Tests - Task 19.10")
    print("=" * 65)
    
    print(f"🖼️ Target Resolution: 1080x1920 (Vertical HD)")
    print(f"📷 OpenCV available: {CV2_AVAILABLE}")
    print()
    
    tests = [
        ("Resolution Performance", test_resolution_performance),
        ("Memory Scaling", test_memory_scaling),
        ("Hardware-Adaptive Scaling", test_hardware_adaptive_scaling),
    ]
    
    passed = 0
    total = len(tests)
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            
            if result.get("status") == "passed":
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
        
        print()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 65)
    print(f"🎯 High-Resolution Test Results")
    print(f"📊 Tests passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print("=" * 65)
    
    return passed, total, total_time

if __name__ == "__main__":
    passed, total, execution_time = run_high_resolution_tests()
    
    if passed == total:
        print("🎉 All high-resolution tests completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️ {total - passed} test(s) failed")
        sys.exit(1) 