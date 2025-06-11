#!/usr/bin/env python3

"""
High-Resolution Image Processing Testing for SpygateAI - Task 19.10
====================================================================

Comprehensive test suite for high-resolution image processing performance
focusing on 1080x1920 resolution images across different processing scenarios.

Tests include:
- YOLOv8 detection on high-res images
- Frame processing pipeline efficiency
- Memory usage with large images
- Hardware-adaptive scaling
- Performance comparison across resolutions
"""

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import cv2
    import numpy as np
    import torch
    TORCH_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    TORCH_AVAILABLE = False
    CV2_AVAILABLE = False

try:
    from spygate.ml.yolov8_model import EnhancedYOLOv8, SpygateYOLO
    SPYGATE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SpygateAI modules not available: {e}")
    SPYGATE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HighResolutionTester:
    """High-resolution image processing performance tester"""
    
    def __init__(self):
        self.test_resolutions = [
            (640, 640, "Standard"),      # Standard YOLOv8
            (1280, 720, "HD 720p"),      # HD Ready
            (1920, 1080, "Full HD"),     # Full HD
            (1080, 1920, "Vertical HD"), # Mobile/Vertical HD (target)
            (2560, 1440, "QHD"),         # Quad HD
        ]
        self.results = {}
        
    def create_test_image(self, width: int, height: int) -> np.ndarray:
        """Create a test image with random content"""
        try:
            # Create realistic test image with varied content
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            # Add some rectangles to simulate objects
            for _ in range(10):
                x1 = np.random.randint(0, width - 100)
                y1 = np.random.randint(0, height - 100)
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(50, 100)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                if CV2_AVAILABLE:
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            
            return image
        except Exception as e:
            logger.error(f"Error creating test image: {e}")
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def test_yolov8_high_res_performance(self) -> Dict:
        """Test YOLOv8 performance with high-resolution images"""
        if not SPYGATE_AVAILABLE:
            return {"status": "skipped", "reason": "SpygateAI modules not available"}
        
        try:
            logger.info("Testing YOLOv8 performance with high-resolution images...")
            
                         # Initialize YOLOv8 model
             model = EnhancedYOLOv8(device='cpu')
             results = {}
             
             for width, height, name in self.test_resolutions:
                 logger.info(f"Testing {name} ({width}x{height})...")
                
                # Create test image
                test_image = self.create_test_image(width, height)
                image_size_mb = (test_image.nbytes) / (1024 * 1024)
                
                # Warm up
                for _ in range(2):
                    _ = model.predict(test_image)
                
                # Benchmark detection times
                times = []
                detection_counts = []
                
                for i in range(5):
                    start_time = time.time()
                    detections = model.predict(test_image)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    detection_counts.append(len(detections) if detections else 0)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                avg_detections = np.mean(detection_counts)
                
                results[name] = {
                    "resolution": f"{width}x{height}",
                    "image_size_mb": round(image_size_mb, 2),
                    "avg_inference_time": round(avg_time, 4),
                    "fps": round(fps, 2),
                    "avg_detections": round(avg_detections, 1),
                    "memory_efficiency": "good" if avg_time < 2.0 else "needs_optimization"
                }
                
                logger.info(f"✅ {name}: {avg_time:.4f}s ({fps:.2f} FPS), {avg_detections:.1f} detections, {image_size_mb:.2f}MB")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ YOLOv8 high-res performance test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_frame_processing_pipeline(self) -> Dict:
        """Test frame processing pipeline with high-resolution images"""
        try:
            logger.info("Testing frame processing pipeline performance...")
            
            results = {}
            
            for width, height, name in self.test_resolutions:
                logger.info(f"Testing pipeline {name} ({width}x{height})...")
                
                # Create test image
                test_image = self.create_test_image(width, height)
                
                # Simulate typical frame processing operations
                times = []
                
                for _ in range(10):
                    start_time = time.time()
                    
                    # Typical processing steps
                    if CV2_AVAILABLE:
                        # Convert to grayscale
                        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
                        
                        # Apply blur (common preprocessing)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        
                        # Edge detection
                        edges = cv2.Canny(blurred, 50, 150)
                        
                        # Resize (common operation)
                        resized = cv2.resize(test_image, (640, 640))
                    else:
                        # Fallback operations without OpenCV
                        gray = np.mean(test_image, axis=2)
                        resized = test_image[::2, ::2]  # Simple downsampling
                    
                    # Statistical operations
                    mean_val = np.mean(test_image)
                    std_val = np.std(test_image)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                results[name] = {
                    "resolution": f"{width}x{height}",
                    "avg_processing_time": round(avg_time, 4),
                    "fps": round(fps, 2),
                    "pixels": width * height,
                    "processing_efficiency": "good" if avg_time < 0.5 else "needs_optimization"
                }
                
                logger.info(f"✅ {name} pipeline: {avg_time:.4f}s ({fps:.2f} FPS)")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ Frame processing pipeline test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_memory_usage_scaling(self) -> Dict:
        """Test memory usage with different image resolutions"""
        try:
            logger.info("Testing memory usage scaling with resolution...")
            
            try:
                import psutil
                process = psutil.Process()
                memory_available = True
            except ImportError:
                memory_available = False
                logger.warning("psutil not available, memory monitoring disabled")
            
            results = {}
            
            for width, height, name in self.test_resolutions:
                logger.info(f"Testing memory usage {name} ({width}x{height})...")
                
                if memory_available:
                    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
                else:
                    initial_memory = 0
                
                # Create and process multiple high-res images
                images = []
                processing_times = []
                
                for i in range(5):
                    start_time = time.time()
                    
                    # Create image
                    image = self.create_test_image(width, height)
                    images.append(image)
                    
                    # Process image
                    if CV2_AVAILABLE:
                        processed = cv2.resize(image, (640, 640))
                        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    else:
                        processed = image[::2, ::2]  # Simple downsampling
                        gray = np.mean(processed, axis=2)
                    
                    end_time = time.time()
                    processing_times.append(end_time - start_time)
                
                if memory_available:
                    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_increase = peak_memory - initial_memory
                else:
                    peak_memory = 0
                    memory_increase = 0
                
                # Cleanup
                images.clear()
                gc.collect()
                
                if memory_available:
                    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_cleanup = peak_memory - final_memory
                else:
                    final_memory = 0
                    memory_cleanup = 0
                
                avg_processing_time = np.mean(processing_times)
                
                results[name] = {
                    "resolution": f"{width}x{height}",
                    "pixels": width * height,
                    "avg_processing_time": round(avg_processing_time, 4),
                    "initial_memory_mb": round(initial_memory, 2) if memory_available else "N/A",
                    "peak_memory_mb": round(peak_memory, 2) if memory_available else "N/A",
                    "memory_increase_mb": round(memory_increase, 2) if memory_available else "N/A",
                    "memory_cleanup_mb": round(memory_cleanup, 2) if memory_available else "N/A",
                    "memory_efficiency": "good" if memory_increase < 200 else "needs_optimization"
                }
                
                                 if memory_available:
                     logger.info(f"✅ {name}: {avg_processing_time:.4f}s, Memory: {initial_memory:.1f}MB → {peak_memory:.1f}MB (Δ+{memory_increase:.1f}MB)")
                 else:
                     logger.info(f"✅ {name}: {avg_processing_time:.4f}s (memory monitoring unavailable)")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ Memory usage scaling test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def test_hardware_adaptive_scaling(self) -> Dict:
        """Test hardware-adaptive resolution scaling"""
        try:
            logger.info("Testing hardware-adaptive resolution scaling...")
            
            # Simulate different hardware tiers
            hardware_configs = {
                "LOW": {"scale": 0.5, "target_fps": 10},
                "MEDIUM": {"scale": 0.7, "target_fps": 15},
                "HIGH": {"scale": 0.85, "target_fps": 20},
                "ULTRA": {"scale": 1.0, "target_fps": 30}
            }
            
            # Test with 1080x1920 (target resolution)
            base_width, base_height = 1080, 1920
            test_image = self.create_test_image(base_width, base_height)
            
            results = {}
            
            for tier, config in hardware_configs.items():
                logger.info(f"Testing {tier} tier (scale: {config['scale']})...")
                
                # Scale image according to tier
                if config['scale'] != 1.0:
                    new_width = int(base_width * config['scale'])
                    new_height = int(base_height * config['scale'])
                    if CV2_AVAILABLE:
                        scaled_image = cv2.resize(test_image, (new_width, new_height))
                    else:
                        # Simple scaling without OpenCV
                        step_w = int(1 / config['scale'])
                        step_h = int(1 / config['scale'])
                        scaled_image = test_image[::step_h, ::step_w]
                        new_height, new_width = scaled_image.shape[:2]
                else:
                    scaled_image = test_image
                    new_width, new_height = base_width, base_height
                
                # Benchmark processing with scaled image
                times = []
                for _ in range(10):
                    start_time = time.time()
                    
                    # Simulate typical processing
                    if CV2_AVAILABLE:
                        gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    else:
                        gray = np.mean(scaled_image, axis=2)
                        blurred = gray  # Skip blur without OpenCV
                    
                    # Statistical operations
                    _ = np.mean(blurred)
                    _ = np.std(blurred)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                meets_target = fps >= config['target_fps']
                
                results[tier] = {
                    "scale_factor": config['scale'],
                    "scaled_resolution": f"{new_width}x{new_height}",
                    "pixels": new_width * new_height,
                    "avg_processing_time": round(avg_time, 4),
                    "fps": round(fps, 2),
                    "target_fps": config['target_fps'],
                    "meets_target": meets_target,
                    "performance": "excellent" if fps > config['target_fps'] * 1.5 else "good" if meets_target else "needs_optimization"
                }
                
                status = "✅" if meets_target else "⚠️"
                logger.info(f"{status} {tier} tier: {avg_time:.4f}s ({fps:.2f} FPS) - Target: {config['target_fps']} FPS")
            
            return {"status": "passed", "results": results}
            
        except Exception as e:
            logger.error(f"❌ Hardware-adaptive scaling test failed: {e}")
            return {"status": "failed", "error": str(e)}

def run_high_resolution_tests():
    """Run comprehensive high-resolution image processing tests"""
    print("🚀 Starting High-Resolution Image Processing Tests - Task 19.10")
    print("=" * 70)
    
    tester = HighResolutionTester()
    
    print(f"🖼️ Target Resolution: 1080x1920 (Vertical HD)")
    print(f"📊 PyTorch available: {TORCH_AVAILABLE}")
    print(f"📷 OpenCV available: {CV2_AVAILABLE}")
    print(f"🎯 SpygateAI available: {SPYGATE_AVAILABLE}")
    print()
    
    tests = [
        ("YOLOv8 High-Resolution Performance", tester.test_yolov8_high_res_performance),
        ("Frame Processing Pipeline", tester.test_frame_processing_pipeline),
        ("Memory Usage Scaling", tester.test_memory_usage_scaling),
        ("Hardware-Adaptive Scaling", tester.test_hardware_adaptive_scaling),
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
        
        print()  # Add spacing between tests
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 70)
    print(f"🎯 High-Resolution Image Processing Test Results")
    print(f"📊 Tests passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print("=" * 70)
    
    return passed, total, total_time

if __name__ == "__main__":
    passed, total, execution_time = run_high_resolution_tests()
    
    if passed == total:
        print("🎉 All high-resolution image processing tests completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️ {total - passed} test(s) failed or were skipped")
        sys.exit(1) 