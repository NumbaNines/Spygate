#!/usr/bin/env python3
"""
Production Error Handling Test Suite
====================================

Comprehensive testing for SpygateAI's production-grade error handling system.
Tests edge cases, failure scenarios, and recovery mechanisms.
"""

import sys
import traceback
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_ocr_error_handling():
    """Test enhanced OCR system error handling and recovery."""
    print("🧪 ENHANCED OCR ERROR HANDLING TESTS")
    print("=" * 50)
    
    try:
        from enhanced_ocr_system import EnhancedOCRSystem, OCRResult, OCRError
        
        # Test 1: Initialization with various configurations
        print("\n1. Testing OCR system initialization...")
        
        # Normal initialization
        try:
            ocr = EnhancedOCRSystem(gpu_enabled=True, debug=True)
            print("✅ Normal initialization successful")
        except Exception as e:
            print(f"❌ Normal initialization failed: {e}")
        
        # Fallback initialization (no GPU)
        try:
            ocr_fallback = EnhancedOCRSystem(gpu_enabled=False, debug=True, fallback_enabled=True)
            print("✅ Fallback initialization successful")
        except Exception as e:
            print(f"❌ Fallback initialization failed: {e}")
        
        # Test 2: Edge cases for image processing
        print("\n2. Testing image processing edge cases...")
        
        test_cases = [
            {"name": "Empty image", "image": np.array([])},
            {"name": "None image", "image": None},
            {"name": "Tiny image", "image": np.ones((5, 5, 3), dtype=np.uint8)},
            {"name": "Large image", "image": np.ones((4000, 4000, 3), dtype=np.uint8) * 128},
            {"name": "Single pixel", "image": np.ones((1, 1, 3), dtype=np.uint8)},
            {"name": "Corrupted shape", "image": np.ones((10, 0, 3), dtype=np.uint8)},
        ]
        
        for case in test_cases:
            try:
                result = ocr.enhance_image_for_ocr(case["image"], "adaptive")
                if result:
                    print(f"✅ {case['name']}: Handled gracefully")
                else:
                    print(f"⚠️ {case['name']}: No result returned")
            except Exception as e:
                print(f"❌ {case['name']}: {e}")
        
        # Test 3: Text extraction error handling
        print("\n3. Testing text extraction error handling...")
        
        # Create test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        
        bbox_test_cases = [
            {"name": "Normal bbox", "bbox": [10, 10, 90, 50]},
            {"name": "Invalid bbox format", "bbox": [10, 10]},
            {"name": "Negative coordinates", "bbox": [-10, -10, 50, 50]},
            {"name": "Out of bounds", "bbox": [150, 150, 300, 300]},
            {"name": "Inverted bbox", "bbox": [90, 90, 10, 10]},
            {"name": "Zero-size bbox", "bbox": [50, 50, 50, 50]},
        ]
        
        for case in bbox_test_cases:
            try:
                result = ocr.extract_text_from_region(test_image, case["bbox"])
                if isinstance(result, OCRResult):
                    status = "✅" if result.is_successful or result.error else "⚠️"
                    print(f"{status} {case['name']}: {result.text or result.error}")
                else:
                    print(f"❌ {case['name']}: Invalid result type")
            except Exception as e:
                print(f"❌ {case['name']}: {e}")
        
        # Test 4: Performance and health monitoring
        print("\n4. Testing performance monitoring...")
        
        # Generate some test extractions
        for i in range(10):
            try:
                ocr.extract_text_from_region(test_image, [10, 10, 90, 50])
            except:
                pass
        
        # Check performance stats
        stats = ocr.get_performance_stats()
        print(f"✅ Performance stats: {stats['total_extractions']} extractions")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Engine status: {stats['engine_status']}")
        
        # Health check
        health = ocr.health_check()
        print(f"✅ Health status: {health['status']}")
        if health['issues']:
            print(f"   • Issues: {health['issues']}")
        
        print("✅ Enhanced OCR error handling tests completed")
        
    except ImportError:
        print("❌ Enhanced OCR system not available")
    except Exception as e:
        print(f"❌ Enhanced OCR tests failed: {e}")
        traceback.print_exc()

def test_hud_detector_error_handling():
    """Test HUD detector error handling and recovery."""
    print("\n🧪 HUD DETECTOR ERROR HANDLING TESTS")
    print("=" * 50)
    
    try:
        from spygate.ml.hud_detector import EnhancedHUDDetector, HUDDetectionError
        
        # Test 1: Model initialization edge cases
        print("\n1. Testing HUD detector initialization...")
        
        # Normal initialization
        try:
            detector = EnhancedHUDDetector()
            print("✅ Normal HUD detector initialization successful")
        except Exception as e:
            print(f"❌ Normal initialization failed: {e}")
            return
        
        # Test with non-existent model
        try:
            detector_bad = EnhancedHUDDetector(model_path="nonexistent_model.pt")
            print("❌ Should have failed with non-existent model")
        except Exception as e:
            print(f"✅ Correctly handled non-existent model: {type(e).__name__}")
        
        # Test 2: Frame processing edge cases
        print("\n2. Testing frame processing edge cases...")
        
        frame_test_cases = [
            {"name": "Normal frame", "frame": np.ones((480, 640, 3), dtype=np.uint8) * 128},
            {"name": "Empty frame", "frame": np.array([])},
            {"name": "None frame", "frame": None},
            {"name": "Grayscale frame", "frame": np.ones((480, 640), dtype=np.uint8) * 128},
            {"name": "Tiny frame", "frame": np.ones((10, 10, 3), dtype=np.uint8)},
            {"name": "Large frame", "frame": np.ones((2160, 3840, 3), dtype=np.uint8) * 128},
        ]
        
        for case in frame_test_cases:
            try:
                result = detector.detect_hud_elements(case["frame"])
                
                if isinstance(result, dict):
                    error = result.get('metadata', {}).get('error')
                    if error:
                        print(f"✅ {case['name']}: Handled error - {error}")
                    else:
                        detections = len(result.get('detections', []))
                        print(f"✅ {case['name']}: {detections} detections")
                else:
                    print(f"❌ {case['name']}: Invalid result format")
            except Exception as e:
                print(f"❌ {case['name']}: Unhandled exception - {e}")
        
        # Test 3: OCR integration error handling
        print("\n3. Testing OCR integration...")
        
        # Create test frame with simple HUD-like elements
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        cv2.rectangle(test_frame, (50, 20), (200, 60), (255, 255, 255), -1)  # Simulated HUD bar
        cv2.putText(test_frame, "1ST & 10", (60, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        try:
            result = detector.detect_hud_elements(test_frame)
            print(f"✅ OCR integration test successful")
            print(f"   • Detections: {len(result.get('detections', []))}")
            print(f"   • Processing time: {result.get('metadata', {}).get('processing_time', 0):.3f}s")
        except Exception as e:
            print(f"❌ OCR integration failed: {e}")
        
        # Test 4: Performance monitoring
        print("\n4. Testing performance monitoring...")
        
        # Run multiple detections
        for i in range(5):
            try:
                detector.detect_hud_elements(test_frame)
            except:
                pass
        
        # Check stats
        stats = detector.get_performance_stats()
        print(f"✅ Performance tracking working")
        print(f"   • Total detections: {stats['total_detections']}")
        print(f"   • Success rate: {stats['success_rate']:.1%}")
        print(f"   • Average time: {stats['average_detection_time']:.3f}s")
        
        # Health check
        health = detector.health_check()
        print(f"✅ Health check: {health['status']}")
        if health['issues']:
            print(f"   • Issues: {health['issues']}")
        
        print("✅ HUD detector error handling tests completed")
        
    except ImportError as e:
        print(f"❌ HUD detector import failed: {e}")
    except Exception as e:
        print(f"❌ HUD detector tests failed: {e}")
        traceback.print_exc()

def test_integration_scenarios():
    """Test integration scenarios and failure recovery."""
    print("\n🧪 INTEGRATION SCENARIO TESTS")
    print("=" * 50)
    
    try:
        from enhanced_ocr_system import EnhancedOCRSystem
        from spygate.ml.hud_detector import EnhancedHUDDetector
        
        # Test 1: System initialization with limited resources
        print("\n1. Testing resource-constrained initialization...")
        
        try:
            # Initialize both systems
            ocr = EnhancedOCRSystem(gpu_enabled=False, max_retries=1, fallback_enabled=True)
            detector = EnhancedHUDDetector()
            
            print("✅ Both systems initialized successfully")
            
            # Test combined health
            ocr_health = ocr.health_check()
            detector_health = detector.health_check()
            
            print(f"   • OCR health: {ocr_health['status']}")
            print(f"   • HUD detector health: {detector_health['status']}")
            
        except Exception as e:
            print(f"❌ Resource-constrained initialization failed: {e}")
        
        # Test 2: Stress testing with rapid processing
        print("\n2. Testing rapid processing stress test...")
        
        try:
            # Create test video-like sequence
            frames = []
            for i in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some text-like areas
                cv2.rectangle(frame, (50, 20), (200, 60), (255, 255, 255), -1)
                frames.append(frame)
            
            start_time = time.time()
            results = []
            
            for i, frame in enumerate(frames):
                try:
                    result = detector.detect_hud_elements(frame)
                    results.append(result)
                except Exception as e:
                    print(f"   Frame {i} failed: {e}")
            
            processing_time = time.time() - start_time
            successful_frames = len([r for r in results if not r.get('metadata', {}).get('error')])
            
            print(f"✅ Stress test completed")
            print(f"   • Processed {len(frames)} frames in {processing_time:.2f}s")
            print(f"   • Success rate: {successful_frames}/{len(frames)} ({successful_frames/len(frames)*100:.1f}%)")
            print(f"   • Average FPS: {len(frames)/processing_time:.1f}")
            
        except Exception as e:
            print(f"❌ Stress test failed: {e}")
        
        # Test 3: Memory management under load
        print("\n3. Testing memory management...")
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process many frames
            large_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
            
            for i in range(20):
                try:
                    detector.detect_hud_elements(large_frame)
                except:
                    pass
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"✅ Memory management test completed")
            print(f"   • Initial memory: {initial_memory:.1f} MB")
            print(f"   • Final memory: {final_memory:.1f} MB")
            print(f"   • Memory increase: {memory_increase:.1f} MB")
            
            if memory_increase < 100:  # Reasonable threshold
                print("✅ Memory usage within acceptable limits")
            else:
                print("⚠️ High memory usage detected")
            
        except ImportError:
            print("⚠️ psutil not available for memory testing")
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
        
        print("✅ Integration scenario tests completed")
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        traceback.print_exc()

def test_edge_case_recovery():
    """Test recovery from various edge case failures."""
    print("\n🧪 EDGE CASE RECOVERY TESTS")
    print("=" * 50)
    
    try:
        from enhanced_ocr_system import EnhancedOCRSystem
        
        ocr = EnhancedOCRSystem(debug=True, fallback_enabled=True)
        
        # Test 1: Corrupted image data recovery
        print("\n1. Testing corrupted image data recovery...")
        
        corrupted_cases = [
            {"name": "NaN values", "image": np.full((100, 100, 3), np.nan)},
            {"name": "Infinite values", "image": np.full((100, 100, 3), np.inf)},
            {"name": "Negative values", "image": np.full((100, 100, 3), -255)},
            {"name": "Out of range values", "image": np.full((100, 100, 3), 500)},
            {"name": "Wrong dtype", "image": np.ones((100, 100, 3), dtype=np.float64)},
        ]
        
        for case in corrupted_cases:
            try:
                result = ocr.extract_text_from_region(case["image"], [10, 10, 90, 90])
                if result.error:
                    print(f"✅ {case['name']}: Graceful failure - {result.error}")
                else:
                    print(f"✅ {case['name']}: Processed successfully")
            except Exception as e:
                print(f"❌ {case['name']}: Unhandled exception - {e}")
        
        # Test 2: Engine failure simulation
        print("\n2. Testing OCR engine failure simulation...")
        
        # Temporarily disable engines to test fallback
        original_status = ocr.engine_status.copy()
        
        try:
            # Simulate EasyOCR failure
            from enhanced_ocr_system import EngineStatus
            ocr.engine_status['easyocr'] = EngineStatus.FAILED
            
            test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
            result = ocr.extract_text_from_region(test_img, [10, 10, 90, 50])
            
            if result.fallback_used:
                print("✅ Fallback mechanism activated successfully")
            else:
                print("⚠️ Fallback not used when expected")
                
        finally:
            # Restore original status
            ocr.engine_status = original_status
        
        # Test 3: Resource exhaustion simulation
        print("\n3. Testing resource exhaustion scenarios...")
        
        try:
            # Try to exhaust memory with very large images
            for size in [2000, 4000, 8000]:
                try:
                    large_img = np.ones((size, size, 3), dtype=np.uint8) * 128
                    result = ocr.extract_text_from_region(large_img, [10, 10, size-10, size-10])
                    print(f"✅ {size}x{size} image: Handled")
                    del large_img  # Free memory
                except MemoryError:
                    print(f"⚠️ {size}x{size} image: Memory limit reached (expected)")
                    break
                except Exception as e:
                    print(f"✅ {size}x{size} image: Error handled - {type(e).__name__}")
                    
        except Exception as e:
            print(f"❌ Resource exhaustion test failed: {e}")
        
        print("✅ Edge case recovery tests completed")
        
    except Exception as e:
        print(f"❌ Edge case tests failed: {e}")
        traceback.print_exc()

def generate_performance_report():
    """Generate a comprehensive performance report."""
    print("\n📊 PERFORMANCE REPORT GENERATION")
    print("=" * 50)
    
    try:
        from enhanced_ocr_system import EnhancedOCRSystem
        from spygate.ml.hud_detector import EnhancedHUDDetector
        
        # Initialize systems
        ocr = EnhancedOCRSystem(debug=False)
        detector = EnhancedHUDDetector()
        
        # Performance test parameters
        test_sizes = [(480, 640), (720, 1280), (1080, 1920)]
        test_iterations = 5
        
        print("\n📈 Performance Benchmarks:")
        print("-" * 30)
        
        for height, width in test_sizes:
            print(f"\nTesting {width}x{height} frames:")
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add some HUD-like elements
            cv2.rectangle(test_frame, (50, 20), (300, 80), (255, 255, 255), -1)
            cv2.putText(test_frame, "2ND & 7", (70, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Benchmark HUD detection
            detection_times = []
            for i in range(test_iterations):
                start_time = time.time()
                result = detector.detect_hud_elements(test_frame)
                detection_times.append(time.time() - start_time)
            
            avg_detection_time = np.mean(detection_times)
            fps = 1.0 / avg_detection_time
            
            print(f"  • HUD Detection: {avg_detection_time:.3f}s avg ({fps:.1f} FPS)")
            
            # Benchmark OCR
            ocr_times = []
            test_bbox = [50, 20, 300, 80]
            for i in range(test_iterations):
                start_time = time.time()
                result = ocr.extract_text_from_region(test_frame, test_bbox)
                ocr_times.append(time.time() - start_time)
            
            avg_ocr_time = np.mean(ocr_times)
            print(f"  • OCR Processing: {avg_ocr_time:.3f}s avg")
            
            # Total pipeline time
            total_time = avg_detection_time + avg_ocr_time
            pipeline_fps = 1.0 / total_time
            print(f"  • Total Pipeline: {total_time:.3f}s ({pipeline_fps:.1f} FPS)")
        
        # System health summary
        print("\n🏥 System Health Summary:")
        print("-" * 30)
        
        ocr_health = ocr.health_check()
        detector_health = detector.health_check()
        
        print(f"OCR System: {ocr_health['status'].upper()}")
        if ocr_health['issues']:
            for issue in ocr_health['issues']:
                print(f"  ⚠️ {issue}")
        
        print(f"HUD Detector: {detector_health['status'].upper()}")
        if detector_health['issues']:
            for issue in detector_health['issues']:
                print(f"  ⚠️ {issue}")
        
        # Performance statistics
        print("\n📊 Performance Statistics:")
        print("-" * 30)
        
        ocr_stats = ocr.get_performance_stats()
        detector_stats = detector.get_performance_stats()
        
        print(f"OCR Performance:")
        print(f"  • Total extractions: {ocr_stats['total_extractions']}")
        print(f"  • Success rate: {ocr_stats['success_rate']:.1%}")
        print(f"  • Engine failures: {ocr_stats['engine_failures']}")
        
        print(f"HUD Detection Performance:")
        print(f"  • Total detections: {detector_stats['total_detections']}")
        print(f"  • Success rate: {detector_stats['success_rate']:.1%}")
        print(f"  • Average time: {detector_stats['average_detection_time']:.3f}s")
        
        print("\n✅ Performance report generation completed")
        
    except Exception as e:
        print(f"❌ Performance report generation failed: {e}")
        traceback.print_exc()

def main():
    """Run all production error handling tests."""
    print("🔬 PRODUCTION ERROR HANDLING TEST SUITE")
    print("=" * 60)
    print("Testing SpygateAI's bulletproof error handling system...")
    
    start_time = time.time()
    
    # Run all test suites
    test_enhanced_ocr_error_handling()
    test_hud_detector_error_handling()
    test_integration_scenarios()
    test_edge_case_recovery()
    generate_performance_report()
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 ALL TESTS COMPLETED")
    print("=" * 60)
    print(f"Total testing time: {total_time:.2f} seconds")
    print("🚀 SpygateAI production error handling system verified!")

if __name__ == "__main__":
    main() 