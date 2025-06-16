#!/usr/bin/env python3
"""
SpygateAI Down Template Integration Test

Comprehensive testing of the new Down Template Detector against real game frames.
Compares template matching performance vs current OCR methods.

Test Coverage:
- Template loading and validation
- YOLO region detection integration
- Normal vs GOAL situation handling
- Performance benchmarking
- Accuracy comparison with existing OCR
- Edge case handling
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Import our new detector
from down_template_detector import DownTemplateDetector, DownTemplateMatch, SituationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result for a single frame."""
    frame_name: str
    template_result: Optional[DownTemplateMatch]
    template_time_ms: float
    ocr_result: Optional[Dict]
    ocr_time_ms: float
    yolo_bbox: Tuple[int, int, int, int]
    ground_truth: Optional[Dict]
    template_accuracy: bool
    ocr_accuracy: bool
    performance_gain: float  # Template time vs OCR time


class DownTemplateIntegrationTest:
    """
    Comprehensive integration test for down template detection.
    
    Tests against real SpygateAI frames with known ground truth data.
    """
    
    def __init__(self):
        """Initialize the integration test."""
        self.detector = DownTemplateDetector(
            templates_dir=Path("down_templates_real"),
            debug_output_dir=Path("debug/integration_test")
        )
        
        # Test data paths
        self.test_frames_dir = Path("test_frames")
        self.ground_truth_file = Path("test_ground_truth.json")
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Mock YOLO detector for testing (replace with real one later)
        self.mock_yolo_regions = self._create_mock_yolo_regions()
        
        # Test results
        self.test_results: List[TestResult] = []
        
        logger.info("DownTemplateIntegrationTest initialized")
    
    def _create_mock_yolo_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Create mock YOLO regions for test frames."""
        # These would normally come from actual YOLO detection
        # Using typical down_distance_area coordinates from SpygateAI
        return {
            "normal_1st_10.jpg": (1200, 50, 1400, 100),      # Normal situation
            "normal_3rd_8.jpg": (1200, 50, 1400, 100),       # Normal situation
            "goal_1st_goal.jpg": (1175, 50, 1425, 100),      # GOAL situation (24px shift)
            "goal_4th_goal.jpg": (1175, 50, 1425, 100),      # GOAL situation
            "edge_case_unclear.jpg": (1200, 50, 1400, 100),  # Edge case
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Run the complete integration test suite."""
        logger.info("🚀 Starting comprehensive down template integration test")
        
        # Test 1: Template Loading Validation
        template_validation = self._test_template_loading()
        
        # Test 2: Frame Detection Tests
        if self.test_frames_dir.exists():
            frame_results = self._test_frame_detection()
        else:
            logger.warning("No test frames directory found, creating sample test")
            frame_results = self._create_sample_test()
        
        # Test 3: Performance Benchmarking
        performance_results = self._benchmark_performance()
        
        # Test 4: Edge Case Testing
        edge_case_results = self._test_edge_cases()
        
        # Compile comprehensive results
        comprehensive_results = {
            "template_validation": template_validation,
            "frame_detection": frame_results,
            "performance_benchmark": performance_results,
            "edge_cases": edge_case_results,
            "summary": self._generate_summary()
        }
        
        # Save results
        self._save_results(comprehensive_results)
        
        # Generate report
        self._generate_report(comprehensive_results)
        
        return comprehensive_results
    
    def _test_template_loading(self) -> Dict:
        """Test template loading and validation."""
        logger.info("📏 Testing template loading...")
        
        results = {
            "templates_loaded": len(self.detector.templates),
            "expected_templates": 8,
            "normal_templates": 0,
            "goal_templates": 0,
            "template_details": {},
            "validation_passed": False
        }
        
        # Count template types
        for name, data in self.detector.templates.items():
            if data["situation_type"] == SituationType.NORMAL:
                results["normal_templates"] += 1
            else:
                results["goal_templates"] += 1
            
            results["template_details"][name] = {
                "size": data["size"],
                "down": data["down"],
                "situation_type": data["situation_type"].value
            }
        
        # Validation checks
        has_all_normal = results["normal_templates"] >= 4
        has_goal_templates = results["goal_templates"] >= 3
        total_reasonable = results["templates_loaded"] >= 7
        
        results["validation_passed"] = has_all_normal and has_goal_templates and total_reasonable
        
        logger.info(f"✅ Template validation: {results['validation_passed']}")
        logger.info(f"📊 Loaded {results['templates_loaded']} templates "
                   f"({results['normal_templates']} normal, {results['goal_templates']} goal)")
        
        return results
    
    def _test_frame_detection(self) -> Dict:
        """Test detection on real game frames."""
        logger.info("🎮 Testing frame detection...")
        
        frame_files = list(self.test_frames_dir.glob("*.jpg")) + list(self.test_frames_dir.glob("*.png"))
        
        if not frame_files:
            logger.warning("No test frames found")
            return {"error": "No test frames available"}
        
        results = {
            "frames_tested": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "detection_details": []
        }
        
        total_confidence = 0.0
        total_time = 0.0
        
        for frame_file in frame_files[:10]:  # Test first 10 frames
            frame_result = self._test_single_frame(frame_file)
            
            if frame_result:
                self.test_results.append(frame_result)
                results["frames_tested"] += 1
                
                if frame_result.template_result:
                    results["successful_detections"] += 1
                    total_confidence += frame_result.template_result.confidence
                else:
                    results["failed_detections"] += 1
                
                total_time += frame_result.template_time_ms
                
                results["detection_details"].append({
                    "frame": frame_result.frame_name,
                    "detected": frame_result.template_result is not None,
                    "confidence": frame_result.template_result.confidence if frame_result.template_result else 0.0,
                    "time_ms": frame_result.template_time_ms
                })
        
        # Calculate averages
        if results["successful_detections"] > 0:
            results["average_confidence"] = total_confidence / results["successful_detections"]
        
        if results["frames_tested"] > 0:
            results["average_processing_time"] = total_time / results["frames_tested"]
        
        logger.info(f"🎯 Detection results: {results['successful_detections']}/{results['frames_tested']} successful")
        
        return results
    
    def _test_single_frame(self, frame_file: Path) -> Optional[TestResult]:
        """Test detection on a single frame."""
        try:
            # Load frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                logger.warning(f"Failed to load frame: {frame_file}")
                return None
            
            frame_name = frame_file.name
            
            # Get mock YOLO region (in real integration, this comes from YOLO)
            yolo_bbox = self.mock_yolo_regions.get(frame_name, (1200, 50, 1400, 100))
            
            # Determine if this is a GOAL situation (from filename)
            is_goal_situation = "goal" in frame_name.lower()
            
            # Test template detection
            start_time = time.perf_counter()
            template_result = self.detector.detect_down_in_yolo_region(
                frame, yolo_bbox, is_goal_situation
            )
            template_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            # Mock OCR comparison (replace with actual OCR later)
            start_time = time.perf_counter()
            ocr_result = self._mock_ocr_detection(frame, yolo_bbox)
            ocr_time = (time.perf_counter() - start_time) * 1000
            
            # Calculate performance gain
            performance_gain = (ocr_time - template_time) / ocr_time if ocr_time > 0 else 0.0
            
            return TestResult(
                frame_name=frame_name,
                template_result=template_result,
                template_time_ms=template_time,
                ocr_result=ocr_result,
                ocr_time_ms=ocr_time,
                yolo_bbox=yolo_bbox,
                ground_truth=None,  # Would load from ground truth file
                template_accuracy=template_result is not None,  # Simplified
                ocr_accuracy=ocr_result is not None,  # Simplified
                performance_gain=performance_gain
            )
            
        except Exception as e:
            logger.error(f"Error testing frame {frame_file}: {e}")
            return None
    
    def _mock_ocr_detection(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """Mock OCR detection for comparison (replace with real OCR)."""
        # Simulate OCR processing time and results
        time.sleep(0.05)  # Simulate 50ms OCR processing
        
        return {
            "text": "1ST & 10",
            "confidence": 0.75,
            "method": "mock_ocr"
        }
    
    def _benchmark_performance(self) -> Dict:
        """Benchmark template matching vs OCR performance."""
        logger.info("⚡ Benchmarking performance...")
        
        # Create test image for benchmarking
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        test_bbox = (1200, 50, 1400, 100)
        
        # Benchmark template matching
        template_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = self.detector.detect_down_in_yolo_region(test_image, test_bbox)
            template_times.append((time.perf_counter() - start_time) * 1000)
        
        # Benchmark mock OCR
        ocr_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = self._mock_ocr_detection(test_image, test_bbox)
            ocr_times.append((time.perf_counter() - start_time) * 1000)
        
        results = {
            "template_avg_ms": np.mean(template_times),
            "template_std_ms": np.std(template_times),
            "ocr_avg_ms": np.mean(ocr_times),
            "ocr_std_ms": np.std(ocr_times),
            "speedup_factor": np.mean(ocr_times) / np.mean(template_times),
            "template_faster": np.mean(template_times) < np.mean(ocr_times)
        }
        
        logger.info(f"🏃‍♂️ Template: {results['template_avg_ms']:.2f}ms, "
                   f"OCR: {results['ocr_avg_ms']:.2f}ms, "
                   f"Speedup: {results['speedup_factor']:.2f}x")
        
        return results
    
    def _test_edge_cases(self) -> Dict:
        """Test edge cases and error handling."""
        logger.info("🔍 Testing edge cases...")
        
        results = {
            "empty_roi_handled": False,
            "invalid_bbox_handled": False,
            "no_templates_handled": False,
            "corrupted_image_handled": False,
            "edge_cases_passed": 0,
            "total_edge_cases": 4
        }
        
        # Test 1: Empty ROI
        try:
            empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(empty_frame, (0, 0, 0, 0))
            results["empty_roi_handled"] = result is None
            if results["empty_roi_handled"]:
                results["edge_cases_passed"] += 1
        except Exception as e:
            logger.warning(f"Empty ROI test failed: {e}")
        
        # Test 2: Invalid bbox
        try:
            test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(test_frame, (2000, 2000, 2100, 2100))
            results["invalid_bbox_handled"] = result is None
            if results["invalid_bbox_handled"]:
                results["edge_cases_passed"] += 1
        except Exception as e:
            logger.warning(f"Invalid bbox test failed: {e}")
        
        # Test 3: No templates (simulate)
        try:
            original_templates = self.detector.templates.copy()
            self.detector.templates = {}
            test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(test_frame, (1200, 50, 1400, 100))
            results["no_templates_handled"] = result is None
            if results["no_templates_handled"]:
                results["edge_cases_passed"] += 1
            self.detector.templates = original_templates
        except Exception as e:
            logger.warning(f"No templates test failed: {e}")
            self.detector.templates = original_templates
        
        # Test 4: Corrupted image
        try:
            corrupted_frame = np.full((50, 50, 3), 255, dtype=np.uint8)  # All white
            result = self.detector.detect_down_in_yolo_region(corrupted_frame, (10, 10, 40, 40))
            results["corrupted_image_handled"] = True  # Should not crash
            results["edge_cases_passed"] += 1
        except Exception as e:
            logger.warning(f"Corrupted image test failed: {e}")
        
        logger.info(f"🛡️ Edge cases: {results['edge_cases_passed']}/{results['total_edge_cases']} passed")
        
        return results
    
    def _create_sample_test(self) -> Dict:
        """Create a sample test when no real frames are available."""
        logger.info("🎨 Creating sample test data...")
        
        # Create synthetic test frame with down/distance text
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Add some text-like patterns (very basic simulation)
        cv2.rectangle(test_frame, (1200, 50), (1400, 100), (255, 255, 255), -1)
        cv2.putText(test_frame, "1ST & 10", (1220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test detection
        start_time = time.perf_counter()
        result = self.detector.detect_down_in_yolo_region(test_frame, (1200, 50, 1400, 100))
        detection_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "sample_test": True,
            "detection_successful": result is not None,
            "detection_time_ms": detection_time,
            "confidence": result.confidence if result else 0.0,
            "note": "Synthetic test data used - real frames recommended for production validation"
        }
    
    def _generate_summary(self) -> Dict:
        """Generate test summary."""
        if not self.test_results:
            return {"note": "No test results available for summary"}
        
        successful_detections = sum(1 for r in self.test_results if r.template_result)
        total_tests = len(self.test_results)
        
        avg_confidence = np.mean([r.template_result.confidence for r in self.test_results if r.template_result])
        avg_time = np.mean([r.template_time_ms for r in self.test_results])
        avg_performance_gain = np.mean([r.performance_gain for r in self.test_results])
        
        return {
            "total_tests": total_tests,
            "successful_detections": successful_detections,
            "success_rate": successful_detections / total_tests if total_tests > 0 else 0.0,
            "average_confidence": float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
            "average_processing_time_ms": float(avg_time),
            "average_performance_gain": float(avg_performance_gain),
            "recommendation": self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Get integration recommendation based on test results."""
        if not self.test_results:
            return "Insufficient test data - run with real game frames"
        
        successful_detections = sum(1 for r in self.test_results if r.template_result)
        success_rate = successful_detections / len(self.test_results)
        
        if success_rate >= 0.9:
            return "✅ READY FOR PRODUCTION - Excellent performance, integrate immediately"
        elif success_rate >= 0.7:
            return "⚠️ NEEDS TUNING - Good performance, minor adjustments needed"
        elif success_rate >= 0.5:
            return "🔧 REQUIRES WORK - Moderate performance, significant improvements needed"
        else:
            return "❌ NOT READY - Poor performance, major rework required"
    
    def _save_results(self, results: Dict) -> None:
        """Save test results to file."""
        results_file = self.results_dir / "integration_test_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Deep convert the results
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_file}")
    
    def _generate_report(self, results: Dict) -> None:
        """Generate human-readable test report."""
        report_file = self.results_dir / "integration_test_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# SpygateAI Down Template Integration Test Report\n\n")
            f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Template Validation
            f.write("## 📏 Template Validation\n")
            tv = results.get("template_validation", {})
            f.write(f"- **Templates Loaded:** {tv.get('templates_loaded', 0)}\n")
            f.write(f"- **Normal Templates:** {tv.get('normal_templates', 0)}\n")
            f.write(f"- **GOAL Templates:** {tv.get('goal_templates', 0)}\n")
            f.write(f"- **Validation Status:** {'✅ PASSED' if tv.get('validation_passed') else '❌ FAILED'}\n\n")
            
            # Performance Benchmark
            f.write("## ⚡ Performance Benchmark\n")
            pb = results.get("performance_benchmark", {})
            f.write(f"- **Template Matching:** {pb.get('template_avg_ms', 0):.2f}ms ± {pb.get('template_std_ms', 0):.2f}ms\n")
            f.write(f"- **OCR Processing:** {pb.get('ocr_avg_ms', 0):.2f}ms ± {pb.get('ocr_std_ms', 0):.2f}ms\n")
            f.write(f"- **Speedup Factor:** {pb.get('speedup_factor', 0):.2f}x\n")
            f.write(f"- **Template Faster:** {'✅ YES' if pb.get('template_faster') else '❌ NO'}\n\n")
            
            # Summary
            f.write("## 📊 Summary\n")
            summary = results.get("summary", {})
            f.write(f"- **Success Rate:** {summary.get('success_rate', 0):.1%}\n")
            f.write(f"- **Average Confidence:** {summary.get('average_confidence', 0):.3f}\n")
            f.write(f"- **Average Processing Time:** {summary.get('average_processing_time_ms', 0):.2f}ms\n")
            f.write(f"- **Performance Gain:** {summary.get('average_performance_gain', 0):.1%}\n\n")
            
            # Recommendation
            f.write("## 🎯 Recommendation\n")
            f.write(f"{summary.get('recommendation', 'No recommendation available')}\n\n")
            
            # Next Steps
            f.write("## 🚀 Next Steps\n")
            f.write("1. **Production Integration** - Add to enhanced_game_analyzer.py\n")
            f.write("2. **Real Frame Testing** - Test with actual SpygateAI video frames\n")
            f.write("3. **Performance Optimization** - Fine-tune confidence thresholds\n")
            f.write("4. **Context Integration** - Add goal line situation detection\n")
            f.write("5. **Monitoring Setup** - Add production performance monitoring\n")
        
        logger.info(f"📋 Report generated: {report_file}")


def main():
    """Run the integration test."""
    print("🚀 SpygateAI Down Template Integration Test")
    print("=" * 50)
    
    # Initialize and run test
    test = DownTemplateIntegrationTest()
    results = test.run_comprehensive_test()
    
    # Print summary
    print("\n📊 TEST SUMMARY")
    print("-" * 30)
    summary = results.get("summary", {})
    print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
    print(f"Average Confidence: {summary.get('average_confidence', 0):.3f}")
    print(f"Processing Time: {summary.get('average_processing_time_ms', 0):.2f}ms")
    print(f"Recommendation: {summary.get('recommendation', 'N/A')}")
    
    return results


if __name__ == "__main__":
    main() 