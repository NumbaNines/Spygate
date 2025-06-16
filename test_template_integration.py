#!/usr/bin/env python3
"""
SpygateAI Down Template Integration Test

Expert-level integration testing for production validation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Import our detector
from down_template_detector import DownTemplateDetector, DownTemplateMatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateIntegrationTest:
    """Expert integration test for down template detection."""

    def __init__(self):
        """Initialize the test."""
        self.detector = DownTemplateDetector(
            templates_dir=Path("down_templates_real"),
            debug_output_dir=Path("debug/integration_test"),
        )

        # Create test results directory
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)

        logger.info("Integration test initialized")

    def run_expert_validation(self) -> dict:
        """Run expert-level validation tests."""
        print("🚀 SpygateAI Down Template Expert Validation")
        print("=" * 50)

        results = {}

        # Test 1: Template System Validation
        print("\n📏 Testing Template System...")
        results["template_validation"] = self._validate_template_system()

        # Test 2: Detection Accuracy Test
        print("\n🎯 Testing Detection Accuracy...")
        results["detection_accuracy"] = self._test_detection_accuracy()

        # Test 3: Performance Benchmark
        print("\n⚡ Benchmarking Performance...")
        results["performance"] = self._benchmark_performance()

        # Test 4: Context Awareness Test
        print("\n🧠 Testing Context Awareness...")
        results["context_awareness"] = self._test_context_awareness()

        # Test 5: Edge Case Handling
        print("\n🛡️ Testing Edge Cases...")
        results["edge_cases"] = self._test_edge_cases()

        # Generate expert recommendation
        results["expert_recommendation"] = self._generate_expert_recommendation(results)

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _validate_template_system(self) -> dict:
        """Validate the template loading and organization system."""
        validation = {
            "templates_loaded": len(self.detector.templates),
            "normal_count": 0,
            "goal_count": 0,
            "all_downs_covered": False,
            "size_analysis": {},
            "validation_score": 0.0,
        }

        # Analyze loaded templates
        normal_downs = set()
        goal_downs = set()

        for name, data in self.detector.templates.items():
            if data["situation_type"].value == "normal":
                validation["normal_count"] += 1
                normal_downs.add(data["down"])
            else:
                validation["goal_count"] += 1
                goal_downs.add(data["down"])

            # Size analysis
            h, w = data["size"]
            validation["size_analysis"][name] = {"width": w, "height": h}

        # Check coverage
        validation["all_downs_covered"] = len(normal_downs) == 4 and len(goal_downs) >= 3

        # Calculate validation score
        score = 0.0
        score += 0.3 if validation["templates_loaded"] >= 7 else 0.0
        score += 0.3 if validation["all_downs_covered"] else 0.0
        score += 0.2 if validation["normal_count"] >= 4 else 0.0
        score += 0.2 if validation["goal_count"] >= 3 else 0.0

        validation["validation_score"] = score

        print(
            f"   ✅ Templates: {validation['templates_loaded']} "
            f"({validation['normal_count']} normal, {validation['goal_count']} goal)"
        )
        print(f"   📊 Validation Score: {score:.1%}")

        return validation

    def _test_detection_accuracy(self) -> dict:
        """Test detection accuracy with synthetic and real-like data."""
        accuracy = {
            "synthetic_tests": 0,
            "successful_detections": 0,
            "average_confidence": 0.0,
            "confidence_distribution": {},
            "accuracy_score": 0.0,
        }

        # Create synthetic test cases
        test_cases = [
            {"text": "1ST & 10", "is_goal": False, "expected_down": 1},
            {"text": "2ND & 7", "is_goal": False, "expected_down": 2},
            {"text": "3RD & 3", "is_goal": False, "expected_down": 3},
            {"text": "4TH & 1", "is_goal": False, "expected_down": 4},
            {"text": "1ST & GOAL", "is_goal": True, "expected_down": 1},
            {"text": "2ND & GOAL", "is_goal": True, "expected_down": 2},
            {"text": "4TH & GOAL", "is_goal": True, "expected_down": 4},
        ]

        confidences = []

        for test_case in test_cases:
            # Create synthetic frame
            frame = self._create_synthetic_frame(test_case["text"])
            bbox = (1200, 50, 1400, 100)

            # Test detection
            result = self.detector.detect_down_in_yolo_region(frame, bbox, test_case["is_goal"])

            accuracy["synthetic_tests"] += 1

            if result and result.down == test_case["expected_down"]:
                accuracy["successful_detections"] += 1
                confidences.append(result.confidence)

                # Confidence distribution
                conf_range = f"{result.confidence:.1f}"
                accuracy["confidence_distribution"][conf_range] = (
                    accuracy["confidence_distribution"].get(conf_range, 0) + 1
                )

        # Calculate metrics
        if confidences:
            accuracy["average_confidence"] = np.mean(confidences)

        accuracy["accuracy_score"] = accuracy["successful_detections"] / accuracy["synthetic_tests"]

        print(
            f"   🎯 Accuracy: {accuracy['accuracy_score']:.1%} "
            f"({accuracy['successful_detections']}/{accuracy['synthetic_tests']})"
        )
        print(f"   📈 Avg Confidence: {accuracy['average_confidence']:.3f}")

        return accuracy

    def _create_synthetic_frame(self, text: str) -> np.ndarray:
        """Create a synthetic frame with down/distance text."""
        # Create 1080p frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Add HUD-like background
        cv2.rectangle(frame, (1150, 30), (1450, 120), (40, 40, 40), -1)

        # Add text region
        cv2.rectangle(frame, (1200, 50), (1400, 100), (255, 255, 255), -1)

        # Add the text
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = 1300 - text_size[0] // 2
        text_y = 75 + text_size[1] // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )

        return frame

    def _benchmark_performance(self) -> dict:
        """Benchmark template matching performance."""
        performance = {
            "template_times_ms": [],
            "average_time_ms": 0.0,
            "std_time_ms": 0.0,
            "fps_capability": 0.0,
            "performance_grade": "Unknown",
        }

        # Create test frame
        test_frame = self._create_synthetic_frame("1ST & 10")
        test_bbox = (1200, 50, 1400, 100)

        # Run 100 iterations for reliable timing
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = self.detector.detect_down_in_yolo_region(test_frame, test_bbox)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        performance["template_times_ms"] = times
        performance["average_time_ms"] = np.mean(times)
        performance["std_time_ms"] = np.std(times)
        performance["fps_capability"] = 1000 / performance["average_time_ms"]

        # Performance grading
        avg_time = performance["average_time_ms"]
        if avg_time < 10:
            performance["performance_grade"] = "A+ (Excellent)"
        elif avg_time < 20:
            performance["performance_grade"] = "A (Very Good)"
        elif avg_time < 50:
            performance["performance_grade"] = "B (Good)"
        elif avg_time < 100:
            performance["performance_grade"] = "C (Acceptable)"
        else:
            performance["performance_grade"] = "D (Needs Improvement)"

        print(f"   ⚡ Average Time: {avg_time:.2f}ms ± {performance['std_time_ms']:.2f}ms")
        print(f"   🏃‍♂️ FPS Capability: {performance['fps_capability']:.1f} FPS")
        print(f"   📊 Grade: {performance['performance_grade']}")

        return performance

    def _test_context_awareness(self) -> dict:
        """Test context-aware template selection."""
        context = {"normal_priority_test": False, "goal_priority_test": False, "context_score": 0.0}

        # Test 1: Normal situation should prioritize normal templates
        normal_frame = self._create_synthetic_frame("1ST & 10")
        normal_result = self.detector.detect_down_in_yolo_region(
            normal_frame, (1200, 50, 1400, 100), is_goal_situation=False
        )

        if normal_result and normal_result.situation_type.value == "normal":
            context["normal_priority_test"] = True

        # Test 2: Goal situation should prioritize goal templates
        goal_frame = self._create_synthetic_frame("1ST & GOAL")
        goal_result = self.detector.detect_down_in_yolo_region(
            goal_frame, (1175, 50, 1425, 100), is_goal_situation=True  # Note: 24px shift
        )

        if goal_result and goal_result.situation_type.value == "goal":
            context["goal_priority_test"] = True

        # Calculate context score
        context["context_score"] = (0.5 if context["normal_priority_test"] else 0.0) + (
            0.5 if context["goal_priority_test"] else 0.0
        )

        print(f"   🧠 Context Awareness: {context['context_score']:.1%}")
        print(f"   📍 Normal Priority: {'✅' if context['normal_priority_test'] else '❌'}")
        print(f"   🥅 Goal Priority: {'✅' if context['goal_priority_test'] else '❌'}")

        return context

    def _test_edge_cases(self) -> dict:
        """Test edge case handling."""
        edge_cases = {
            "empty_roi": False,
            "invalid_bbox": False,
            "no_match": False,
            "corrupted_image": False,
            "edge_case_score": 0.0,
        }

        # Test 1: Empty ROI
        try:
            empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(empty_frame, (0, 0, 0, 0))
            edge_cases["empty_roi"] = result is None  # Should return None gracefully
        except:
            edge_cases["empty_roi"] = False

        # Test 2: Invalid bbox (outside frame)
        try:
            test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(test_frame, (2000, 2000, 2100, 2100))
            edge_cases["invalid_bbox"] = result is None  # Should handle gracefully
        except:
            edge_cases["invalid_bbox"] = False

        # Test 3: No match scenario
        try:
            noise_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            result = self.detector.detect_down_in_yolo_region(noise_frame, (1200, 50, 1400, 100))
            edge_cases["no_match"] = True  # Should not crash
        except:
            edge_cases["no_match"] = False

        # Test 4: Corrupted image
        try:
            corrupted_frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)  # All white
            result = self.detector.detect_down_in_yolo_region(
                corrupted_frame, (1200, 50, 1400, 100)
            )
            edge_cases["corrupted_image"] = True  # Should not crash
        except:
            edge_cases["corrupted_image"] = False

        # Calculate edge case score
        edge_cases["edge_case_score"] = (
            sum(
                [
                    edge_cases["empty_roi"],
                    edge_cases["invalid_bbox"],
                    edge_cases["no_match"],
                    edge_cases["corrupted_image"],
                ]
            )
            / 4.0
        )

        print(f"   🛡️ Edge Case Handling: {edge_cases['edge_case_score']:.1%}")

        return edge_cases

    def _generate_expert_recommendation(self, results: dict) -> dict:
        """Generate expert recommendation based on all test results."""
        # Calculate overall score
        weights = {
            "template_validation": 0.25,
            "detection_accuracy": 0.30,
            "performance": 0.20,
            "context_awareness": 0.15,
            "edge_cases": 0.10,
        }

        scores = {
            "template_validation": results["template_validation"]["validation_score"],
            "detection_accuracy": results["detection_accuracy"]["accuracy_score"],
            "performance": 1.0 if results["performance"]["average_time_ms"] < 50 else 0.5,
            "context_awareness": results["context_awareness"]["context_score"],
            "edge_cases": results["edge_cases"]["edge_case_score"],
        }

        overall_score = sum(scores[key] * weights[key] for key in weights.keys())

        # Generate recommendation
        if overall_score >= 0.9:
            recommendation = "🚀 PRODUCTION READY - Deploy immediately"
            priority = "HIGH"
        elif overall_score >= 0.8:
            recommendation = "✅ READY WITH MONITORING - Deploy with close monitoring"
            priority = "MEDIUM-HIGH"
        elif overall_score >= 0.7:
            recommendation = "⚠️ NEEDS MINOR FIXES - Address issues before deployment"
            priority = "MEDIUM"
        elif overall_score >= 0.6:
            recommendation = "🔧 REQUIRES IMPROVEMENTS - Significant work needed"
            priority = "LOW"
        else:
            recommendation = "❌ NOT READY - Major rework required"
            priority = "BLOCKED"

        return {
            "overall_score": overall_score,
            "component_scores": scores,
            "recommendation": recommendation,
            "priority": priority,
            "next_steps": self._get_next_steps(overall_score, results),
        }

    def _get_next_steps(self, score: float, results: dict) -> list:
        """Get recommended next steps based on test results."""
        steps = []

        if score >= 0.8:
            steps.extend(
                [
                    "Integrate into enhanced_game_analyzer.py",
                    "Add production monitoring",
                    "Test with real SpygateAI video frames",
                    "Deploy to production environment",
                ]
            )
        else:
            if results["template_validation"]["validation_score"] < 0.8:
                steps.append("Add missing template variants")

            if results["detection_accuracy"]["accuracy_score"] < 0.8:
                steps.append("Tune confidence thresholds")

            if results["performance"]["average_time_ms"] > 50:
                steps.append("Optimize preprocessing pipeline")

            if results["context_awareness"]["context_score"] < 0.8:
                steps.append("Improve context detection logic")

        return steps

    def _save_results(self, results: dict) -> None:
        """Save test results."""
        results_file = self.results_dir / "expert_validation_results.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        json_results = json.loads(json.dumps(results, default=convert_types))

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    def _print_summary(self, results: dict) -> None:
        """Print expert summary."""
        print("\n" + "=" * 50)
        print("📊 EXPERT VALIDATION SUMMARY")
        print("=" * 50)

        rec = results["expert_recommendation"]
        print(f"Overall Score: {rec['overall_score']:.1%}")
        print(f"Recommendation: {rec['recommendation']}")
        print(f"Priority: {rec['priority']}")

        print("\nComponent Scores:")
        for component, score in rec["component_scores"].items():
            print(f"  {component}: {score:.1%}")

        print("\nNext Steps:")
        for i, step in enumerate(rec["next_steps"], 1):
            print(f"  {i}. {step}")


def main():
    """Run the expert validation."""
    test = TemplateIntegrationTest()
    results = test.run_expert_validation()
    return results


if __name__ == "__main__":
    main()
