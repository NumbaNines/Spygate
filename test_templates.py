#!/usr/bin/env python3
"""
SpygateAI Down Template Test

Simple but comprehensive test for the down template detector.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from down_template_detector import DownTemplateDetector


def create_test_frame(text: str) -> np.ndarray:
    """Create a synthetic test frame with down/distance text."""
    # Create 1080p frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add HUD-like background
    cv2.rectangle(frame, (1150, 30), (1450, 120), (40, 40, 40), -1)

    # Add text region (white background)
    cv2.rectangle(frame, (1200, 50), (1400, 100), (255, 255, 255), -1)

    # Add the text
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = 1300 - text_size[0] // 2
    text_y = 75 + text_size[1] // 2

    cv2.putText(
        frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
    )

    return frame


def test_template_loading():
    """Test template loading."""
    print("📏 Testing Template Loading...")

    detector = DownTemplateDetector()

    print(f"   ✅ Loaded {len(detector.templates)} templates")

    normal_count = 0
    goal_count = 0

    for name, data in detector.templates.items():
        if data["situation_type"].value == "normal":
            normal_count += 1
        else:
            goal_count += 1

        h, w = data["size"]
        print(f"   📏 {name}: {w}x{h}px ({data['situation_type'].value})")

    print(f"   📊 Summary: {normal_count} normal, {goal_count} goal templates")

    return detector


def test_detection_accuracy(detector):
    """Test detection accuracy."""
    print("\n🎯 Testing Detection Accuracy...")

    test_cases = [
        {"text": "1ST & 10", "is_goal": False, "expected_down": 1},
        {"text": "2ND & 7", "is_goal": False, "expected_down": 2},
        {"text": "3RD & 3", "is_goal": False, "expected_down": 3},
        {"text": "4TH & 1", "is_goal": False, "expected_down": 4},
        {"text": "1ST & GOAL", "is_goal": True, "expected_down": 1},
        {"text": "2ND & GOAL", "is_goal": True, "expected_down": 2},
        {"text": "4TH & GOAL", "is_goal": True, "expected_down": 4},
    ]

    successful = 0
    total = len(test_cases)
    confidences = []

    for test_case in test_cases:
        # Create test frame
        frame = create_test_frame(test_case["text"])
        bbox = (1200, 50, 1400, 100)

        # Adjust bbox for GOAL situations (24px shift)
        if test_case["is_goal"]:
            bbox = (1175, 50, 1425, 100)

        # Test detection
        result = detector.detect_down_in_yolo_region(frame, bbox, test_case["is_goal"])

        if result:
            correct = result.down == test_case["expected_down"]
            if correct:
                successful += 1
                confidences.append(result.confidence)

            print(
                f"   {'✅' if correct else '❌'} {test_case['text']}: "
                f"Detected {result.down} (conf: {result.confidence:.3f}, "
                f"template: {result.template_name})"
            )
        else:
            print(f"   ❌ {test_case['text']}: No detection")

    accuracy = successful / total
    avg_confidence = np.mean(confidences) if confidences else 0.0

    print(f"   📊 Accuracy: {accuracy:.1%} ({successful}/{total})")
    print(f"   📈 Average Confidence: {avg_confidence:.3f}")

    return accuracy, avg_confidence


def test_performance(detector):
    """Test performance."""
    print("\n⚡ Testing Performance...")

    # Create test frame
    test_frame = create_test_frame("1ST & 10")
    test_bbox = (1200, 50, 1400, 100)

    # Warm up
    for _ in range(5):
        detector.detect_down_in_yolo_region(test_frame, test_bbox)

    # Benchmark
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        result = detector.detect_down_in_yolo_region(test_frame, test_bbox)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps_capability = 1000 / avg_time

    print(f"   ⚡ Average Time: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"   🏃‍♂️ FPS Capability: {fps_capability:.1f} FPS")

    # Performance grade
    if avg_time < 10:
        grade = "A+ (Excellent)"
    elif avg_time < 20:
        grade = "A (Very Good)"
    elif avg_time < 50:
        grade = "B (Good)"
    elif avg_time < 100:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs Improvement)"

    print(f"   📊 Performance Grade: {grade}")

    return avg_time, fps_capability


def test_context_awareness(detector):
    """Test context-aware template selection."""
    print("\n🧠 Testing Context Awareness...")

    # Test normal situation
    normal_frame = create_test_frame("1ST & 10")
    normal_result = detector.detect_down_in_yolo_region(
        normal_frame, (1200, 50, 1400, 100), is_goal_situation=False
    )

    normal_correct = normal_result and normal_result.situation_type.value == "normal"

    # Test goal situation
    goal_frame = create_test_frame("1ST & GOAL")
    goal_result = detector.detect_down_in_yolo_region(
        goal_frame, (1175, 50, 1425, 100), is_goal_situation=True
    )

    goal_correct = goal_result and goal_result.situation_type.value == "goal"

    print(f"   📍 Normal Context: {'✅' if normal_correct else '❌'}")
    print(f"   🥅 Goal Context: {'✅' if goal_correct else '❌'}")

    context_score = (normal_correct + goal_correct) / 2
    print(f"   🧠 Context Score: {context_score:.1%}")

    return context_score


def test_edge_cases(detector):
    """Test edge case handling."""
    print("\n🛡️ Testing Edge Cases...")

    edge_cases_passed = 0
    total_edge_cases = 4

    # Test 1: Empty ROI
    try:
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_down_in_yolo_region(empty_frame, (0, 0, 0, 0))
        if result is None:
            edge_cases_passed += 1
            print("   ✅ Empty ROI handled gracefully")
        else:
            print("   ❌ Empty ROI not handled properly")
    except Exception as e:
        print(f"   ❌ Empty ROI crashed: {e}")

    # Test 2: Invalid bbox
    try:
        test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_down_in_yolo_region(test_frame, (2000, 2000, 2100, 2100))
        if result is None:
            edge_cases_passed += 1
            print("   ✅ Invalid bbox handled gracefully")
        else:
            print("   ❌ Invalid bbox not handled properly")
    except Exception as e:
        print(f"   ❌ Invalid bbox crashed: {e}")

    # Test 3: No match scenario
    try:
        noise_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = detector.detect_down_in_yolo_region(noise_frame, (1200, 50, 1400, 100))
        edge_cases_passed += 1
        print("   ✅ No match scenario handled")
    except Exception as e:
        print(f"   ❌ No match scenario crashed: {e}")

    # Test 4: Corrupted image
    try:
        corrupted_frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        result = detector.detect_down_in_yolo_region(corrupted_frame, (1200, 50, 1400, 100))
        edge_cases_passed += 1
        print("   ✅ Corrupted image handled")
    except Exception as e:
        print(f"   ❌ Corrupted image crashed: {e}")

    edge_score = edge_cases_passed / total_edge_cases
    print(f"   🛡️ Edge Case Score: {edge_score:.1%} ({edge_cases_passed}/{total_edge_cases})")

    return edge_score


def generate_recommendation(accuracy, avg_confidence, avg_time, context_score, edge_score):
    """Generate expert recommendation."""
    print("\n" + "=" * 50)
    print("📊 EXPERT RECOMMENDATION")
    print("=" * 50)

    # Calculate overall score
    overall_score = (
        accuracy * 0.30
        + (avg_confidence) * 0.20  # 30% weight on accuracy
        + (1.0 if avg_time < 50 else 0.5) * 0.20  # 20% weight on confidence
        + context_score * 0.15  # 20% weight on performance
        + edge_score * 0.15  # 15% weight on context awareness  # 15% weight on edge case handling
    )

    print(f"Overall Score: {overall_score:.1%}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Avg Confidence: {avg_confidence:.3f}")
    print(f"Performance: {avg_time:.2f}ms")
    print(f"Context Awareness: {context_score:.1%}")
    print(f"Edge Case Handling: {edge_score:.1%}")

    # Generate recommendation
    if overall_score >= 0.9:
        recommendation = "🚀 PRODUCTION READY - Deploy immediately"
        next_steps = [
            "Integrate into enhanced_game_analyzer.py",
            "Add production monitoring",
            "Test with real SpygateAI frames",
        ]
    elif overall_score >= 0.8:
        recommendation = "✅ READY WITH MONITORING - Deploy with close monitoring"
        next_steps = [
            "Minor tuning of confidence thresholds",
            "Add production monitoring",
            "Gradual rollout",
        ]
    elif overall_score >= 0.7:
        recommendation = "⚠️ NEEDS MINOR FIXES - Address issues before deployment"
        next_steps = [
            "Improve template matching accuracy",
            "Optimize performance if needed",
            "Add more test cases",
        ]
    else:
        recommendation = "❌ NOT READY - Major improvements needed"
        next_steps = [
            "Significant accuracy improvements needed",
            "Performance optimization required",
            "Better edge case handling",
        ]

    print(f"\nRecommendation: {recommendation}")
    print("\nNext Steps:")
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")

    return overall_score, recommendation


def main():
    """Run the comprehensive test."""
    print("🚀 SpygateAI Down Template Detector Test")
    print("=" * 50)

    # Test 1: Template Loading
    detector = test_template_loading()

    # Test 2: Detection Accuracy
    accuracy, avg_confidence = test_detection_accuracy(detector)

    # Test 3: Performance
    avg_time, fps_capability = test_performance(detector)

    # Test 4: Context Awareness
    context_score = test_context_awareness(detector)

    # Test 5: Edge Cases
    edge_score = test_edge_cases(detector)

    # Generate Expert Recommendation
    overall_score, recommendation = generate_recommendation(
        accuracy, avg_confidence, avg_time, context_score, edge_score
    )

    return {
        "overall_score": overall_score,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_time_ms": avg_time,
        "fps_capability": fps_capability,
        "context_score": context_score,
        "edge_score": edge_score,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    results = main()
