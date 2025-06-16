#!/usr/bin/env python3
"""
Test SpygateAI Down Template Detection with Correct Real Templates

This script tests the down template detector using the correct real Madden templates
that were properly cropped from actual gameplay screenshots.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.append("src")

from spygate.ml.down_template_detector import DownDetectionContext, DownTemplateDetector


def load_test_images():
    """Load test images from templates/raw_gameplay/"""
    test_dir = Path("templates/raw_gameplay")
    test_images = {}

    # Map test files to expected down numbers
    test_mapping = {
        "1st_10.png": (1, 10, "1ST"),
        "2nd_7.png": (2, 7, "2ND"),
        "3rd_goal.png": (3, None, "3RD_GOAL"),
        "4th_goal.png": (4, None, "4TH_GOAL"),
    }

    for filename, (down, distance, expected) in test_mapping.items():
        filepath = test_dir / filename
        if filepath.exists():
            image = cv2.imread(str(filepath))
            if image is not None:
                test_images[filename] = {
                    "image": image,
                    "expected_down": down,
                    "expected_distance": distance,
                    "expected_template": expected,
                    "path": str(filepath),
                }
                print(f"✓ Loaded test image: {filename} ({image.shape[1]}x{image.shape[0]})")
            else:
                print(f"✗ Failed to load: {filename}")
        else:
            print(f"✗ File not found: {filename}")

    return test_images


def create_mock_yolo_bbox(image_shape, down_area_coords=None):
    """Create a mock YOLO bounding box for the down_distance_area"""
    height, width = image_shape[:2]

    if down_area_coords:
        x, y, w, h = down_area_coords
        return (x, y, w, h)

    # For small test images (125x50), use the entire image as the down area
    if width <= 150 and height <= 100:
        # These are cropped down regions, use most of the image
        x = 5
        y = 5
        w = width - 10
        h = height - 10
        return (x, y, w, h)

    # Default: assume down area is in upper portion of HUD
    # Based on real Madden coordinates from metadata
    x = int(width * 0.65)  # Around 65% from left
    y = int(height * 0.05)  # Around 5% from top
    w = int(width * 0.15)  # About 15% of width
    h = int(height * 0.08)  # About 8% of height

    return (x, y, w, h)


def test_template_detection():
    """Test the down template detection system"""
    print("=" * 60)
    print("SpygateAI Down Template Detection Test")
    print("=" * 60)

    # Initialize detector
    print("\n1. Initializing DownTemplateDetector...")
    detector = DownTemplateDetector(
        templates_dir=Path("down_templates_real"), debug_output_dir=Path("debug_template_test")
    )

    print(f"   Templates loaded: {len(detector.templates)}")
    for name, template in detector.templates.items():
        size = template["size"]
        print(f"   - {name}: {size[1]}x{size[0]} ({template['situation_type'].value})")

    # Load test images
    print("\n2. Loading test images...")
    test_images = load_test_images()

    if not test_images:
        print("✗ No test images found!")
        return

    print(f"   Test images loaded: {len(test_images)}")

    # Run detection tests
    print("\n3. Running detection tests...")
    results = []
    total_tests = len(test_images)
    correct_detections = 0

    for filename, test_data in test_images.items():
        print(f"\n   Testing: {filename}")

        image = test_data["image"]
        expected_down = test_data["expected_down"]
        expected_template = test_data["expected_template"]

        # Create mock YOLO bbox (simulate YOLO detection)
        bbox = create_mock_yolo_bbox(image.shape)
        print(f"   Mock YOLO bbox: {bbox}")

        # Create context for GOAL situations
        context = None
        if "goal" in filename.lower():
            context = DownDetectionContext(field_position="GOAL", is_goal_line=True)

        # Run detection
        start_time = time.time()
        match = detector.detect_down_in_yolo_region(image, bbox, context)
        detection_time = time.time() - start_time

        # Analyze results
        if match:
            detected_down = match.down
            detected_template = match.template_name
            confidence = match.confidence

            is_correct = detected_down == expected_down
            if is_correct:
                correct_detections += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            print(f"   {status}: Detected {detected_down} (expected {expected_down})")
            print(f"   Template: {detected_template} (confidence: {confidence:.3f})")
            print(f"   Position: {match.position}, Scale: {match.scale_factor:.2f}")
            print(f"   Time: {detection_time*1000:.1f}ms")

            results.append(
                {
                    "filename": filename,
                    "expected": expected_down,
                    "detected": detected_down,
                    "correct": is_correct,
                    "confidence": confidence,
                    "template": detected_template,
                    "time_ms": detection_time * 1000,
                }
            )
        else:
            print(f"   ✗ NO DETECTION")
            results.append(
                {
                    "filename": filename,
                    "expected": expected_down,
                    "detected": None,
                    "correct": False,
                    "confidence": 0.0,
                    "template": None,
                    "time_ms": detection_time * 1000,
                }
            )

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION RESULTS SUMMARY")
    print("=" * 60)

    accuracy = (correct_detections / total_tests) * 100 if total_tests > 0 else 0
    print(f"Overall Accuracy: {correct_detections}/{total_tests} ({accuracy:.1f}%)")

    if results:
        avg_time = np.mean([r["time_ms"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results if r["confidence"] > 0])
        print(f"Average Detection Time: {avg_time:.1f}ms")
        print(f"Average Confidence: {avg_confidence:.3f}")

        print(f"\nFPS Capability: {1000/avg_time:.1f} FPS")

    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result["correct"] else "✗"
        conf_str = f"{result['confidence']:.3f}" if result["confidence"] > 0 else "N/A"
        template_str = result["template"] or "None"
        print(
            f"  {status} {result['filename']}: {result['detected']} (conf: {conf_str}, template: {template_str})"
        )

    return results


def test_template_quality():
    """Test template quality and preprocessing"""
    print("\n" + "=" * 60)
    print("TEMPLATE QUALITY ANALYSIS")
    print("=" * 60)

    templates_dir = Path("down_templates_real")

    for template_file in templates_dir.glob("*.png"):
        template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if template is not None:
            # Basic quality metrics
            if len(template.shape) == 2:
                height, width = template.shape
            else:
                height, width, _ = template.shape
            mean_intensity = np.mean(template)
            std_intensity = np.std(template)

            # Edge detection for sharpness
            edges = cv2.Canny(template, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)

            print(f"{template_file.name}:")
            print(f"  Size: {width}x{height}")
            print(f"  Mean intensity: {mean_intensity:.1f}")
            print(f"  Std intensity: {std_intensity:.1f}")
            print(f"  Edge density: {edge_density:.4f}")
            print()


if __name__ == "__main__":
    # Run template quality analysis
    test_template_quality()

    # Run detection tests
    results = test_template_detection()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
