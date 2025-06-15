#!/usr/bin/env python3
"""
Test the YOLO-integrated template triangle detector.
Uses YOLO-detected bounding boxes for precise template matching.
"""

from pathlib import Path

import cv2
import numpy as np

from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.ml.template_triangle_detector import YOLOIntegratedTriangleDetector


def test_yolo_integrated_template_matching():
    """Test YOLO-integrated template matching with real extracted templates."""

    # Initialize the enhanced game analyzer with the trained YOLO model
    print("🔄 Initializing Enhanced Game Analyzer with YOLO model...")
    game_analyzer = EnhancedGameAnalyzer()

    # Initialize YOLO-integrated template detector
    templates_dir = Path("templates/triangles")
    debug_dir = Path("debug_template_matching")
    debug_dir.mkdir(exist_ok=True)

    print(f"📁 Loading templates from: {templates_dir}")

    # Check what templates exist
    template_files = list(templates_dir.glob("*.png"))
    print(f"📋 Available template files:")
    for template_file in template_files:
        print(f"   - {template_file.name}")

    detector = YOLOIntegratedTriangleDetector(
        game_analyzer=game_analyzer, templates_dir=templates_dir, debug_output_dir=debug_dir
    )

    # Load test image
    test_image_path = "comprehensive_hud_detections.jpg"
    print(f"📸 Loading test image: {test_image_path}")

    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"❌ Failed to load image: {test_image_path}")
        return

    print(f"📐 Image size: {frame.shape[1]}x{frame.shape[0]}")

    # First, show what YOLO detects
    print("\n🔍 Running YOLO detection...")
    detections = game_analyzer.model.detect(frame)

    print(f"📊 YOLO found {len(detections)} total detections:")
    for i, detection in enumerate(detections, 1):
        bbox = detection["bbox"]
        conf = detection["confidence"]
        class_name = detection["class"]
        print(f"  {i}. {class_name} (conf: {conf:.3f}) bbox: {bbox}")

    # Filter triangle area detections
    triangle_detections = [d for d in detections if "triangle_area" in d["class"]]
    print(f"\n🔺 Triangle area detections: {len(triangle_detections)}")
    for detection in triangle_detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        class_name = detection["class"]
        conf = detection["confidence"]
        print(
            f"  📍 {class_name}: bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) size=({w}x{h}) conf={conf:.3f}"
        )

    # Now run template matching
    print(f"\n🎯 Running YOLO-integrated template matching...")
    matches = detector.detect_triangles_in_yolo_regions(frame)

    print(f"\n📊 Template Matching Results:")
    print(f"   Total matches found: {len(matches)}")

    possession_matches = [m for m in matches if m.triangle_type.value == "possession"]
    territory_matches = [m for m in matches if m.triangle_type.value == "territory"]

    print(f"   Possession triangles: {len(possession_matches)}")
    print(f"   Territory triangles: {len(territory_matches)}")

    if matches:
        print(f"\n🎯 Match details:")
        for i, match in enumerate(matches, 1):
            print(
                f"   {i}. {match.triangle_type.value} {match.direction.value} - conf: {match.confidence:.3f} scale: {match.scale_factor:.2f}x"
            )

    # Save debug visualization
    debug_file = debug_dir / "yolo_vs_template_debug.jpg"
    print(f"📁 Debug visualization saved: {debug_file}")

    print(f"\n✅ YOLO-integrated template matching test complete!")
    print(f"📁 Debug output saved to: {debug_dir}")


if __name__ == "__main__":
    test_yolo_integrated_template_matching()
