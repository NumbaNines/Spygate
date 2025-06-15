"""
Test the trained 8-class YOLOv8 model on real images.
Visualizes detections with bounding boxes and confidence scores.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def test_8class_model():
    """Test the 8-class model on sample images."""

    print("🚀 Testing 8-Class YOLOv8 Model")
    print("=" * 50)

    # Load the trained model
    model_path = "hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt"
    model = YOLO(model_path)

    print(f"✅ Model loaded: {model_path}")
    print(f"📊 Model size: {Path(model_path).stat().st_size / (1024*1024):.1f} MB")

    # Class names for the 8-class model
    class_names = [
        "hud",  # 0 - Main HUD bar
        "possession_triangle_area",  # 1 - Left triangle (ball possession)
        "territory_triangle_area",  # 2 - Right triangle (field territory)
        "preplay_indicator",  # 3 - Pre-play UI element
        "play_call_screen",  # 4 - Play call screen overlay
        "down_distance_area",  # 5 - Down & distance text (NEW)
        "game_clock_area",  # 6 - Game clock display (NEW)
        "play_clock_area",  # 7 - Play clock display (NEW)
    ]

    # Test images directory
    test_images_dir = Path("hud_region_training_8class/datasets_8class/train/images")
    test_images = list(test_images_dir.glob("*.png"))[:5]  # Test first 5 images

    if not test_images:
        print("❌ No test images found!")
        return

    print(f"🖼️ Testing on {len(test_images)} images...")

    # Process each test image
    for i, img_path in enumerate(test_images):
        print(f"\n📸 Processing: {img_path.name}")

        # Run inference
        results = model(str(img_path), conf=0.3, verbose=False)

        # Get the first result
        result = results[0]

        # Load original image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw detections
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            print(f"   🎯 Found {len(boxes)} detections:")

            for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = class_names[cls_id]

                print(f"      {j+1}. {class_name}: {conf:.3f}")

                # Draw bounding box
                color = plt.cm.tab10(cls_id)[:3]  # Get color for class
                color = tuple(int(c * 255) for c in color)

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    img_rgb, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1
                )
                cv2.putText(
                    img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        else:
            print("   ❌ No detections found")

        # Save result
        output_path = f"test_results_8class_{i+1}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"   💾 Saved: {output_path}")

    print(f"\n🎉 Testing complete! Check the test_results_8class_*.jpg files")
    print("🔍 Look for:")
    print("   • HUD detection (main bar)")
    print("   • Triangle areas (possession & territory)")
    print("   • NEW: Down/distance area")
    print("   • NEW: Game clock area")
    print("   • NEW: Play clock area")


if __name__ == "__main__":
    test_8class_model()
