#!/usr/bin/env python3
"""
Hybrid Triangle Detection System
================================
Uses YOLO for HUD detection + OpenCV for triangle detection within HUD regions
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class HybridTriangleDetector:
    """Hybrid detector using YOLO for HUD + OpenCV for triangles."""

    def __init__(self, yolo_model_path):
        """Initialize hybrid detector."""
        self.yolo_model = YOLO(yolo_model_path)

        # Triangle detection parameters
        self.color_ranges = self._define_color_ranges()

    def _define_color_ranges(self):
        """Define color ranges for triangle detection."""
        return {
            "orange": {
                "lower": np.array([10, 100, 100], dtype=np.uint8),
                "upper": np.array([25, 255, 255], dtype=np.uint8),
            },
            "purple": {
                "lower": np.array([120, 50, 50], dtype=np.uint8),
                "upper": np.array([150, 255, 255], dtype=np.uint8),
            },
            "white": {
                "lower": np.array([0, 0, 200], dtype=np.uint8),
                "upper": np.array([180, 30, 255], dtype=np.uint8),
            },
        }

    def detect_hud_regions(self, image, conf_threshold=0.3):
        """Use YOLO to detect HUD regions."""
        results = self.yolo_model(image, conf=conf_threshold, verbose=False)

        hud_regions = []
        if results and len(results) > 0:
            detections = results[0].boxes
            if detections is not None:
                for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
                    class_id = int(cls.item())
                    class_name = self.yolo_model.names[class_id]

                    if class_name == "hud" and conf.item() > conf_threshold:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        hud_regions.append({"bbox": (x1, y1, x2, y2), "confidence": conf.item()})

        return hud_regions

    def detect_triangles_in_region(self, image, hud_bbox):
        """Detect triangles within a HUD region using OpenCV."""
        x1, y1, x2, y2 = hud_bbox
        hud_roi = image[y1:y2, x1:x2]

        if hud_roi.size == 0:
            return []

        triangles = []

        # Convert to HSV for color detection
        hsv_roi = cv2.cvtColor(hud_roi, cv2.COLOR_BGR2HSV)

        # Method 1: Color-based detection
        triangles.extend(self._detect_by_color(hud_roi, hsv_roi, x1, y1))

        # Method 2: Edge and contour detection
        triangles.extend(self._detect_by_contours(hud_roi, x1, y1))

        return triangles

    def _detect_by_color(self, bgr_roi, hsv_roi, offset_x, offset_y):
        """Detect triangles by color filtering."""
        triangles = []

        for color_name, color_range in self.color_ranges.items():
            # Create mask for color
            mask = cv2.inRange(hsv_roi, color_range["lower"], color_range["upper"])

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by area (triangles should be small)
                area = cv2.contourArea(contour)
                if 5 < area < 200:  # Smaller area range for triangles
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Adjust coordinates to full image
                    abs_x = x + offset_x
                    abs_y = y + offset_y

                    triangles.append(
                        {
                            "bbox": (abs_x, abs_y, abs_x + w, abs_y + h),
                            "type": self._classify_triangle_by_position(abs_x, bgr_roi.shape[1]),
                            "method": f"color_{color_name}",
                            "confidence": 0.8,
                        }
                    )

        return triangles

    def _detect_by_contours(self, roi, offset_x, offset_y):
        """Detect triangles using contour analysis."""
        triangles = []

        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Edge detection with different thresholds
        edges = cv2.Canny(gray_roi, 30, 100)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly triangular (3-4 vertices)
            if 3 <= len(approx) <= 4:
                area = cv2.contourArea(contour)
                if 5 < area < 200:  # Size filter
                    x, y, w, h = cv2.boundingRect(contour)

                    # Aspect ratio check (triangles should be roughly square)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:
                        # Adjust coordinates to full image
                        abs_x = x + offset_x
                        abs_y = y + offset_y

                        triangles.append(
                            {
                                "bbox": (abs_x, abs_y, abs_x + w, abs_y + h),
                                "type": self._classify_triangle_by_position(abs_x, roi.shape[1]),
                                "method": "contour",
                                "confidence": 0.7,
                            }
                        )

        return triangles

    def _classify_triangle_by_position(self, x, roi_width):
        """Classify triangle type based on position in HUD."""
        # Left side = possession triangle (between team abbreviations/scores)
        if x < roi_width * 0.5:
            return "possession_triangle"  # Points to team with ball possession
        # Right side = territory triangle (next to field position)
        else:
            return "territory_triangle"  # ▲=in opponent's territory, ▼=in own territory

    def detect(self, image):
        """Main detection method combining YOLO + OpenCV."""
        start_time = time.time()

        # Step 1: Detect HUD regions with YOLO
        hud_regions = self.detect_hud_regions(image, conf_threshold=0.3)

        all_triangles = []

        # Step 2: Detect triangles in each HUD region with OpenCV
        for hud_region in hud_regions:
            triangles = self.detect_triangles_in_region(image, hud_region["bbox"])
            all_triangles.extend(triangles)

        detection_time = time.time() - start_time

        return {
            "hud_regions": hud_regions,
            "triangles": all_triangles,
            "detection_time": detection_time,
        }


def test_hybrid_detector():
    """Test the hybrid detector on the sample image."""
    print("🧪 Testing Hybrid Triangle Detection System")
    print("=" * 50)

    # Load model and image
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    test_image = "triangle_visualization_3.jpg"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"✅ Loading hybrid detector...")
    detector = HybridTriangleDetector(model_path)

    print(f"✅ Loading test image: {test_image}")
    image = cv2.imread(test_image)

    print(f"🔍 Running hybrid detection...")
    results = detector.detect(image)

    print(f"\n📊 RESULTS:")
    print(f"  Detection time: {results['detection_time']:.3f}s")
    print(f"  HUD regions found: {len(results['hud_regions'])}")
    print(f"  Triangles found: {len(results['triangles'])}")

    # Print HUD details
    if results["hud_regions"]:
        print(f"\n🎮 HUD REGIONS:")
        for i, hud in enumerate(results["hud_regions"]):
            print(f"  {i+1}. HUD confidence: {hud['confidence']:.3f}")

    # Print triangle details
    if results["triangles"]:
        print(f"\n🔺 TRIANGLE DETAILS:")
        for i, triangle in enumerate(results["triangles"]):
            print(
                f"  {i+1}. {triangle['type']} via {triangle['method']}: {triangle['confidence']:.3f}"
            )
    else:
        print(f"\n❌ No triangles detected with OpenCV methods")

    # Draw results
    display_image = image.copy()

    # Draw HUD regions
    for hud in results["hud_regions"]:
        x1, y1, x2, y2 = hud["bbox"]
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display_image,
            f"HUD: {hud['confidence']:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Draw triangles
    for triangle in results["triangles"]:
        x1, y1, x2, y2 = triangle["bbox"]
        color = (255, 165, 0) if triangle["type"] == "possession_triangle" else (128, 0, 128)
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

        label = f"{triangle['type'][:4]}: {triangle['confidence']:.2f}"
        cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save result
    output_path = "hybrid_detection_result.jpg"
    cv2.imwrite(output_path, display_image)
    print(f"✅ Result saved to: {output_path}")


if __name__ == "__main__":
    test_hybrid_detector()
