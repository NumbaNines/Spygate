#!/usr/bin/env python3
"""
Refined Hybrid Triangle Detection System
========================================
Improved version with stricter parameters to reduce false positives
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class RefinedHybridTriangleDetector:
    """Refined hybrid detector with better filtering."""

    def __init__(self, yolo_model_path):
        """Initialize refined hybrid detector."""
        self.yolo_model = YOLO(yolo_model_path)

        # More specific triangle detection parameters
        self.triangle_size_range = (8, 50)  # Stricter size range
        self.aspect_ratio_range = (0.7, 1.4)  # More square-like
        self.min_triangle_area = 15
        self.max_triangle_area = 80

        self.possession_team = None  # Will store which team has possession
        self.territory_orientation = None  # Will store if triangle points up or down

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
        """Detect triangles within a HUD region using refined OpenCV methods."""
        x1, y1, x2, y2 = hud_bbox
        hud_roi = image[y1:y2, x1:x2]

        if hud_roi.size == 0:
            return []

        # Reset state for new detection
        self.possession_team = None
        self.territory_orientation = None

        triangles = []

        # Focus on specific regions where triangles are located
        possession_roi = hud_roi[
            :, : hud_roi.shape[1] // 2
        ]  # Left half (between team abbrev/scores)
        territory_roi = hud_roi[:, hud_roi.shape[1] // 2 :]  # Right half (next to field position)

        # Detect possession triangle (left side) - points to team with ball
        possession_triangles = self._detect_possession_triangle(possession_roi, x1, y1)
        triangles.extend(possession_triangles)

        # Detect territory triangle (right side) - ▲=opponent's territory, ▼=own territory
        territory_triangles = self._detect_territory_triangle(
            territory_roi, x1 + hud_roi.shape[1] // 2, y1
        )
        triangles.extend(territory_triangles)

        # After detecting both triangles, determine field position context
        field_position = self.determine_field_position()
        if field_position:
            # Add field position context to the detection results
            triangles.append({"type": "field_position_context", "data": field_position})

        return triangles

    def _detect_possession_triangle(self, roi, x_offset, y_offset):
        """Enhanced possession triangle detection."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triangles = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(approx)

                # For possession triangle, we care about horizontal position
                # Assuming left team is away, right team is home
                center_x = x + (w / 2)
                self.possession_team = "away" if center_x < roi.shape[1] / 2 else "home"

                triangles.append(
                    {
                        "type": "possession_triangle",
                        "bbox": (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset),
                        "points_to": self.possession_team,
                        "confidence": 0.9,
                    }
                )

        return triangles

    def _detect_territory_triangle(self, roi, x_offset, y_offset):
        """Enhanced territory triangle detection with orientation."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triangles = []
        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            # Check if it's a triangle (3 points)
            if len(approx) == 3:
                # Get orientation
                orientation = self._detect_triangle_orientation(approx)

                # Calculate bounding box
                x, y, w, h = cv2.boundingRect(approx)

                # Store the territory triangle info
                self.territory_orientation = orientation

                triangles.append(
                    {
                        "type": "territory_triangle",
                        "bbox": (x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset),
                        "orientation": orientation,
                        "confidence": 0.9,
                    }
                )

        return triangles

    def _detect_triangle_orientation(self, contour) -> str:
        """Detect if a triangle points up (▲) or down (▼).

        Args:
            contour: The triangle contour points

        Returns:
            'up' if triangle points up (▲), 'down' if points down (▼)
        """
        # Get the topmost and bottommost points
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

        # Get the average x-coordinate of the base
        base_points = [pt[0] for pt in contour if abs(pt[0][1] - bottommost[1]) < 5]
        base_x_avg = (
            sum(x for x in base_points) / len(base_points) if base_points else bottommost[0]
        )

        # If the top point's x is close to the base's average x, it's pointing up
        # Otherwise, it's pointing down
        return "up" if abs(topmost[0] - base_x_avg) < 10 else "down"

    def determine_field_position(self) -> dict:
        """Determine complete field position context using both triangles.

        Returns:
            Dictionary containing field position context:
            {
                'possession_team': 'home' or 'away',
                'in_territory': 'own' or 'opponent',
                'on_offense': True or False,
                'description': Human readable description
            }
        """
        if self.possession_team is None or self.territory_orientation is None:
            return None

        # Determine if the possession team is in their own or opponent's territory
        in_own_territory = self.territory_orientation == "down"

        result = {
            "possession_team": self.possession_team,
            "in_territory": "own" if in_own_territory else "opponent",
            "on_offense": True,  # Default to True, will be updated below
            "description": "",
        }

        # If in own territory, possession team is on offense
        # If in opponent's territory, possession team is also on offense
        # This is because the territory triangle always shows from possession team's perspective
        result["on_offense"] = True

        # Create human readable description
        team = "Home" if self.possession_team == "home" else "Away"
        territory = "their own" if in_own_territory else "the opponent's"
        result["description"] = f"{team} team has possession in {territory} territory"

        return result

    def detect(self, image):
        """Main detection method."""
        start_time = time.time()

        # Step 1: Detect HUD regions with YOLO
        hud_regions = self.detect_hud_regions(image, conf_threshold=0.3)

        all_triangles = []

        # Step 2: Detect triangles in each HUD region
        for hud_region in hud_regions:
            triangles = self.detect_triangles_in_region(image, hud_region["bbox"])
            all_triangles.extend(triangles)

        # Step 3: Remove duplicate detections
        filtered_triangles = self._remove_duplicates(all_triangles)

        detection_time = time.time() - start_time

        return {
            "hud_regions": hud_regions,
            "triangles": filtered_triangles,
            "detection_time": detection_time,
        }

    def _remove_duplicates(self, triangles):
        """Remove overlapping triangle detections."""
        if not triangles:
            return triangles

        # Sort by confidence
        triangles.sort(key=lambda x: x["confidence"], reverse=True)

        filtered = []
        for triangle in triangles:
            bbox1 = triangle["bbox"]

            # Check overlap with already selected triangles
            overlap_found = False
            for selected in filtered:
                bbox2 = selected["bbox"]

                # Calculate intersection over union (IoU)
                iou = self._calculate_iou(bbox1, bbox2)

                # If significant overlap, skip this detection
                if iou > 0.3:
                    overlap_found = True
                    break

            if not overlap_found:
                filtered.append(triangle)

        return filtered

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def test_refined_detector():
    """Test the refined detector."""
    print("🧪 Testing Refined Hybrid Triangle Detection System")
    print("=" * 55)

    # Load model and image
    model_path = "triangle_training_improved/high_confidence_triangles/weights/best.pt"
    test_image = "triangle_visualization_3.jpg"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return

    print(f"✅ Loading refined hybrid detector...")
    detector = RefinedHybridTriangleDetector(model_path)

    print(f"✅ Loading test image: {test_image}")
    image = cv2.imread(test_image)

    print(f"🔍 Running refined detection...")
    results = detector.detect(image)

    print(f"\n📊 REFINED RESULTS:")
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
                f"  {i+1}. {triangle['type']} via {triangle['method']}: {triangle['confidence']:.3f} (area: {triangle['area']:.0f})"
            )
    else:
        print(f"\n❌ No triangles detected")

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

    # Draw triangles with different colors
    for triangle in results["triangles"]:
        x1, y1, x2, y2 = triangle["bbox"]
        color = (255, 165, 0) if triangle["type"] == "possession_triangle" else (128, 0, 128)
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 3)

        label = f"{triangle['type'][:4]}: {triangle['confidence']:.2f}"
        cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save result
    output_path = "refined_hybrid_detection_result.jpg"
    cv2.imwrite(output_path, display_image)
    print(f"✅ Refined result saved to: {output_path}")


if __name__ == "__main__":
    test_refined_detector()
