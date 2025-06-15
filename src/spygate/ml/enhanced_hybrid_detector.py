"""Enhanced hybrid detector combining YOLO and OpenCV for robust HUD detection."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionRegion:
    """Region of interest for element detection."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    elements: list[dict[str, Any]] = None


class EnhancedHybridDetector:
    """Enhanced hybrid detector combining YOLO and OpenCV approaches.

    Key improvements:
    1. Two-stage detection pipeline:
       - YOLO: Fast, reliable HUD region identification
       - OpenCV: Precise element detection within HUD regions
    2. Adaptive thresholding based on region characteristics
    3. Multi-scale detection for different sized elements
    4. Enhanced triangle detection with geometric validation
    5. Temporal smoothing for stable detections
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the enhanced hybrid detector."""
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize YOLO model with optimization
        optimization_config = OptimizationConfig(
            enable_dynamic_switching=True,
            enable_adaptive_batch_size=True,
            enable_performance_monitoring=True,
            enable_auto_optimization=True,
        )

        self.model = EnhancedYOLOv8(
            model_path=model_path,
            hardware_tier=self.hardware.tier,
            optimization_config=optimization_config,
        )

        # OpenCV detection parameters
        self.triangle_params = {
            "min_area": 50,
            "max_area": 500,
            "angle_threshold": 15,  # degrees
            "aspect_ratio_range": (0.8, 1.2),
        }

        # Detection history for temporal smoothing
        self.detection_history = []
        self.history_size = 5

        logger.info("Enhanced hybrid detector initialized")

    def detect_elements(self, frame: np.ndarray) -> dict[str, Any]:
        """Main detection pipeline combining YOLO and OpenCV."""
        try:
            start_time = time.time()

            # Stage 1: YOLO Detection
            yolo_regions = self._detect_with_yolo(frame)

            # Stage 2: OpenCV Processing
            processed_regions = []
            for region in yolo_regions:
                elements = self._process_region_with_opencv(frame, region)
                region.elements = elements
                processed_regions.append(region)

            # Stage 3: Temporal Smoothing
            smoothed_regions = self._apply_temporal_smoothing(processed_regions)

            # Stage 4: Final Validation
            validated_regions = self._validate_detections(smoothed_regions)

            processing_time = time.time() - start_time

            return {
                "regions": validated_regions,
                "processing_time": processing_time,
                "hardware_tier": self.hardware.tier.name,
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {"regions": [], "error": str(e)}

    def _detect_with_yolo(self, frame: np.ndarray) -> list[DetectionRegion]:
        """Run YOLO detection with optimized settings."""
        results = self.model.detect_hud_elements(frame)

        regions = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls_id in zip(boxes, scores, classes):
                    if cls_id < len(UI_CLASSES):
                        regions.append(
                            DetectionRegion(
                                bbox=tuple(map(int, box)),
                                confidence=float(score),
                                class_name=UI_CLASSES[cls_id],
                            )
                        )

        return regions

    def _process_region_with_opencv(
        self, frame: np.ndarray, region: DetectionRegion
    ) -> list[dict[str, Any]]:
        """Process a detected region with OpenCV for precise element detection."""
        x1, y1, x2, y2 = region.bbox
        roi = frame[y1:y2, x1:x2]

        elements = []

        if region.class_name in ["possession_triangle_area", "territory_triangle_area"]:
            # Enhanced triangle detection
            triangles = self._detect_triangles(roi)
            elements.extend(triangles)

        elif region.class_name == "hud":
            # Enhanced text region detection
            text_regions = self._detect_text_regions(roi)
            elements.extend(text_regions)

        # Adjust coordinates to original frame
        for element in elements:
            if "bbox" in element:
                x, y, w, h = element["bbox"]
                element["bbox"] = (x + x1, y + y1, w, h)

        return elements

    def _detect_triangles(self, roi: np.ndarray) -> list[dict[str, Any]]:
        """Enhanced triangle detection with geometric validation."""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        triangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.triangle_params["min_area"] <= area <= self.triangle_params["max_area"]:
                # Approximate the contour
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 3:  # Triangle
                    # Validate triangle geometry
                    if self._validate_triangle_geometry(approx):
                        x, y, w, h = cv2.boundingRect(approx)
                        triangles.append(
                            {
                                "type": "triangle",
                                "bbox": (x, y, w, h),
                                "contour": approx.tolist(),
                                "confidence": self._calculate_triangle_confidence(approx, area),
                            }
                        )

        return triangles

    def _validate_triangle_geometry(self, triangle) -> bool:
        """Validate triangle geometry using angle and aspect ratio checks."""
        # Calculate angles
        angles = []
        for i in range(3):
            pt1 = triangle[i][0]
            pt2 = triangle[(i + 1) % 3][0]
            pt3 = triangle[(i + 2) % 3][0]

            v1 = pt2 - pt1
            v2 = pt3 - pt1

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(angle)

        # Check if angles are reasonable (close to 60 degrees for equilateral)
        angle_valid = all(
            abs(angle - 60) < self.triangle_params["angle_threshold"] for angle in angles
        )

        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(triangle)
        aspect_ratio = w / h if h != 0 else 0
        ratio_valid = (
            self.triangle_params["aspect_ratio_range"][0]
            <= aspect_ratio
            <= self.triangle_params["aspect_ratio_range"][1]
        )

        return angle_valid and ratio_valid

    def _calculate_triangle_confidence(self, triangle, area: float) -> float:
        """Calculate confidence score for triangle detection."""
        # Normalize area confidence
        area_conf = min(area / self.triangle_params["max_area"], 1.0)

        # Calculate shape confidence based on angles
        angles = []
        for i in range(3):
            pt1 = triangle[i][0]
            pt2 = triangle[(i + 1) % 3][0]
            pt3 = triangle[(i + 2) % 3][0]

            v1 = pt2 - pt1
            v2 = pt3 - pt1

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            angles.append(abs(angle - 60))  # Difference from ideal 60°

        angle_conf = 1.0 - (max(angles) / self.triangle_params["angle_threshold"])

        # Combine confidences
        return (area_conf + angle_conf) / 2

    def _detect_text_regions(self, roi: np.ndarray) -> list[dict[str, Any]]:
        """Detect potential text regions in the HUD."""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # MSER detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

            # Filter by aspect ratio and size
            aspect_ratio = w / h if h > 0 else 0
            if 0.2 <= aspect_ratio <= 15 and w >= 10 and h >= 8:
                text_regions.append({"type": "text_region", "bbox": (x, y, w, h)})

        return text_regions

    def _apply_temporal_smoothing(self, regions: list[DetectionRegion]) -> list[DetectionRegion]:
        """Apply temporal smoothing to reduce detection jitter."""
        self.detection_history.append(regions)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        if not self.detection_history:
            return regions

        # Average detection coordinates over history
        smoothed_regions = []
        for region in regions:
            matching_history = []
            for past_regions in self.detection_history:
                for past_region in past_regions:
                    if (
                        past_region.class_name == region.class_name
                        and self._iou(past_region.bbox, region.bbox) > 0.5
                    ):
                        matching_history.append(past_region)

            if matching_history:
                # Average coordinates
                avg_x1 = sum(r.bbox[0] for r in matching_history) / len(matching_history)
                avg_y1 = sum(r.bbox[1] for r in matching_history) / len(matching_history)
                avg_x2 = sum(r.bbox[2] for r in matching_history) / len(matching_history)
                avg_y2 = sum(r.bbox[3] for r in matching_history) / len(matching_history)

                smoothed_regions.append(
                    DetectionRegion(
                        bbox=(int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)),
                        confidence=region.confidence,
                        class_name=region.class_name,
                        elements=region.elements,
                    )
                )
            else:
                smoothed_regions.append(region)

        return smoothed_regions

    def _validate_detections(self, regions: list[DetectionRegion]) -> list[dict[str, Any]]:
        """Final validation of detected regions."""
        validated = []
        for region in regions:
            # Convert to dict for output
            detection = {
                "class_name": region.class_name,
                "bbox": region.bbox,
                "confidence": region.confidence,
                "elements": region.elements or [],
            }

            # Additional validation based on class
            if region.class_name in ["possession_triangle_area", "territory_triangle_area"]:
                if not region.elements:  # No triangles found
                    continue

                # Keep only highest confidence triangle
                best_triangle = max(
                    region.elements,
                    key=lambda x: x.get("confidence", 0) if x["type"] == "triangle" else 0,
                )
                detection["elements"] = [best_triangle]

            validated.append(detection)

        return validated

    def _iou(self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

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
