#!/usr/bin/env python3
"""
SpygateAI Down Template Detector

Production-ready down detection using YOLO-guided template matching with PaddleOCR fallback.
Uses TM_CCOEFF_NORMED (optimal for text) and handles both Normal + GOAL templates.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SituationType(Enum):
    """Situation types for template selection."""

    NORMAL = "normal"
    GOAL = "goal"


@dataclass
class DownTemplateMatch:
    """A down template match result."""

    down: int  # 1, 2, 3, or 4
    distance: Optional[int]  # Yards to go (None for GOAL)
    confidence: float  # 0.0 - 1.0
    template_name: str  # Template used
    situation_type: SituationType  # Normal or GOAL
    position: tuple[int, int]  # Match location (x, y)
    bounding_box: tuple[int, int, int, int]  # x, y, w, h
    scale_factor: float  # Template scale used
    method_used: str  # "template" or "ocr_fallback"


class DownTemplateDetector:
    """
    Production-ready down template detector using YOLO-guided template matching.

    Key Features:
    - TM_CCOEFF_NORMED matching (optimal for text)
    - Multi-scale template matching
    - Dual template sets (Normal + GOAL)
    - PaddleOCR fallback integration
    - SpygateAI preprocessing pipeline
    """

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        debug_output_dir: Optional[Path] = None,
        ocr_engine: Optional[Any] = None,
    ):
        """Initialize down template detector."""
        self.templates_dir = templates_dir or Path("down_templates_real")
        self.debug_output_dir = debug_output_dir
        self.ocr_engine = ocr_engine

        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # Template matching parameters (optimized for down detection)
        self.SCALE_FACTORS = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]  # Focused range for HUD text
        self.MIN_MATCH_CONFIDENCE = 0.15  # ⭐ LOWERED: Real templates should match better
        self.TEMPLATE_METHOD = cv2.TM_CCOEFF_NORMED  # ⭐ BEST for text matching
        self.NMS_OVERLAP_THRESHOLD = 0.6

        # Load templates
        self.templates = {}
        self.load_templates()

        logger.info(f"DownTemplateDetector initialized with {len(self.templates)} templates")

    def load_templates(self) -> None:
        """Load down templates from disk."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        # Load normal templates
        normal_templates = ["1ST.png", "2ND.png", "3RD.png", "4TH.png"]
        for template_file in normal_templates:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                self._load_single_template(template_path, SituationType.NORMAL)

        # Load GOAL templates
        goal_templates = ["1ST_GOAL.png", "2ND_GOAL.png", "3RD_GOAL.png", "4TH_GOAL.png"]
        for template_file in goal_templates:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                self._load_single_template(template_path, SituationType.GOAL)

        logger.info(f"Loaded {len(self.templates)} down templates")

    def _load_single_template(self, template_path: Path, situation_type: SituationType) -> None:
        """Load a single template file."""
        try:
            # Load template image (grayscale for matching)
            template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                logger.warning(f"Failed to load template: {template_path}")
                return

            # Parse template name
            template_name = template_path.stem
            down_num = self._parse_down_from_name(template_name)

            if down_num is None:
                logger.warning(f"Could not parse down number from: {template_name}")
                return

            # Store template
            self.templates[template_name] = {
                "image": template_img,
                "down": down_num,
                "situation_type": situation_type,
                "size": template_img.shape,
                "path": str(template_path),
            }

            logger.debug(
                f"Loaded template: {template_name} ({template_img.shape[1]}x{template_img.shape[0]})"
            )

        except Exception as e:
            logger.error(f"Error loading template {template_path}: {e}")

    def _parse_down_from_name(self, template_name: str) -> Optional[int]:
        """Parse down number from template name."""
        name_upper = template_name.upper()

        if "1ST" in name_upper:
            return 1
        elif "2ND" in name_upper:
            return 2
        elif "3RD" in name_upper:
            return 3
        elif "4TH" in name_upper:
            return 4

        return None

    def detect_down_in_yolo_region(
        self,
        frame: np.ndarray,
        down_distance_bbox: tuple[int, int, int, int],
        is_goal_situation: bool = False,
    ) -> Optional[DownTemplateMatch]:
        """
        Detect down using YOLO-guided template matching.

        Args:
            frame: Full game frame
            down_distance_bbox: YOLO-detected down_distance_area bbox (x1, y1, x2, y2)
            is_goal_situation: True if this is a goal line situation

        Returns:
            Best down template match or None
        """
        try:
            # Extract ROI from YOLO detection
            x1, y1, x2, y2 = map(int, down_distance_bbox)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                logger.warning("Empty ROI from YOLO detection")
                return None

            # Apply SpygateAI's optimal preprocessing
            preprocessed_roi = self._apply_spygate_preprocessing(roi)

            # Template matching with context-aware template selection
            template_matches = self._match_templates_with_context(
                preprocessed_roi, (x1, y1), is_goal_situation
            )

            if not template_matches:
                # Fallback to PaddleOCR if available
                if self.ocr_engine:
                    return self._ocr_fallback(roi, (x1, y1))
                return None

            # Apply NMS to remove overlapping detections
            nms_matches = self._apply_nms(template_matches)

            # Select best match
            best_match = max(nms_matches, key=lambda x: x.confidence) if nms_matches else None

            # Debug visualization
            if self.debug_output_dir and best_match:
                self._create_debug_visualization(frame, roi, best_match, down_distance_bbox)

            return best_match

        except Exception as e:
            logger.error(f"Error in down detection: {e}")
            return None

    def _apply_spygate_preprocessing(self, roi: np.ndarray) -> np.ndarray:
        """
        Apply SpygateAI's optimal OCR preprocessing pipeline.
        Score: 0.939 (Scale=3.5x, CLAHE clip=1.0, Gamma=0.8, etc.)
        """
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            # 1. LANCZOS4 scaling - Scale=3.5x
            height, width = gray.shape
            scaled = cv2.resize(
                gray, (int(width * 3.5), int(height * 3.5)), interpolation=cv2.INTER_LANCZOS4
            )

            # 2. CLAHE - clip=1.0, grid=(4,4)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
            clahe_applied = clahe.apply(scaled)

            # 3. Gamma correction - Gamma=0.8
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
                "uint8"
            )
            gamma_corrected = cv2.LUT(clahe_applied, table)

            # 4. Gaussian blur - (3,3)
            blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

            # 5. Adaptive threshold - mean, block=13, C=3
            threshold = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 3
            )

            # 6. Morphological closing - kernel=(3,3)
            kernel = np.ones((3, 3), np.uint8)
            morphological = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            return morphological

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return roi

    def _match_templates_with_context(
        self, preprocessed_roi: np.ndarray, offset: tuple[int, int], is_goal_situation: bool
    ) -> list[DownTemplateMatch]:
        """Match templates with context-aware selection."""
        matches = []

        # Select appropriate template set based on context
        if is_goal_situation:
            # Prioritize GOAL templates but also try normal ones
            template_priority = ["GOAL", "normal"]
        else:
            # Prioritize normal templates but also try GOAL ones
            template_priority = ["normal", "GOAL"]

        for priority_type in template_priority:
            for template_name, template_data in self.templates.items():
                situation_type = template_data["situation_type"]

                # Skip if not matching current priority
                if (priority_type == "GOAL" and situation_type != SituationType.GOAL) or (
                    priority_type == "normal" and situation_type != SituationType.NORMAL
                ):
                    continue

                template_img = template_data["image"]
                down = template_data["down"]

                # Multi-scale template matching
                for scale in self.SCALE_FACTORS:
                    scaled_matches = self._match_single_template_scaled(
                        preprocessed_roi,
                        template_img,
                        template_name,
                        down,
                        situation_type,
                        scale,
                        offset,
                    )
                    matches.extend(scaled_matches)

            # If we found good matches with priority templates, stop
            if matches and max(matches, key=lambda x: x.confidence).confidence > 0.75:
                break

        return matches

    def _match_single_template_scaled(
        self,
        roi: np.ndarray,
        template: np.ndarray,
        template_name: str,
        down: int,
        situation_type: SituationType,
        scale: float,
        offset: tuple[int, int],
    ) -> list[DownTemplateMatch]:
        """Match a single template at a specific scale using TM_CCOEFF_NORMED."""
        matches = []

        try:
            # Scale the template
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)

            # Skip if scaled template is larger than ROI
            if scaled_w > roi.shape[1] or scaled_h > roi.shape[0]:
                return matches

            # Skip if scaled template is too small
            if scaled_w < 10 or scaled_h < 10:
                return matches

            scaled_template = cv2.resize(template, (scaled_w, scaled_h))

            # Template matching using TM_CCOEFF_NORMED (⭐ BEST for text)
            result = cv2.matchTemplate(roi, scaled_template, self.TEMPLATE_METHOD)

            # Find matches above threshold
            locations = np.where(result >= self.MIN_MATCH_CONFIDENCE)

            for pt in zip(*locations[::-1]):  # Switch x and y
                confidence = result[pt[1], pt[0]]

                # Determine distance based on situation type
                distance = 0 if situation_type == SituationType.GOAL else None

                match = DownTemplateMatch(
                    down=down,
                    distance=distance,
                    confidence=confidence,
                    template_name=template_name,
                    situation_type=situation_type,
                    position=(pt[0] + offset[0], pt[1] + offset[1]),
                    bounding_box=(pt[0] + offset[0], pt[1] + offset[1], scaled_w, scaled_h),
                    scale_factor=scale,
                    method_used="template",
                )
                matches.append(match)

        except Exception as e:
            logger.error(f"Error matching template {template_name} at scale {scale}: {e}")

        return matches

    def _apply_nms(self, matches: list[DownTemplateMatch]) -> list[DownTemplateMatch]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not matches:
            return []

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        final_matches = []

        for match in matches:
            # Check if this match overlaps significantly with any accepted match
            is_duplicate = False

            for accepted in final_matches:
                overlap = self._calculate_overlap(match.bounding_box, accepted.bounding_box)
                if overlap > self.NMS_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_matches.append(match)

        return final_matches

    def _calculate_overlap(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        x1_2, y1_2, w2, h2 = box2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _ocr_fallback(
        self, roi: np.ndarray, offset: tuple[int, int]
    ) -> Optional[DownTemplateMatch]:
        """Fallback to PaddleOCR when template matching fails."""
        try:
            if not self.ocr_engine:
                return None

            # Use PaddleOCR to extract text
            ocr_results = self.ocr_engine.extract_text(roi)

            if not ocr_results:
                return None

            # Parse OCR results for down/distance
            for result in ocr_results:
                text = result.get("text", "").strip()
                confidence = result.get("confidence", 0.0)

                # Parse down from text
                down, distance = self._parse_down_distance_from_text(text)

                if down is not None:
                    # Determine situation type
                    situation_type = SituationType.GOAL if distance == 0 else SituationType.NORMAL

                    return DownTemplateMatch(
                        down=down,
                        distance=distance,
                        confidence=confidence * 0.8,  # Reduce confidence for OCR
                        template_name="ocr_fallback",
                        situation_type=situation_type,
                        position=offset,
                        bounding_box=(offset[0], offset[1], roi.shape[1], roi.shape[0]),
                        scale_factor=1.0,
                        method_used="ocr_fallback",
                    )

        except Exception as e:
            logger.error(f"Error in OCR fallback: {e}")

        return None

    def _parse_down_distance_from_text(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse down and distance from OCR text."""
        import re

        text_upper = text.upper()

        # Common patterns
        patterns = [
            r"(\d+)(?:ST|ND|RD|TH)?\s*&\s*(\d+)",  # "1ST & 10", "3 & 8"
            r"(\d+)(?:ST|ND|RD|TH)?\s*&\s*GOAL",  # "1ST & GOAL"
            r"(\d+)(?:ST|ND|RD|TH)?\s*&\s*G",  # "1ST & G"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                try:
                    down = int(match.group(1))
                    if 1 <= down <= 4:
                        if "GOAL" in text_upper or "G" in match.group(2):
                            return down, 0  # Goal line
                        else:
                            distance = int(match.group(2))
                            return down, distance
                except (ValueError, IndexError):
                    continue

        return None, None

    def _create_debug_visualization(
        self,
        frame: np.ndarray,
        roi: np.ndarray,
        match: DownTemplateMatch,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """Create debug visualization showing the detection."""
        try:
            debug_frame = frame.copy()

            # Draw YOLO bbox
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_frame,
                "YOLO: down_distance_area",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Draw template match
            x, y, w, h = match.bounding_box
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Add match info
            info_text = f"{match.template_name} ({match.confidence:.3f})"
            cv2.putText(
                debug_frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            # Save debug image
            debug_path = self.debug_output_dir / f"down_detection_debug_{match.template_name}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)

        except Exception as e:
            logger.error(f"Error creating debug visualization: {e}")


def create_down_template_detector(ocr_engine=None) -> DownTemplateDetector:
    """Factory function to create a down template detector with PaddleOCR."""
    return DownTemplateDetector(
        templates_dir=Path("down_templates_real"),
        debug_output_dir=Path("debug/down_detection"),
        ocr_engine=ocr_engine,
    )


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = create_down_template_detector()
    print(f"✅ Down Template Detector created with {len(detector.templates)} templates")

    # List loaded templates
    for name, data in detector.templates.items():
        print(f"📏 {name}: {data['size']} ({data['situation_type'].value})")
