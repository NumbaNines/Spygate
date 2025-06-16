#!/usr/bin/env python3
"""
Content-Aware Down/Distance Detection System
Combines template matching for raw gameplay with OCR fallback for streamer content.
"""

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of video content we can detect."""

    RAW_GAMEPLAY = "raw_gameplay"
    STREAMER_CONTENT = "streamer_content"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result from down/distance detection."""

    down: Optional[int] = None
    distance: Optional[int] = None
    confidence: float = 0.0
    method_used: str = "unknown"
    raw_text: str = ""
    content_type: ContentType = ContentType.UNKNOWN


@dataclass
class TemplateMatch:
    """Template matching result."""

    down: int
    distance: int
    confidence: float
    location: tuple[int, int]
    template_name: str


class ContentTypeDetector:
    """Detects whether content is raw gameplay or streamer content."""

    def __init__(self):
        self.detection_cache = {}

    def detect_content_type(self, frame: np.ndarray, hud_region: np.ndarray) -> ContentType:
        """
        Detect content type based on visual indicators.
        """
        # Cache key based on frame characteristics
        cache_key = self._get_frame_signature(frame)
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]

        indicators = self._analyze_content_indicators(frame, hud_region)
        content_type = self._classify_content_type(indicators)

        # Cache result
        self.detection_cache[cache_key] = content_type

        # Keep cache size manageable
        if len(self.detection_cache) > 100:
            oldest_keys = list(self.detection_cache.keys())[:50]
            for key in oldest_keys:
                del self.detection_cache[key]

        return content_type

    def _get_frame_signature(self, frame: np.ndarray) -> str:
        """Generate a signature for frame caching."""
        h, w = frame.shape[:2]
        mean_val = np.mean(frame)
        return f"{w}x{h}_{mean_val:.1f}"

    def _analyze_content_indicators(
        self, frame: np.ndarray, hud_region: np.ndarray
    ) -> dict[str, float]:
        """Analyze visual indicators to determine content type."""
        indicators = {}

        # Check for webcam indicators
        indicators["webcam_score"] = self._detect_webcam_indicators(frame)

        # Check for stream overlay elements
        indicators["overlay_score"] = self._detect_stream_overlays(frame)

        # Check HUD region consistency
        indicators["hud_consistency"] = self._analyze_hud_consistency(hud_region)

        # Check for chat/donation overlays
        indicators["chat_overlay_score"] = self._detect_chat_overlays(frame)

        # Analyze aspect ratio and resolution patterns
        indicators["resolution_score"] = self._analyze_resolution_patterns(frame)

        return indicators

    def _detect_webcam_indicators(self, frame: np.ndarray) -> float:
        """Detect webcam presence indicators."""
        score = 0.0
        h, w = frame.shape[:2]

        # Check corners for rounded regions (common webcam placement)
        corner_regions = [
            frame[0 : h // 4, 0 : w // 4],  # Top-left
            frame[0 : h // 4, 3 * w // 4 : w],  # Top-right
            frame[3 * h // 4 : h, 0 : w // 4],  # Bottom-left
            frame[3 * h // 4 : h, 3 * w // 4 : w],  # Bottom-right
        ]

        for region in corner_regions:
            # Look for circular/rounded shapes
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100
            )

            if circles is not None:
                score += 0.3  # Found circular shapes

            # Check for skin tone colors (webcam indicator)
            if len(region.shape) == 3:
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                # Skin tone range in HSV
                skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
                skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                if skin_ratio > 0.1:  # 10% skin tone pixels
                    score += 0.2

        return min(score, 1.0)

    def _detect_stream_overlays(self, frame: np.ndarray) -> float:
        """Detect stream overlay elements."""
        score = 0.0

        # Look for text overlays
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        # Find contours that might be text
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_like_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Text-like characteristics
            if 2 < aspect_ratio < 10 and 10 < w < 200 and 5 < h < 30:
                text_like_regions += 1

        # More text regions = more likely to be streamer content
        if text_like_regions > 20:
            score += 0.4
        elif text_like_regions > 10:
            score += 0.2

        return min(score, 1.0)

    def _analyze_hud_consistency(self, hud_region: np.ndarray) -> float:
        """Analyze HUD region for consistency."""
        if hud_region.size == 0:
            return 0.0

        # Check for consistent edges and shapes
        gray = (
            cv2.cvtColor(hud_region, cv2.COLOR_BGR2GRAY)
            if len(hud_region.shape) == 3
            else hud_region
        )
        edges = cv2.Canny(gray, 50, 150)

        # Raw gameplay has very clean, consistent edges
        edge_density = np.sum(edges > 0) / edges.size

        # Raw gameplay typically has 0.05-0.15 edge density
        if 0.05 <= edge_density <= 0.15:
            return 0.8  # High consistency
        elif 0.02 <= edge_density <= 0.25:
            return 0.5  # Medium consistency
        else:
            return 0.2  # Low consistency

    def _detect_chat_overlays(self, frame: np.ndarray) -> float:
        """Detect chat or donation overlays."""
        score = 0.0
        h, w = frame.shape[:2]

        # Check right side for chat
        right_region = frame[:, 3 * w // 4 :]

        # Look for scrolling text patterns
        gray = (
            cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY)
            if len(right_region.shape) == 3
            else right_region
        )

        # Detect horizontal lines (chat messages)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

        line_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
        if line_density > 0.01:
            score += 0.3

        return min(score, 1.0)

    def _analyze_resolution_patterns(self, frame: np.ndarray) -> float:
        """Analyze resolution patterns."""
        h, w = frame.shape[:2]

        # Common raw gameplay resolutions
        raw_resolutions = [
            (1920, 1080),
            (1280, 720),
            (2560, 1440),
            (3840, 2160),
            (1600, 900),
            (1366, 768),
            (1440, 900),
        ]

        # Check if current resolution matches common raw gameplay resolutions
        for res_w, res_h in raw_resolutions:
            if abs(w - res_w) < 50 and abs(h - res_h) < 50:
                return 0.6  # Likely raw gameplay resolution

        return 0.3

    def _classify_content_type(self, indicators: dict[str, float]) -> ContentType:
        """Classify content type based on indicators."""
        # Weighted scoring
        streamer_score = (
            indicators["webcam_score"] * 0.3
            + indicators["overlay_score"] * 0.25
            + indicators["chat_overlay_score"] * 0.2
            + (1.0 - indicators["hud_consistency"]) * 0.15
            + (1.0 - indicators["resolution_score"]) * 0.1
        )

        raw_score = (
            indicators["hud_consistency"] * 0.4
            + indicators["resolution_score"] * 0.3
            + (1.0 - indicators["webcam_score"]) * 0.2
            + (1.0 - indicators["overlay_score"]) * 0.1
        )

        logger.debug(
            f"Content classification - Streamer: {streamer_score:.3f}, Raw: {raw_score:.3f}"
        )

        if streamer_score > 0.6:
            return ContentType.STREAMER_CONTENT
        elif raw_score > 0.6:
            return ContentType.RAW_GAMEPLAY
        else:
            return ContentType.UNKNOWN


class TemplateEngine:
    """Template matching engine for down/distance detection."""

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.raw_templates = {}
        self.streamer_templates = {}
        self.load_templates()

    def load_templates(self):
        """Load template images for both raw and streamer content."""
        # Create template directories if they don't exist
        raw_dir = self.templates_dir / "raw_gameplay"
        streamer_dir = self.templates_dir / "streamer_content"

        raw_dir.mkdir(parents=True, exist_ok=True)
        streamer_dir.mkdir(parents=True, exist_ok=True)

        # Load raw gameplay templates
        self.raw_templates = self._load_template_set(raw_dir)

        # Load streamer content templates
        self.streamer_templates = self._load_template_set(streamer_dir)

        logger.info(
            f"Loaded {len(self.raw_templates)} raw templates, {len(self.streamer_templates)} streamer templates"
        )

    def _load_template_set(self, template_dir: Path) -> dict[str, dict]:
        """Load a set of templates from directory."""
        templates = {}

        if not template_dir.exists():
            return templates

        for template_file in template_dir.glob("*.png"):
            try:
                template_img = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template_img is not None:
                    # Parse filename for down/distance info
                    name = template_file.stem
                    down, distance = self._parse_template_name(name)

                    if down is not None and distance is not None:
                        templates[name] = {
                            "image": template_img,
                            "down": down,
                            "distance": distance,
                            "scales": [0.8, 0.9, 1.0, 1.1, 1.2],
                        }
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")

        return templates

    def _parse_template_name(self, name: str) -> tuple[Optional[int], Optional[int]]:
        """Parse template filename to extract down and distance."""
        try:
            # Handle formats like "1st_10", "3rd_7", "4th_goal"
            parts = name.lower().split("_")
            if len(parts) >= 2:
                # Parse down
                down_str = parts[0]
                if down_str.endswith("st"):
                    down = 1
                elif down_str.endswith("nd"):
                    down = 2
                elif down_str.endswith("rd"):
                    down = 3
                elif down_str.endswith("th"):
                    down = 4
                else:
                    down = int(down_str)

                # Parse distance
                distance_str = parts[1]
                if distance_str == "goal":
                    distance = 0  # Goal line
                else:
                    distance = int(distance_str)

                return down, distance
        except (ValueError, IndexError):
            pass

        return None, None

    def match_templates(self, region: np.ndarray, content_type: ContentType) -> list[TemplateMatch]:
        """Match templates against region."""
        matches = []

        # Choose appropriate template set
        if content_type == ContentType.RAW_GAMEPLAY:
            templates = self.raw_templates
        elif content_type == ContentType.STREAMER_CONTENT:
            templates = self.streamer_templates
        else:
            # Try both sets for unknown content
            templates = {**self.raw_templates, **self.streamer_templates}

        for template_name, template_data in templates.items():
            template_img = template_data["image"]
            scales = template_data["scales"]

            best_match = None
            best_confidence = 0.0

            # Multi-scale template matching
            for scale in scales:
                scaled_template = self._scale_template(template_img, scale)
                match = self._match_single_template(region, scaled_template)

                if match > best_confidence:
                    best_confidence = match
                    best_match = (template_data["down"], template_data["distance"])

            if best_confidence > 0.6:  # Confidence threshold
                matches.append(
                    TemplateMatch(
                        down=best_match[0],
                        distance=best_match[1],
                        confidence=best_confidence,
                        location=(0, 0),
                        template_name=template_name,
                    )
                )

        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _scale_template(self, template: np.ndarray, scale: float) -> np.ndarray:
        """Scale template image."""
        if scale == 1.0:
            return template

        h, w = template.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _match_single_template(self, region: np.ndarray, template: np.ndarray) -> float:
        """Match single template against region."""
        if region.shape[:2] < template.shape[:2]:
            return 0.0

        # Convert to grayscale for matching
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        template_gray = (
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        )

        # Template matching
        result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max_val


class ContentAwareDownDetector:
    """Main content-aware down/distance detection system."""

    def __init__(self, templates_dir: str = "templates"):
        self.content_detector = ContentTypeDetector()
        self.template_engine = TemplateEngine(templates_dir)

        # Initialize OCR fallback
        try:
            from src.spygate.ml.simple_paddle_ocr import SimplePaddleOCRWrapper

            self.ocr_engine = SimplePaddleOCRWrapper()
            self.ocr_available = True
        except ImportError:
            logger.warning("OCR engine not available, template matching only")
            self.ocr_engine = None
            self.ocr_available = False

    def detect_down_distance(self, frame: np.ndarray, hud_region: np.ndarray) -> DetectionResult:
        """
        Detect down and distance using content-aware approach.
        """
        # Step 1: Detect content type
        content_type = self.content_detector.detect_content_type(frame, hud_region)

        # Step 2: Choose primary detection method based on content type
        if content_type == ContentType.RAW_GAMEPLAY:
            return self._detect_raw_gameplay(hud_region, content_type)
        elif content_type == ContentType.STREAMER_CONTENT:
            return self._detect_streamer_content(hud_region, content_type)
        else:
            return self._detect_unknown_content(hud_region, content_type)

    def _detect_raw_gameplay(
        self, region: np.ndarray, content_type: ContentType
    ) -> DetectionResult:
        """Detect down/distance for raw gameplay (template matching primary)."""
        # Primary: Template matching
        template_matches = self.template_engine.match_templates(region, content_type)

        if template_matches and template_matches[0].confidence > 0.8:
            best_match = template_matches[0]
            return DetectionResult(
                down=best_match.down,
                distance=best_match.distance,
                confidence=best_match.confidence,
                method_used="template_matching",
                raw_text=f"{best_match.down} & {best_match.distance}",
                content_type=content_type,
            )

        # Fallback: OCR
        if self.ocr_available:
            return self._detect_with_ocr(region, content_type, "template_fallback")

        return DetectionResult(content_type=content_type, method_used="template_only_failed")

    def _detect_streamer_content(
        self, region: np.ndarray, content_type: ContentType
    ) -> DetectionResult:
        """Detect down/distance for streamer content (OCR primary)."""
        # Primary: OCR
        if self.ocr_available:
            ocr_result = self._detect_with_ocr(region, content_type, "ocr_primary")

            if ocr_result.confidence > 0.3:
                return ocr_result

        # Fallback: Template matching with streamer templates
        template_matches = self.template_engine.match_templates(region, content_type)

        if template_matches and template_matches[0].confidence > 0.6:
            best_match = template_matches[0]
            return DetectionResult(
                down=best_match.down,
                distance=best_match.distance,
                confidence=best_match.confidence * 0.8,
                method_used="template_fallback",
                raw_text=f"{best_match.down} & {best_match.distance}",
                content_type=content_type,
            )

        return DetectionResult(content_type=content_type, method_used="all_methods_failed")

    def _detect_unknown_content(
        self, region: np.ndarray, content_type: ContentType
    ) -> DetectionResult:
        """Detect down/distance for unknown content type."""
        # Try template matching first
        template_matches = self.template_engine.match_templates(region, content_type)

        if template_matches and template_matches[0].confidence > 0.8:
            best_match = template_matches[0]
            return DetectionResult(
                down=best_match.down,
                distance=best_match.distance,
                confidence=best_match.confidence,
                method_used="template_matching",
                raw_text=f"{best_match.down} & {best_match.distance}",
                content_type=content_type,
            )

        # Try OCR
        if self.ocr_available:
            ocr_result = self._detect_with_ocr(region, content_type, "ocr_fallback")

            if ocr_result.confidence > 0.3:
                return ocr_result

        # Use best template match even if low confidence
        if template_matches:
            best_match = template_matches[0]
            return DetectionResult(
                down=best_match.down,
                distance=best_match.distance,
                confidence=best_match.confidence * 0.7,
                method_used="template_low_confidence",
                raw_text=f"{best_match.down} & {best_match.distance}",
                content_type=content_type,
            )

        return DetectionResult(content_type=content_type, method_used="all_methods_failed")

    def _detect_with_ocr(
        self, region: np.ndarray, content_type: ContentType, method: str
    ) -> DetectionResult:
        """Detect using OCR engine."""
        try:
            # Extract text using OCR
            text = self.ocr_engine.extract_text(region)

            # Parse down and distance from text
            down, distance = self._parse_down_distance_text(text)

            if down is not None and distance is not None:
                confidence = self._calculate_ocr_confidence(text)
                return DetectionResult(
                    down=down,
                    distance=distance,
                    confidence=confidence,
                    method_used=method,
                    raw_text=text,
                    content_type=content_type,
                )
        except Exception as e:
            logger.warning(f"OCR detection failed: {e}")

        return DetectionResult(content_type=content_type, method_used=f"{method}_failed")

    def _parse_down_distance_text(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse down and distance from OCR text."""
        import re

        # Clean text
        text = text.upper().strip()

        # Common patterns
        patterns = [
            r"(\d+)(?:ST|ND|RD|TH)\s*&\s*(\d+)",  # "1ST & 10"
            r"(\d+)\s*&\s*(\d+)",  # "1 & 10"
            r"(\d+)(?:ST|ND|RD|TH)\s*(\d+)",  # "1ST 10"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    down = int(match.group(1))
                    distance = int(match.group(2))

                    # Validate ranges
                    if 1 <= down <= 4 and 0 <= distance <= 99:
                        return down, distance
                except ValueError:
                    continue

        # Check for goal line
        if "GOAL" in text:
            down_match = re.search(r"(\d+)(?:ST|ND|RD|TH)", text)
            if down_match:
                try:
                    down = int(down_match.group(1))
                    if 1 <= down <= 4:
                        return down, 0  # Goal line
                except ValueError:
                    pass

        return None, None

    def _calculate_ocr_confidence(self, text: str) -> float:
        """Calculate confidence score for OCR result."""
        if not text:
            return 0.0

        confidence = 0.3  # Base confidence

        # Length-based confidence
        if 5 <= len(text) <= 10:
            confidence += 0.2

        # Pattern-based confidence
        if "&" in text:
            confidence += 0.3

        if any(ord in text for ord in ["1ST", "2ND", "3RD", "4TH"]):
            confidence += 0.2

        return min(confidence, 1.0)


def create_sample_templates():
    """Create sample templates for testing."""
    templates_dir = Path("templates")
    raw_dir = templates_dir / "raw_gameplay"
    streamer_dir = templates_dir / "streamer_content"

    raw_dir.mkdir(parents=True, exist_ok=True)
    streamer_dir.mkdir(parents=True, exist_ok=True)

    # Create sample template images
    common_downs = [
        ("1st_10", 1, 10),
        ("2nd_10", 2, 10),
        ("3rd_10", 3, 10),
        ("4th_10", 4, 10),
        ("1st_5", 1, 5),
        ("2nd_7", 2, 7),
        ("3rd_3", 3, 3),
        ("4th_1", 4, 1),
        ("3rd_goal", 3, 0),
        ("4th_goal", 4, 0),
    ]

    for name, down, distance in common_downs:
        # Create placeholder template
        template = np.ones((30, 80, 3), dtype=np.uint8) * 50

        # Add text
        cv2.putText(
            template,
            f"{down} & {distance}" if distance > 0 else f"{down} & GOAL",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Save raw template
        cv2.imwrite(str(raw_dir / f"{name}.png"), template)

        # Create scaled version for streamer content
        streamer_template = cv2.resize(template, (60, 22))
        cv2.imwrite(str(streamer_dir / f"{name}.png"), streamer_template)

    print(f"Created sample templates in {templates_dir}")


if __name__ == "__main__":
    # Create sample templates for testing
    create_sample_templates()

    # Test the system
    detector = ContentAwareDownDetector()

    # Create test image
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 100
    cv2.putText(test_image, "1ST & 10", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Test detection
    result = detector.detect_down_distance(test_image, test_image)
    print(f"Detection result: {result}")
