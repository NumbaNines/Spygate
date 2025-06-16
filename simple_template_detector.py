#!/usr/bin/env python3
"""
Simple Template-Based Down/Distance Detector
Uses 4 down templates (1ST, 2ND, 3RD, 4TH) + OCR for distance.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types based on resolution."""

    RAW_GAMEPLAY = "raw_gameplay"  # 1080p, 720p, etc. - full resolution
    STREAMER_CONTENT = "streamer_content"  # Scaled down for streaming


@dataclass
class DetectionResult:
    """Result from down/distance detection."""

    down: Optional[int] = None
    distance: Optional[int] = None
    confidence: float = 0.0
    method_used: str = "template_ocr_hybrid"
    raw_text: str = ""
    content_type: ContentType = ContentType.RAW_GAMEPLAY
    template_match: str = ""
    ocr_text: str = ""


@dataclass
class DownTemplateMatch:
    """Down template matching result."""

    down: int
    confidence: float
    location: tuple[int, int, int, int]  # x, y, w, h
    template_name: str


class ResolutionDetector:
    """Simple resolution-based content type detection."""

    def detect_content_type(self, hud_region: np.ndarray) -> ContentType:
        """
        Detect content type based on HUD region size.

        Raw gameplay: Larger HUD regions (1080p, 720p)
        Streamer content: Smaller HUD regions (scaled for streaming)
        """
        h, w = hud_region.shape[:2]
        region_area = h * w

        # Thresholds based on typical HUD region sizes
        # Raw gameplay: ~80x30 pixels = 2400 area
        # Streamer content: ~60x22 pixels = 1320 area

        if region_area > 2000:
            return ContentType.RAW_GAMEPLAY
        else:
            return ContentType.STREAMER_CONTENT


class DownTemplateEngine:
    """Template engine for down detection (1ST, 2ND, 3RD, 4TH)."""

    def __init__(self, templates_dir: str = "down templates"):
        self.templates_dir = Path(templates_dir)
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        """Load real down templates from Madden 25 screenshots."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        # Map template files to down numbers
        template_mapping = {
            "1.png": ("1ST", 1),
            "2.png": ("2ND", 2),
            "3rd.png": ("3RD", 3),
            "4th.png": ("4TH", 4),
        }

        for filename, (down_name, down_num) in template_mapping.items():
            template_path = self.templates_dir / filename
            if template_path.exists():
                try:
                    template_img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                    if template_img is not None:
                        self.templates[down_name] = {
                            "image": template_img,
                            "down": down_num,
                            "filename": filename,
                        }
                        logger.info(f"Loaded real template: {down_name} from {filename}")
                    else:
                        logger.warning(f"Failed to load image: {template_path}")
                except Exception as e:
                    logger.warning(f"Error loading template {template_path}: {e}")
            else:
                logger.warning(f"Template file not found: {template_path}")

        logger.info(f"Loaded {len(self.templates)} real down templates from Madden 25")

    def _load_template_set(self, template_dir: Path) -> dict[str, dict]:
        """Legacy method - not used with real templates."""
        return {}

    def _parse_down_template_name(self, name: str) -> Optional[int]:
        """Parse down template filename: '1ST', '2ND', '3RD', '4TH'."""
        down_map = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}
        return down_map.get(name.upper())

    def match_down_templates(
        self, region: np.ndarray, content_type: ContentType
    ) -> list[DownTemplateMatch]:
        """Match real down templates against region."""
        matches = []

        # Use real templates (ignore content_type for now since we have actual screenshots)
        for template_name, template_data in self.templates.items():
            template_img = template_data["image"]
            confidence, location = self._match_single_down_template(region, template_img)

            if confidence > 0.3:  # Lower threshold for real templates
                matches.append(
                    DownTemplateMatch(
                        down=template_data["down"],
                        confidence=confidence,
                        location=location,
                        template_name=template_name,
                    )
                )

        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _match_single_down_template(
        self, region: np.ndarray, template: np.ndarray
    ) -> tuple[float, tuple[int, int, int, int]]:
        """Match single down template against region."""
        if region.shape[:2] < template.shape[:2]:
            return 0.0, (0, 0, 0, 0)

        # Convert to grayscale
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        template_gray = (
            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        )

        # Template matching
        result = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Get template dimensions
        h, w = template_gray.shape
        location = (max_loc[0], max_loc[1], w, h)

        return max_val, location


class DistanceOCREngine:
    """OCR engine for distance extraction."""

    def __init__(self):
        self.ocr_available = False
        try:
            from src.spygate.ml.simple_paddle_ocr import SimplePaddleOCRWrapper

            self.ocr_engine = SimplePaddleOCRWrapper()
            self.ocr_available = True
            logger.info("OCR engine available for distance extraction")
        except ImportError:
            logger.warning("OCR engine not available - distance extraction will fail")

    def extract_distance_from_region(
        self, region: np.ndarray, down_location: tuple[int, int, int, int]
    ) -> tuple[Optional[int], str, float]:
        """
        Extract distance from region after the down template match.

        Args:
            region: Full HUD region
            down_location: (x, y, w, h) of matched down template

        Returns:
            (distance, raw_ocr_text, confidence)
        """
        if not self.ocr_available:
            return None, "", 0.0

        try:
            # Calculate distance region (area after the down template)
            x, y, w, h = down_location

            # Distance region starts after the down template
            distance_x = x + w
            distance_region = region[y : y + h, distance_x:]

            if distance_region.size == 0:
                return None, "", 0.0

            # Extract text using OCR
            raw_text = self.ocr_engine.extract_text(distance_region)

            # Parse distance from OCR text
            distance, confidence = self._parse_distance_text(raw_text)

            return distance, raw_text, confidence

        except Exception as e:
            logger.warning(f"Distance OCR extraction failed: {e}")
            return None, "", 0.0

    def _parse_distance_text(self, text: str) -> tuple[Optional[int], float]:
        """Parse distance from OCR text."""
        if not text:
            return None, 0.0

        text = text.upper().strip()
        confidence = 0.3

        # Look for "GOAL" first
        if "GOAL" in text:
            confidence += 0.4
            return 0, min(confidence, 1.0)

        # Look for "& NUMBER" pattern
        patterns = [
            r"&\s*(\d+)",  # "& 10"
            r"[&\s]+(\d+)",  # "& 10" or " 10"
            r"(\d+)",  # Just a number
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    distance = int(match.group(1))
                    if 0 <= distance <= 99:
                        confidence += 0.3
                        if "&" in text:
                            confidence += 0.2
                        return distance, min(confidence, 1.0)
                except ValueError:
                    continue

        return None, 0.0


class SimpleTemplateDetector:
    """Main template-based down detection + OCR distance detection system."""

    def __init__(self, templates_dir: str = "down_templates"):
        self.resolution_detector = ResolutionDetector()
        self.down_engine = DownTemplateEngine(templates_dir)
        self.distance_engine = DistanceOCREngine()

    def detect_down_distance(self, hud_region: np.ndarray) -> DetectionResult:
        """
        Detect down and distance using template matching + OCR.

        Args:
            hud_region: HUD region containing down/distance text

        Returns:
            DetectionResult with down, distance, and confidence
        """
        # Step 1: Detect content type based on region size
        content_type = self.resolution_detector.detect_content_type(hud_region)

        # Step 2: Template matching for down
        down_matches = self.down_engine.match_down_templates(hud_region, content_type)

        if not down_matches:
            return DetectionResult(content_type=content_type, method_used="no_down_template_match")

        # Step 3: Use best down match
        best_down_match = down_matches[0]

        # Step 4: OCR for distance
        distance, ocr_text, distance_confidence = self.distance_engine.extract_distance_from_region(
            hud_region, best_down_match.location
        )

        # Step 5: Calculate combined confidence
        combined_confidence = (best_down_match.confidence * 0.6) + (distance_confidence * 0.4)

        # Step 6: Build result
        if distance is not None:
            raw_text = (
                f"{best_down_match.down} & {distance}"
                if distance > 0
                else f"{best_down_match.down} & GOAL"
            )
            method = "template_ocr_hybrid"
        else:
            # Fallback: use down only
            raw_text = f"{best_down_match.down} & ?"
            method = "template_only"
            combined_confidence *= 0.5  # Lower confidence without distance

        return DetectionResult(
            down=best_down_match.down,
            distance=distance,
            confidence=combined_confidence,
            method_used=method,
            raw_text=raw_text,
            content_type=content_type,
            template_match=best_down_match.template_name,
            ocr_text=ocr_text,
        )


def create_down_templates():
    """Create down templates for different resolutions."""
    templates_dir = Path("down_templates")
    raw_dir = templates_dir / "raw_gameplay"
    streamer_dir = templates_dir / "streamer_content"

    raw_dir.mkdir(parents=True, exist_ok=True)
    streamer_dir.mkdir(parents=True, exist_ok=True)

    # Down strings
    down_strings = ["1ST", "2ND", "3RD", "4TH"]

    for down_str in down_strings:
        # Raw gameplay template (larger, 1080p/720p size)
        raw_template = np.ones((30, 40, 3), dtype=np.uint8) * 40  # Dark background
        cv2.putText(
            raw_template, down_str, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        cv2.imwrite(str(raw_dir / f"{down_str}.png"), raw_template)

        # Streamer content template (smaller, scaled for streaming)
        streamer_template = np.ones((22, 30, 3), dtype=np.uint8) * 40
        cv2.putText(
            streamer_template, down_str, (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        cv2.imwrite(str(streamer_dir / f"{down_str}.png"), streamer_template)

    print(f"✅ Created down templates:")
    print(f"   📁 Raw gameplay: {len(down_strings)} templates (30x40px)")
    print(f"   📁 Streamer content: {len(down_strings)} templates (22x30px)")
    print(f"   📂 Location: {templates_dir}")


def test_template_detector():
    """Test the template detection system."""
    print("\n🧪 Testing 4-Template + OCR Detection System")
    print("=" * 50)

    # Create templates
    create_down_templates()

    # Initialize detector
    detector = SimpleTemplateDetector()

    # Test 1: Raw gameplay simulation
    print("\n1️⃣ Testing Raw Gameplay Detection")
    raw_test = np.ones((30, 80, 3), dtype=np.uint8) * 40
    cv2.putText(raw_test, "1ST & 10", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    result1 = detector.detect_down_distance(raw_test)
    print(f"   Result: {result1.down} & {result1.distance}")
    print(f"   Confidence: {result1.confidence:.3f}")
    print(f"   Content Type: {result1.content_type.value}")
    print(f"   Method: {result1.method_used}")
    print(f"   Template Match: {result1.template_match}")
    print(f"   OCR Text: '{result1.ocr_text}'")

    # Test 2: Streamer content simulation
    print("\n2️⃣ Testing Streamer Content Detection")
    streamer_test = np.ones((22, 60, 3), dtype=np.uint8) * 40
    cv2.putText(
        streamer_test, "3RD & 7", (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
    )

    result2 = detector.detect_down_distance(streamer_test)
    print(f"   Result: {result2.down} & {result2.distance}")
    print(f"   Confidence: {result2.confidence:.3f}")
    print(f"   Content Type: {result2.content_type.value}")
    print(f"   Method: {result2.method_used}")
    print(f"   Template Match: {result2.template_match}")
    print(f"   OCR Text: '{result2.ocr_text}'")

    # Test 3: Goal line situation
    print("\n3️⃣ Testing Goal Line Detection")
    goal_test = np.ones((30, 80, 3), dtype=np.uint8) * 40
    cv2.putText(goal_test, "4TH & GOAL", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    result3 = detector.detect_down_distance(goal_test)
    print(f"   Result: {result3.down} & {result3.distance}")
    print(f"   Confidence: {result3.confidence:.3f}")
    print(f"   Content Type: {result3.content_type.value}")
    print(f"   Method: {result3.method_used}")
    print(f"   Template Match: {result3.template_match}")
    print(f"   OCR Text: '{result3.ocr_text}'")

    print("\n✅ 4-Template + OCR Detection System Test Complete!")
    print("\n📋 Summary:")
    print("   🎯 Template matching: 1ST, 2ND, 3RD, 4TH (4 templates)")
    print("   🔍 OCR extraction: Distance numbers and GOAL")
    print("   ⚡ Hybrid approach: Best of both worlds")


if __name__ == "__main__":
    test_template_detector()
