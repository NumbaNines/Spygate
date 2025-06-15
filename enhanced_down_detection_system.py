#!/usr/bin/env python3
"""
Enhanced Down Detection System - Multi-Method Approach
Combines OCR + Game State Context + Possession/Territory + Template Matching
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


@dataclass
class GameState:
    """Track game state for context-based validation"""

    current_down: Optional[str] = None
    current_distance: Optional[int] = None
    previous_down: Optional[str] = None
    previous_distance: Optional[int] = None
    possession_changed: bool = False
    territory_changed: bool = False
    yards_gained: Optional[int] = None
    confidence: float = 0.0


@dataclass
class DetectionResult:
    """Result from down detection with confidence scoring"""

    down: Optional[str] = None
    distance: Optional[int] = None
    method: str = "unknown"
    confidence: float = 0.0
    raw_ocr: str = ""


class EnhancedDownDetector:
    """Multi-method down detection system"""

    def __init__(self):
        self.game_state = GameState()
        self.detection_history = []
        self.number_templates = self._load_number_templates()

        # OCR patterns for down detection
        self.down_patterns = [
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)",  # "1st & 10", "3rd & 8"
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*Goal",  # "1st & Goal"
            r"(\d+)\s*&\s*(\d+)",  # Simple "3 & 8"
            r"(\d+)(?:nd|rd|th|st)\s*&\s*(\d+)",  # OCR variations
        ]

        # Set Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def _load_number_templates(self) -> dict[str, np.ndarray]:
        """Load number templates for template matching"""
        templates = {}
        template_dir = Path("templates/numbers")

        if template_dir.exists():
            for i in range(1, 5):  # 1st, 2nd, 3rd, 4th
                template_path = template_dir / f"{i}.png"
                if template_path.exists():
                    templates[str(i)] = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)

        return templates

    def detect_down_adaptive(
        self,
        frame: np.ndarray,
        hud_bbox: tuple,
        possession_triangle_bbox: Optional[tuple] = None,
        territory_triangle_bbox: Optional[tuple] = None,
    ) -> DetectionResult:
        """
        Adaptive down detection using multiple methods
        """
        # 1. Calculate adaptive coordinates based on detected HUD elements
        down_region = self._calculate_adaptive_region(frame, hud_bbox, possession_triangle_bbox)

        if down_region is None:
            return DetectionResult(method="failed", confidence=0.0)

        # 2. Multi-method detection
        ocr_result = self._detect_with_enhanced_ocr(down_region)
        template_result = self._detect_with_template_matching(down_region)
        context_result = self._validate_with_game_context(
            ocr_result, possession_triangle_bbox, territory_triangle_bbox
        )

        # 3. Combine results with confidence weighting
        final_result = self._combine_detection_results(ocr_result, template_result, context_result)

        # 4. Update game state
        self._update_game_state(final_result)

        return final_result

    def _calculate_adaptive_region(
        self, frame: np.ndarray, hud_bbox: tuple, possession_triangle_bbox: Optional[tuple] = None
    ) -> Optional[np.ndarray]:
        """
        Calculate down/distance region relative to detected HUD elements
        """
        try:
            x1, y1, x2, y2 = hud_bbox
            hud_region = frame[int(y1) : int(y2), int(x1) : int(x2)]
            h, w = hud_region.shape[:2]

            if possession_triangle_bbox:
                # Use possession triangle as anchor point
                px1, py1, px2, py2 = possession_triangle_bbox

                # Down/distance is typically to the RIGHT of possession triangle
                # Calculate relative position
                relative_x_start = max(0.6, (px2 - x1) / (x2 - x1))  # Start after possession area
                relative_x_end = min(1.0, relative_x_start + 0.3)  # 30% width for down/distance

            else:
                # Fallback: assume down/distance is in right portion of HUD
                relative_x_start = 0.6  # 60% across
                relative_x_end = 0.95  # 95% across

            # Vertical positioning (typically center of HUD)
            relative_y_start = 0.2  # 20% down
            relative_y_end = 0.8  # 80% down

            # Convert to pixel coordinates
            x_start = int(w * relative_x_start)
            x_end = int(w * relative_x_end)
            y_start = int(h * relative_y_start)
            y_end = int(h * relative_y_end)

            # Extract the region
            down_region = hud_region[y_start:y_end, x_start:x_end]

            return down_region

        except Exception as e:
            print(f"❌ Error calculating adaptive region: {e}")
            return None

    def _detect_with_enhanced_ocr(self, region: np.ndarray) -> DetectionResult:
        """
        Enhanced OCR with preprocessing and pattern validation
        """
        try:
            # Preprocess region for better OCR
            processed_region = self._preprocess_for_ocr(region)

            # Try multiple OCR configurations
            configs = [
                "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789&stndGoalAMP",
                "--oem 3 --psm 6",
            ]

            best_result = DetectionResult(method="ocr", confidence=0.0)

            for config in configs:
                try:
                    ocr_text = pytesseract.image_to_string(processed_region, config=config).strip()

                    if ocr_text:
                        parsed = self._parse_down_distance(ocr_text)
                        if parsed.confidence > best_result.confidence:
                            best_result = parsed
                            best_result.raw_ocr = ocr_text
                            best_result.method = "ocr"

                except Exception:
                    continue

            return best_result

        except Exception as e:
            print(f"❌ OCR detection error: {e}")
            return DetectionResult(method="ocr_failed", confidence=0.0)

    def _preprocess_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess region for optimal OCR performance
        """
        # Scale up for better OCR
        scale_factor = 5
        scaled = cv2.resize(
            region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        if len(scaled.shape) == 3:
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = scaled

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _parse_down_distance(self, ocr_text: str) -> DetectionResult:
        """
        Parse OCR text with smart handling of 2nd/3rd confusion
        """
        result = DetectionResult()

        for pattern in self.down_patterns:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                try:
                    down_num = int(match.group(1))

                    # Validate down number
                    if 1 <= down_num <= 4:
                        result.down = f"{down_num}"

                        # Handle distance
                        if len(match.groups()) > 1 and match.group(2).lower() != "goal":
                            try:
                                result.distance = int(match.group(2))
                            except ValueError:
                                result.distance = None
                        else:
                            result.distance = 0  # Goal line

                        # Smart handling of 2nd/3rd confusion
                        if "2" in ocr_text and (
                            "nd" in ocr_text.lower() or "rd" in ocr_text.lower()
                        ):
                            # Check context - if previous was 1st, likely 2nd
                            if self.game_state.previous_down == "1":
                                result.down = "2"
                                result.confidence = 0.8
                            else:
                                result.down = "2"  # Default to 2nd when ambiguous
                                result.confidence = 0.6
                        elif "3" in ocr_text and (
                            "nd" in ocr_text.lower() or "rd" in ocr_text.lower()
                        ):
                            if self.game_state.previous_down == "2":
                                result.down = "3"
                                result.confidence = 0.8
                            else:
                                result.down = "3"  # Default to 3rd when ambiguous
                                result.confidence = 0.6
                        else:
                            result.confidence = 0.9  # High confidence for clear matches

                        break

                except ValueError:
                    continue

        return result

    def _detect_with_template_matching(self, region: np.ndarray) -> DetectionResult:
        """
        Template matching for number recognition
        """
        if not self.number_templates:
            return DetectionResult(method="template_no_data", confidence=0.0)

        try:
            # Convert region to grayscale
            if len(region.shape) == 3:
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = region

            best_match = DetectionResult(method="template", confidence=0.0)

            for down_num, template in self.number_templates.items():
                # Multi-scale template matching
                for scale in [0.8, 1.0, 1.2, 1.5]:
                    scaled_template = cv2.resize(template, None, fx=scale, fy=scale)

                    if (
                        scaled_template.shape[0] <= gray_region.shape[0]
                        and scaled_template.shape[1] <= gray_region.shape[1]
                    ):
                        result = cv2.matchTemplate(
                            gray_region, scaled_template, cv2.TM_CCOEFF_NORMED
                        )
                        _, max_val, _, _ = cv2.minMaxLoc(result)

                        if max_val > best_match.confidence:
                            best_match.down = down_num
                            best_match.confidence = max_val
                            best_match.method = "template"

            return best_match

        except Exception as e:
            print(f"❌ Template matching error: {e}")
            return DetectionResult(method="template_failed", confidence=0.0)

    def _validate_with_game_context(
        self,
        ocr_result: DetectionResult,
        possession_bbox: Optional[tuple],
        territory_bbox: Optional[tuple],
    ) -> DetectionResult:
        """
        Validate detection using game state context
        """
        context_result = DetectionResult(method="context", confidence=0.0)

        # Check for possession change (indicates new 1st down)
        if self._detect_possession_change(possession_bbox):
            context_result.down = "1"
            context_result.confidence = 0.9
            context_result.method = "context_possession_change"
            return context_result

        # Validate against previous game state
        if self.game_state.previous_down and ocr_result.down:
            prev_down = int(self.game_state.previous_down)
            curr_down = int(ocr_result.down)

            # Logical progression validation
            if curr_down == prev_down + 1:  # Normal progression
                context_result.confidence = 0.8
            elif curr_down == 1 and prev_down >= 1:  # New set of downs
                context_result.confidence = 0.7
            elif curr_down == prev_down:  # Same down (possible)
                context_result.confidence = 0.5
            else:  # Unusual progression
                context_result.confidence = 0.2

            context_result.down = ocr_result.down
            context_result.distance = ocr_result.distance
            context_result.method = "context_validation"

        return context_result

    def _detect_possession_change(self, possession_bbox: Optional[tuple]) -> bool:
        """
        Detect if possession has changed (indicates new 1st down)
        """
        # This would integrate with our proven triangle detection system
        # For now, return False - implement with actual triangle detection
        return False

    def _combine_detection_results(
        self,
        ocr_result: DetectionResult,
        template_result: DetectionResult,
        context_result: DetectionResult,
    ) -> DetectionResult:
        """
        Combine multiple detection methods with weighted confidence
        """
        # Weight the methods
        ocr_weight = 0.5
        template_weight = 0.3
        context_weight = 0.2

        # Calculate weighted confidence
        total_confidence = (
            ocr_result.confidence * ocr_weight
            + template_result.confidence * template_weight
            + context_result.confidence * context_weight
        )

        # Choose the result with highest individual confidence
        candidates = [ocr_result, template_result, context_result]
        best_result = max(candidates, key=lambda x: x.confidence)

        # If results agree, boost confidence
        if (
            ocr_result.down == template_result.down == context_result.down
            and ocr_result.down is not None
        ):
            best_result.confidence = min(0.95, total_confidence + 0.2)
            best_result.method = "consensus"
        else:
            best_result.confidence = total_confidence

        return best_result

    def _update_game_state(self, result: DetectionResult):
        """
        Update internal game state for context tracking
        """
        if result.down:
            self.game_state.previous_down = self.game_state.current_down
            self.game_state.current_down = result.down

        if result.distance is not None:
            self.game_state.previous_distance = self.game_state.current_distance
            self.game_state.current_distance = result.distance

        # Add to detection history
        self.detection_history.append(result)

        # Keep only recent history (last 10 detections)
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)


def test_enhanced_detection():
    """Test the enhanced detection system"""
    detector = EnhancedDownDetector()

    # Load test frame
    frame_path = "found_and_frame_3000.png"
    if not Path(frame_path).exists():
        print(f"❌ Test frame not found: {frame_path}")
        return

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"❌ Could not load frame: {frame_path}")
        return

    # Mock HUD bbox (would come from YOLO detection)
    height, width = frame.shape[:2]
    hud_bbox = (0, int(height * 0.85), width, height)  # Bottom 15% of frame

    print("🎯 Testing Enhanced Down Detection System")
    print("=" * 50)

    # Test detection
    result = detector.detect_down_adaptive(frame, hud_bbox)

    print(f"📊 Detection Result:")
    print(f"   Down: {result.down}")
    print(f"   Distance: {result.distance}")
    print(f"   Method: {result.method}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Raw OCR: '{result.raw_ocr}'")


if __name__ == "__main__":
    test_enhanced_detection()
