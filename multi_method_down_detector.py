#!/usr/bin/env python3
"""
Multi-Method Down Detection System
Combines: OCR + Game State Context + Possession/Territory Analysis
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


class MultiMethodDownDetector:
    """Professional down detection using multiple validation methods"""

    def __init__(self):
        self.game_state = GameState()
        self.detection_history = []

        # OCR patterns for down detection
        self.down_patterns = [
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+)",  # "1st & 10", "3rd & 8"
            r"(\d+)(?:st|nd|rd|th)?\s*&\s*Goal",  # "1st & Goal"
            r"(\d+)\s*&\s*(\d+)",  # Simple "3 & 8"
            r"(\d+)(?:nd|rd|th|st)\s*&\s*(\d+)",  # OCR variations
        ]

        # Set Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def detect_down_professional(
        self,
        frame: np.ndarray,
        hud_bbox: tuple,
        possession_triangle_bbox: Optional[tuple] = None,
        territory_triangle_bbox: Optional[tuple] = None,
    ) -> DetectionResult:
        """
        Professional-grade down detection using multiple methods
        """
        print("🎯 Starting Multi-Method Down Detection")

        # Method 1: Enhanced OCR
        ocr_result = self._method_1_enhanced_ocr(frame, hud_bbox, possession_triangle_bbox)
        print(
            f"   Method 1 (OCR): {ocr_result.down} & {ocr_result.distance} (conf: {ocr_result.confidence:.3f})"
        )

        # Method 5: Game State Context Analysis
        context_result = self._method_5_game_state_context(ocr_result)
        print(
            f"   Method 5 (Context): {context_result.down} & {context_result.distance} (conf: {context_result.confidence:.3f})"
        )

        # Method 6: Possession/Territory State Validation
        possession_result = self._method_6_possession_territory_validation(
            ocr_result, possession_triangle_bbox, territory_triangle_bbox
        )
        print(
            f"   Method 6 (Possession): {possession_result.down} & {possession_result.distance} (conf: {possession_result.confidence:.3f})"
        )

        # Combine all methods
        final_result = self._combine_all_methods(ocr_result, context_result, possession_result)
        print(
            f"   🏆 Final Result: {final_result.down} & {final_result.distance} (conf: {final_result.confidence:.3f})"
        )

        # Update game state
        self._update_game_state(final_result)

        return final_result

    def _method_1_enhanced_ocr(
        self, frame: np.ndarray, hud_bbox: tuple, possession_triangle_bbox: Optional[tuple] = None
    ) -> DetectionResult:
        """
        Method 1: Enhanced OCR with adaptive region calculation
        """
        try:
            # Calculate adaptive region based on HUD elements
            down_region = self._calculate_adaptive_region(frame, hud_bbox, possession_triangle_bbox)

            if down_region is None:
                return DetectionResult(method="ocr_failed", confidence=0.0)

            # Preprocess for better OCR
            processed_region = self._preprocess_for_ocr(down_region)

            # Try multiple OCR configurations
            configs = [
                "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789&stndGoalAMP",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789&stndGoalAMP",
                "--oem 3 --psm 6",
            ]

            best_result = DetectionResult(method="enhanced_ocr", confidence=0.0)

            for config in configs:
                try:
                    ocr_text = pytesseract.image_to_string(processed_region, config=config).strip()

                    if ocr_text:
                        parsed = self._parse_down_distance_smart(ocr_text)
                        if parsed.confidence > best_result.confidence:
                            best_result = parsed
                            best_result.raw_ocr = ocr_text
                            best_result.method = "enhanced_ocr"

                except Exception:
                    continue

            return best_result

        except Exception as e:
            print(f"❌ Enhanced OCR error: {e}")
            return DetectionResult(method="ocr_error", confidence=0.0)

    def _method_5_game_state_context(self, ocr_result: DetectionResult) -> DetectionResult:
        """
        Method 5: Game State Context Analysis
        Track game flow patterns to validate down progression
        """
        context_result = DetectionResult(method="game_context", confidence=0.0)

        # If we have previous game state, validate logical progression
        if self.game_state.previous_down and ocr_result.down:
            try:
                prev_down = int(self.game_state.previous_down)
                curr_down = int(ocr_result.down)

                # Logical progression patterns
                if curr_down == prev_down + 1:  # Normal progression
                    context_result.confidence = 0.9
                elif curr_down == 1 and prev_down >= 1:  # New set of downs
                    context_result.confidence = 0.8
                    context_result.down = "1"
                    context_result.distance = 10
                elif curr_down == prev_down:  # Same down
                    context_result.confidence = 0.6
                else:  # Unusual progression
                    context_result.confidence = 0.3

                if not context_result.down:
                    context_result.down = ocr_result.down
                    context_result.distance = ocr_result.distance

            except (ValueError, TypeError):
                context_result.confidence = 0.1
        else:
            # No previous state - accept OCR result with moderate confidence
            context_result.confidence = 0.5
            context_result.down = ocr_result.down
            context_result.distance = ocr_result.distance

        return context_result

    def _method_6_possession_territory_validation(
        self,
        ocr_result: DetectionResult,
        possession_bbox: Optional[tuple],
        territory_bbox: Optional[tuple],
    ) -> DetectionResult:
        """
        Method 6: Possession/Territory State Validation
        Use possession changes to detect new first downs
        """
        possession_result = DetectionResult(method="possession_validation", confidence=0.0)

        # Check for possession change (indicates new 1st down)
        if self._detect_possession_change(possession_bbox):
            possession_result.down = "1"
            possession_result.distance = 10
            possession_result.confidence = 0.95
            possession_result.method = "possession_change_detected"
            return possession_result

        # Check for territory change (field position shift)
        if self._detect_territory_change(territory_bbox):
            if ocr_result.down == "1":
                possession_result.confidence = 0.8
            else:
                possession_result.confidence = 0.6
        else:
            possession_result.confidence = 0.7

        possession_result.down = ocr_result.down
        possession_result.distance = ocr_result.distance

        return possession_result

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
                relative_x_start = max(0.6, (px2 - x1) / (x2 - x1))
                relative_x_end = min(1.0, relative_x_start + 0.35)

            else:
                # Fallback: assume down/distance is in right portion of HUD
                relative_x_start = 0.6
                relative_x_end = 0.95

            # Vertical positioning (center of HUD)
            relative_y_start = 0.2
            relative_y_end = 0.8

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

    def _parse_down_distance_smart(self, ocr_text: str) -> DetectionResult:
        """
        Smart parsing with 2nd/3rd confusion handling
        """
        result = DetectionResult()

        for pattern in self.down_patterns:
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                try:
                    down_num = int(match.group(1))

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

                        # Smart 2nd/3rd confusion handling
                        if "2" in ocr_text and (
                            "nd" in ocr_text.lower() or "rd" in ocr_text.lower()
                        ):
                            if self.game_state.previous_down == "1":
                                result.down = "2"
                                result.confidence = 0.85
                            else:
                                result.down = "2"
                                result.confidence = 0.65
                        elif "3" in ocr_text and (
                            "nd" in ocr_text.lower() or "rd" in ocr_text.lower()
                        ):
                            if self.game_state.previous_down == "2":
                                result.down = "3"
                                result.confidence = 0.85
                            else:
                                result.down = "3"
                                result.confidence = 0.65
                        else:
                            result.confidence = 0.9

                        break

                except ValueError:
                    continue

        return result

    def _detect_possession_change(self, possession_bbox: Optional[tuple]) -> bool:
        """
        Detect possession change (would integrate with triangle detection)
        """
        # Placeholder - would integrate with proven triangle detection system
        return False

    def _detect_territory_change(self, territory_bbox: Optional[tuple]) -> bool:
        """
        Detect territory change (would integrate with triangle detection)
        """
        # Placeholder - would integrate with proven triangle detection system
        return False

    def _combine_all_methods(
        self,
        ocr_result: DetectionResult,
        context_result: DetectionResult,
        possession_result: DetectionResult,
    ) -> DetectionResult:
        """
        Combine all detection methods with weighted confidence
        """
        # Method weights
        ocr_weight = 0.5
        context_weight = 0.3
        possession_weight = 0.2

        # Calculate weighted confidence
        total_confidence = (
            ocr_result.confidence * ocr_weight
            + context_result.confidence * context_weight
            + possession_result.confidence * possession_weight
        )

        # Choose the result with highest individual confidence
        candidates = [ocr_result, context_result, possession_result]
        best_result = max(candidates, key=lambda x: x.confidence)

        # If multiple methods agree, boost confidence
        agreements = 0
        if ocr_result.down == context_result.down == possession_result.down:
            agreements = 3
        elif (
            ocr_result.down == context_result.down
            or ocr_result.down == possession_result.down
            or context_result.down == possession_result.down
        ):
            agreements = 2

        if agreements >= 2:
            best_result.confidence = min(0.95, total_confidence + 0.15)
            best_result.method = f"consensus_{agreements}_methods"
        else:
            best_result.confidence = total_confidence
            best_result.method = f"best_individual_{best_result.method}"

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


def test_multi_method_detection():
    """Test the multi-method detection system"""
    detector = MultiMethodDownDetector()

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
    hud_bbox = (0, int(height * 0.85), width, height)

    print("🎯 Testing Multi-Method Down Detection System")
    print("=" * 60)

    # Test multiple scenarios
    scenarios = [
        {"name": "3rd & 8 (Critical)", "expected": ("3", 8)},
        {"name": "1st & 10 (Fresh Set)", "expected": ("1", 10)},
        {"name": "4th & 2 (Decision Point)", "expected": ("4", 2)},
        {"name": "2nd & Goal (Red Zone)", "expected": ("2", 0)},
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * 40)

        result = detector.detect_down_professional(frame, hud_bbox)

        print(f"   Final Result: {result.down} & {result.distance}")
        print(f"   Method: {result.method}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Raw OCR: '{result.raw_ocr}'")

        # Simulate game state progression
        detector.game_state.previous_down = result.down
        detector.game_state.previous_distance = result.distance


if __name__ == "__main__":
    test_multi_method_detection()
