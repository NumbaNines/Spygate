import time
from collections import Counter, deque
import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
import copy

class OCREnsemble:
    """Multi-model ensemble for higher accuracy OCR."""

    def __init__(self, optimized_paddle_ocr):
        # Keep user's optimized PaddleOCR as primary (0.939 score)
        self.primary_ocr = optimized_paddle_ocr

        # Add secondary OCR engines for ensemble voting
        try:
            import pytesseract
            self.tesseract_available = True
            # Configure Tesseract for Madden HUD
            self.tesseract_config = '--psm 8 -c tessedit_char_whitelist=0123456789stndrdthGoal&: '
        except ImportError:
            self.tesseract_available = False

        try:
            import easyocr
            self.easyocr = easyocr.Reader(['en'], gpu=True)
            self.easyocr_available = True
        except ImportError:
            self.easyocr_available = False

        print(f"🤖 OCR Ensemble initialized:")
        print(f"   Primary: Optimized PaddleOCR (0.939 score)")
        print(f"   Tesseract: {'✅' if self.tesseract_available else '❌'}")
        print(f"   EasyOCR: {'✅' if self.easyocr_available else '❌'}")

    def ensemble_predict(self, processed_image):
        """Get ensemble prediction from multiple OCR engines."""
        results = []

        # Primary OCR (user's optimized PaddleOCR)
        try:
            paddle_result = self.primary_ocr.ocr(processed_image, cls=True)
            if paddle_result and paddle_result[0]:
                text = paddle_result[0][0][1][0]
                confidence = paddle_result[0][0][1][1]
                results.append({
                    'engine': 'paddle_optimized',
                    'text': text,
                    'confidence': confidence,
                    'weight': 0.6  # Highest weight for optimized model
                })
        except Exception as e:
            print(f"⚠️ Paddle OCR error: {e}")

        # Secondary: Tesseract
        if self.tesseract_available:
            try:
                import pytesseract
                tesseract_text = pytesseract.image_to_string(
                    processed_image, config=self.tesseract_config
                ).strip()
                if tesseract_text:
                    results.append({
                        'engine': 'tesseract',
                        'text': tesseract_text,
                        'confidence': 0.8,  # Estimated confidence
                        'weight': 0.25
                    })
            except Exception as e:
                print(f"⚠️ Tesseract error: {e}")

        # Secondary: EasyOCR
        if self.easyocr_available:
            try:
                easy_results = self.easyocr.readtext(processed_image)
                if easy_results:
                    # Get best result
                    best_result = max(easy_results, key=lambda x: x[2])
                    results.append({
                        'engine': 'easyocr',
                        'text': best_result[1],
                        'confidence': best_result[2],
                        'weight': 0.15
                    })
            except Exception as e:
                print(f"⚠️ EasyOCR error: {e}")

        # Weighted consensus
        return self._weighted_consensus(results)

    def _weighted_consensus(self, results):
        """Create weighted consensus from multiple OCR results."""
        if not results:
            return None

        # If only one result, return it
        if len(results) == 1:
            return results[0]

        # Parse all results and vote
        parsed_results = []
        for result in results:
            parsed = self._parse_down_distance(result['text'])
            if parsed:
                parsed['original_confidence'] = result['confidence']
                parsed['weight'] = result['weight']
                parsed['engine'] = result['engine']
                parsed_results.append(parsed)

        if not parsed_results:
            return results[0]  # Fallback to first result

        # Vote on down and distance separately
        down_votes = {}
        distance_votes = {}

        for result in parsed_results:
            down = result.get('down')
            distance = result.get('distance')
            weight = result['weight']

            if down:
                down_votes[down] = down_votes.get(down, 0) + weight
            if distance is not None:
                distance_votes[distance] = distance_votes.get(distance, 0) + weight

        # Get consensus
        consensus_down = max(down_votes.items(), key=lambda x: x[1])[0] if down_votes else None
        consensus_distance = max(distance_votes.items(), key=lambda x: x[1])[0] if distance_votes else None

        # Calculate ensemble confidence
        total_weight = sum(r['weight'] for r in parsed_results)
        down_confidence = down_votes.get(consensus_down, 0) / total_weight if consensus_down else 0
        distance_confidence = distance_votes.get(consensus_distance, 0) / total_weight if consensus_distance else 0

        ensemble_confidence = (down_confidence + distance_confidence) / 2

        return {
            'engine': 'ensemble',
            'down': consensus_down,
            'distance': consensus_distance,
            'confidence': ensemble_confidence,
            'text': f"{consensus_down}{self._get_ordinal(consensus_down)} & {consensus_distance if consensus_distance != 0 else 'Goal'}",
            'contributing_engines': [r['engine'] for r in parsed_results]
        }

    def _parse_down_distance(self, text):
        """Parse down and distance from OCR text."""
        import re
        pattern = re.compile(r'(\d+)(?:st|nd|rd|th)?\s*&\s*(\d+|Goal|goal)', re.IGNORECASE)
        match = pattern.search(text)

        if match:
            try:
                down = int(match.group(1))
                distance_str = match.group(2).lower()
                distance = 0 if distance_str == 'goal' else int(distance_str)

                if 1 <= down <= 4 and 0 <= distance <= 30:
                    return {'down': down, 'distance': distance}
            except ValueError:
                pass

        return None

    def _get_ordinal(self, number):
        if 10 <= number % 100 <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")

class TemporalOCRFilter:
    """Filter OCR results using temporal consistency."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.recent_results = deque(maxlen=window_size)
        self.stable_state = None
        self.stability_count = 0

    def filter_with_temporal_context(self, ocr_result, frame_number):
        """Filter OCR result using recent frame context."""

        if not ocr_result:
            return ocr_result

        current_down = ocr_result.get('down')
        current_distance = ocr_result.get('distance')

        # Store current result
        self.recent_results.append({
            'frame': frame_number,
            'down': current_down,
            'distance': current_distance,
            'confidence': ocr_result.get('confidence', 0)
        })

        # Need at least 3 frames for temporal filtering
        if len(self.recent_results) < 3:
            return ocr_result

        # Check consistency with recent frames
        recent_downs = [r['down'] for r in self.recent_results if r['down'] is not None]
        recent_distances = [r['distance'] for r in self.recent_results if r['distance'] is not None]

        if not recent_downs:
            return ocr_result

        # Count occurrences
        down_counter = Counter(recent_downs)
        distance_counter = Counter(recent_distances)

        # Get most common values
        most_common_down = down_counter.most_common(1)[0]
        most_common_distance = distance_counter.most_common(1)[0] if recent_distances else (None, 0)

        # Calculate consistency scores
        down_consistency = most_common_down[1] / len(recent_downs)
        distance_consistency = most_common_distance[1] / len(recent_distances) if recent_distances else 0

        # If current result is inconsistent with majority, use majority
        if (current_down != most_common_down[0] and down_consistency >= 0.6):
            print(f"🔄 Temporal filter: {current_down} → {most_common_down[0]} (consistency: {down_consistency:.2f})")
            ocr_result['down'] = most_common_down[0]
            ocr_result['temporal_corrected'] = True

        if (current_distance != most_common_distance[0] and distance_consistency >= 0.6):
            print(f"🔄 Temporal filter: {current_distance} → {most_common_distance[0]} (consistency: {distance_consistency:.2f})")
            ocr_result['distance'] = most_common_distance[0]
            ocr_result['temporal_corrected'] = True

        # Add temporal confidence boost for consistent results
        temporal_boost = (down_consistency + distance_consistency) / 2
        ocr_result['temporal_confidence'] = temporal_boost
        ocr_result['boosted_confidence'] = min(1.0, ocr_result.get('confidence', 0) + temporal_boost * 0.1)

        return ocr_result

class GameLogicValidator:
    """Validate OCR results using Madden game logic."""

    def __init__(self):
        self.previous_state = None
        self.impossible_transitions = set()

    def validate_with_game_logic(self, ocr_result):
        """Apply game logic validation to OCR result."""

        if not ocr_result:
            return ocr_result

        down = ocr_result.get('down')
        distance = ocr_result.get('distance')

        validation_score = 1.0
        validation_notes = []

        # Rule 1: Valid down range
        if down and not (1 <= down <= 4):
            validation_score *= 0.1
            validation_notes.append(f"Invalid down: {down}")

        # Rule 2: Valid distance range
        if distance is not None and not (0 <= distance <= 99):
            validation_score *= 0.1
            validation_notes.append(f"Invalid distance: {distance}")

        # Rule 3: Down progression logic
        if self.previous_state and down:
            prev_down = self.previous_state.get('down')
            if prev_down:
                # Valid transitions: same down, +1, or reset to 1
                valid_transitions = [prev_down, prev_down + 1, 1]
                if down not in valid_transitions:
                    validation_score *= 0.3
                    validation_notes.append(f"Suspicious down transition: {prev_down} → {down}")

        # Rule 4: Common patterns boost
        if down and distance is not None:
            common_patterns = {
                (1, 10): 1.2,   # 1st & 10 is very common
                (2, 7): 1.1,    # 2nd & 7 after 3-yard gain
                (3, 1): 1.1,    # 3rd & 1 short yardage
                (4, 1): 1.05,   # 4th & 1
            }

            pattern = (down, distance)
            if pattern in common_patterns:
                validation_score *= common_patterns[pattern]
                validation_notes.append(f"Common pattern boost: {pattern}")

        # Rule 5: Impossible combinations
        impossible_patterns = {
            (1, 0), (2, 0), (3, 0),  # Can't have 0 yards unless goal line
            (4, 25),  # 4th & 25+ is extremely rare
        }

        if down and distance is not None:
            pattern = (down, distance)
            if pattern in impossible_patterns:
                validation_score *= 0.2
                validation_notes.append(f"Rare/impossible pattern: {pattern}")

        # Apply validation score
        original_confidence = ocr_result.get('confidence', 0)
        validated_confidence = original_confidence * validation_score

        ocr_result['validation_score'] = validation_score
        ocr_result['validated_confidence'] = validated_confidence
        ocr_result['validation_notes'] = validation_notes

        if validation_notes:
            print(f"🎮 Game logic validation: {validation_notes}")

        # Update previous state
        self.previous_state = {'down': down, 'distance': distance}

        return ocr_result

class SuperEnhancedOCR:
    """Complete enhanced OCR system building on optimized preprocessing."""

    def __init__(self, optimized_paddle_ocr):
        self.ensemble = OCREnsemble(optimized_paddle_ocr)
        self.temporal_filter = TemporalOCRFilter(window_size=5)
        self.game_validator = GameLogicValidator()

        print("🚀 SuperEnhancedOCR initialized")
        print("   Building on optimized preprocessing (0.939 score)")
        print("   Added: Ensemble + Temporal + Game Logic validation")

    def extract_enhanced(self, processed_image, frame_number):
        """Full enhanced extraction pipeline."""

        # 1. Ensemble prediction (builds on optimized preprocessing)
        ensemble_result = self.ensemble.ensemble_predict(processed_image)

        if not ensemble_result:
            return None

        # 2. Temporal consistency filtering
        filtered_result = self.temporal_filter.filter_with_temporal_context(
            ensemble_result, frame_number
        )

        # 3. Game logic validation
        validated_result = self.game_validator.validate_with_game_logic(filtered_result)

        # 4. Final confidence calculation
        final_confidence = self._calculate_final_confidence(validated_result)
        validated_result['final_confidence'] = final_confidence

        return validated_result

    def _calculate_final_confidence(self, result):
        """Calculate final confidence from all enhancement stages."""
        base_confidence = result.get('confidence', 0)
        temporal_confidence = result.get('temporal_confidence', 0)
        validation_score = result.get('validation_score', 1.0)

        # Weighted combination
        final_confidence = (
            base_confidence * 0.5 +           # Base OCR confidence
            temporal_confidence * 0.3 +       # Temporal consistency
            validation_score * 0.2            # Game logic validation
        )

        return min(1.0, final_confidence) 