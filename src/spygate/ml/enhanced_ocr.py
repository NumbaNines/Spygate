"""
Enhanced OCR module for SpygateAI.
Handles text detection and recognition from game HUD elements.
"""

import logging
from typing import Dict, Optional, Tuple, List, Any
import re
import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class OCRValidation:
    """Validation parameters for OCR results."""
    min_confidence: float = 0.7
    max_retries: int = 3
    history_size: int = 5
    temporal_threshold: float = 0.8
    yard_line_max_change: int = 10
    min_yard_line_confidence: float = 0.4
    territory_consistency_window: int = 3

@dataclass
class OCRResult:
    """OCR result with confidence and source."""
    text: str
    confidence: float
    source: str
    bbox: Optional[List[int]] = None

class EnhancedOCR:
    """
    Enhanced OCR processor with multi-engine fallback and game-specific optimizations.
    
    Features:
    - Multi-engine OCR (EasyOCR primary, Tesseract fallback)
    - Game-specific text preprocessing
    - Robust error handling and validation
    - Confidence scoring
    - Temporal smoothing
    - Partial occlusion handling
    """
    
    def __init__(self, hardware=None):
        """Initialize OCR engines and configurations.
        
        Args:
            hardware: Hardware tier for adaptive settings
        """
        self.hardware_tier = hardware
        self.validation = OCRValidation()
        
        # Initialize OCR engines with fallback
        try:
            self.reader = easyocr.Reader(['en'])
            logger.info("Initialized EasyOCR engine")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
        
        # Historical tracking for temporal smoothing
        self.history = {
            'down': deque(maxlen=self.validation.history_size),
            'distance': deque(maxlen=self.validation.history_size),
            'score_home': deque(maxlen=self.validation.history_size),
            'score_away': deque(maxlen=self.validation.history_size),
            'quarter': deque(maxlen=self.validation.history_size),
            'time': deque(maxlen=self.validation.history_size),
            'yard_line': deque(maxlen=self.validation.history_size),
            'territory': deque(maxlen=self.validation.history_size)  # Track field territory
        }
        
        # Common game text patterns with validation rules
        self.patterns = {
            'down': {
                'pattern': r'(\d)(st|nd|rd|th)',
                'validate': lambda x: 1 <= int(x) <= 4,
                'format': lambda x: int(x)
            },
            'distance': {
                'pattern': r'(\d+)',
                'validate': lambda x: 1 <= int(x) <= 99,
                'format': lambda x: int(x)
            },
            'score': {
                'pattern': r'(\d{1,2})',
                'validate': lambda x: 0 <= int(x) <= 99,
                'format': lambda x: int(x)
            },
            'quarter': {
                'pattern': r'(\d)(st|nd|rd|th)',
                'validate': lambda x: 1 <= int(x) <= 5,  # Including overtime
                'format': lambda x: int(x)
            },
            'time': {
                'pattern': r'(\d{1,2}):(\d{2})',
                'validate': lambda m, s: 0 <= int(m) <= 15 and 0 <= int(s) <= 59,
                'format': lambda m, s: f"{int(m):02d}:{int(s):02d}"
            },
            'yard_line': {
                'pattern': r'(?:OWN |OPP )?(\d{1,2})',  # Now handles "OWN" and "OPP" prefixes
                'validate': lambda x, t: self._validate_yard_line(x, t),
                'format': lambda x, t: self._format_yard_line(x, t)
            },
            'territory': {
                'pattern': r'(OWN|OPP)',
                'validate': lambda x: x in ['OWN', 'OPP'],
                'format': lambda x: x
            }
        }

        # Common OCR correction mappings for yard lines
        self.yard_line_corrections = {
            '0': ['o', 'O', 'D', 'Q'],
            '1': ['l', 'I', '|'],
            '2': ['z', 'Z'],
            '3': ['8', 'B'],
            '5': ['S', 's'],
            '6': ['b'],
            '8': ['3', 'B'],
            '9': ['g', 'q']
        }
        
        # Define fixed HUD text patterns
        self.text_patterns = {
            'team_abbrev': r'^[A-Z]{3}$',  # 3 uppercase letters
            'score': r'^\d{1,2}$',         # 1-2 digits
            'down': r'^[1-4]$',            # 1-4
            'distance': r'^\d{1,3}$',      # 1-3 digits
            'yard_line': r'^\d{1,2}$'      # Just the yard line number (1-50)
        }
        
        # Define relative ROI coordinates (percentages of HUD region height/width)
        self.roi_regions = {
            'left_team': {'x': (0.05, 0.15), 'y': (0.3, 0.7)},
            'left_score': {'x': (0.15, 0.25), 'y': (0.3, 0.7)},
            'right_team': {'x': (0.3, 0.4), 'y': (0.3, 0.7)},
            'right_score': {'x': (0.4, 0.5), 'y': (0.3, 0.7)},
            'down_distance': {'x': (0.45, 0.65), 'y': (0.3, 0.7)},
            'yard_line': {'x': (0.85, 0.95), 'y': (0.3, 0.7)}  # Just the number, territory triangle detected separately
        }
    
    def _validate_yard_line(self, yard_line: str, territory: Optional[str] = None) -> bool:
        """
        Enhanced yard line validation with territory context.
        
        Args:
            yard_line: Detected yard line number
            territory: Field territory indicator ('OWN' or 'OPP')
            
        Returns:
            bool: Whether the yard line is valid
        """
        try:
            # Clean and correct common OCR mistakes
            yard_line = self._correct_yard_line(yard_line)
            
            # Convert to int and validate basic range
            yard = int(yard_line)
            if not (1 <= yard <= 50):
                return False
                
            # If we have history, check for reasonable progression
            if self.history['yard_line'] and len(self.history['yard_line']) > 0:
                last_yard = int(self.history['yard_line'][-1])
                if abs(yard - last_yard) > self.validation.yard_line_max_change:
                    logger.warning(f"Suspicious yard line change: {last_yard} -> {yard}")
                    return False
            
            # If we have territory context, validate consistency
            if territory and self.history['territory'] and len(self.history['territory']) > 0:
                last_territory = self.history['territory'][-1]
                if territory != last_territory and yard == int(self.history['yard_line'][-1]):
                    # Territory changed but yard line didn't - suspicious
                    return False
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Yard line validation error: {e}")
            return False
    
    def _format_yard_line(self, yard_line: str, territory: Optional[str] = None) -> int:
        """Format yard line with territory context."""
        try:
            yard = int(self._correct_yard_line(yard_line))
            if territory == 'OPP':
                # Store territory for future validation
                self.history['territory'].append('OPP')
            elif territory == 'OWN':
                self.history['territory'].append('OWN')
            
            # Store yard line for future validation
            self.history['yard_line'].append(str(yard))
            return yard
        except (ValueError, TypeError):
            return None
    
    def _correct_yard_line(self, text: str) -> str:
        """Apply common OCR corrections for yard lines."""
        text = text.strip()
        
        # Apply corrections
        for digit, mistakes in self.yard_line_corrections.items():
            for mistake in mistakes:
                text = text.replace(mistake, digit)
        
        return text

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR with enhanced yard line detection.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Store original dimensions
            original_height, original_width = image.shape[:2]
            
            # Ensure we're working with RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Increase image size for better OCR
            scale_factor = 2
            enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(enlarged, 9, 75, 75)
            
            # Enhance contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Sharpen edges
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Use Otsu's thresholding for better binarization
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up small noise
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Dilate slightly to connect broken characters
            kernel = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(cleaned, kernel, iterations=1)
            
            # Resize back to original dimensions
            resized = cv2.resize(dilated, (original_width, original_height), interpolation=cv2.INTER_AREA)
            
            # Convert back to BGR for OCR
            processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image
    
    def process_region(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image region and extract text information.
        
        Args:
            image: Image region to process
            
        Returns:
            Dict containing extracted information
        """
        try:
            # Preprocess image for OCR
            processed = self.preprocess_image(image)
            
            # Run OCR
            results = self.reader.readtext(processed)
            
            # Initialize results dictionary
            extracted_info = {}
            confidence_scores = []
            
            # First pass - look for territory indicators
            territory = None
            max_territory_conf = 0
            for bbox, text, conf in results:
                text = text.strip().upper()
                if match := re.search(self.patterns['territory']['pattern'], text):
                    detected_territory = match.group(1)
                    # Only update territory if confidence is higher
                    if conf > max_territory_conf:
                        territory = detected_territory
                        max_territory_conf = conf
                        confidence_scores.append(conf)
            
            # Second pass - extract yard line with multiple attempts
            yard_line_candidates = []
            for bbox, text, conf in results:
                text = text.strip()
                
                # Try to find yard line numbers
                if match := re.search(self.patterns['yard_line']['pattern'], text):
                    value = match.group(1)
                    
                    # Clean and validate the yard line
                    if self.patterns['yard_line']['validate'](value, territory):
                        formatted_value = self.patterns['yard_line']['format'](value, territory)
                        if formatted_value is not None:
                            yard_line_candidates.append((formatted_value, conf))
            
            # Select the most likely yard line based on:
            # 1. Historical consistency
            # 2. Confidence score
            # 3. Territory context
            if yard_line_candidates:
                # Sort by confidence
                yard_line_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # If we have history, prefer candidates close to previous value
                if self.history['yard_line'] and len(self.history['yard_line']) > 0:
                    last_yard = int(self.history['yard_line'][-1])
                    valid_candidates = [
                        (yard, conf) for yard, conf in yard_line_candidates
                        if abs(yard - last_yard) <= self.validation.yard_line_max_change
                    ]
                    
                    if valid_candidates:
                        yard_line_candidates = valid_candidates
                
                # Use the best remaining candidate
                best_yard, best_conf = yard_line_candidates[0]
                extracted_info['yard_line'] = best_yard
                confidence_scores.append(best_conf)
                
                # Update history
                self.history['yard_line'].append(str(best_yard))
            
            # Add territory to results if found
            if territory:
                extracted_info['territory'] = territory
                self.history['territory'].append(territory)
            
            # Add confidence score if we have any matches
            if confidence_scores:
                extracted_info['confidence'] = sum(confidence_scores) / len(confidence_scores)
            
            return extracted_info
        except Exception as e:
            logger.error(f"Error processing region: {e}")
            return {}
    
    def _validate_and_smooth(self, text_info: Dict[str, Any], region_type: str) -> Dict[str, Any]:
        """
        Validate OCR results and apply temporal smoothing.
        
        Args:
            text_info: Dictionary of extracted text
            region_type: Type of HUD region
            
        Returns:
            Validated and smoothed text information
        """
        validated = {}
        
        for key, value in text_info.items():
            if not value:
                continue
                
            pattern_info = self.patterns.get(key)
            if not pattern_info:
                continue
            
            try:
                # Apply validation rules
                if key == 'time':
                    minutes, seconds = value.split(':')
                    if pattern_info['validate'](minutes, seconds):
                        formatted = pattern_info['format'](minutes, seconds)
                        self.history[key].append(formatted)
                        validated[key] = formatted
                else:
                    if pattern_info['validate'](value):
                        formatted = pattern_info['format'](value)
                        self.history[key].append(formatted)
                        validated[key] = formatted
                        
            except Exception as e:
                logger.warning(f"Validation failed for {key}: {e}")
                
        return validated if validated else None
    
    def _get_best_historical_values(self, region_type: str = None) -> Dict[str, Any]:
        """
        Get most reliable values from history when current detection fails.
        
        Args:
            region_type: Type of HUD region
            
        Returns:
            Dictionary of best historical values
        """
        historical = {}
        
        for key, history in self.history.items():
            if not history:
                continue
                
            if region_type and not key.startswith(region_type):
                continue
                
            # Get most common value in recent history
            values = list(history)
            if len(values) >= self.validation.history_size * self.validation.temporal_threshold:
                from collections import Counter
                most_common = Counter(values).most_common(1)
                if most_common:
                    historical[key] = most_common[0][0]
        
        return historical

    def handle_partial_occlusion(self, img: np.ndarray, visible_regions: List[str]) -> Dict[str, Any]:
        """
        Handle cases where HUD is partially occluded.
        
        Args:
            img: Input image array
            visible_regions: List of visible HUD regions
            
        Returns:
            Best-effort OCR results using visible regions
        """
        results = {}
        
        for region in visible_regions:
            # Process visible regions
            region_results = self.process_region(img)
            results.update(region_results)
            
        # Fill in missing data from history
        historical = self._get_best_historical_values()
        
        # Only use historical values for missing fields
        for key, value in historical.items():
            if key not in results:
                results[key] = value
        
        return results

    def _parse_ocr_results(self, results: list, region_type: str) -> Dict[str, Optional[str]]:
        """Parse OCR results and extract game information."""
        text_info = {}
        
        for (bbox, text, conf) in results:
            text = text.strip().lower()
            
            # Extract down
            if match := re.search(self.patterns['down']['pattern'], text):
                text_info['down'] = int(match.group(1))
            
            # Extract distance
            if match := re.search(self.patterns['distance']['pattern'], text):
                text_info['distance'] = int(match.group(1))
            
            # Extract score
            if match := re.search(self.patterns['score']['pattern'], text):
                # Determine if home or away based on position
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                if x_center < bbox[2][0] / 2:  # Using bbox width instead of img.shape
                    text_info['score_away'] = int(match.group(1))
                else:
                    text_info['score_home'] = int(match.group(1))
            
            # Extract quarter
            if match := re.search(self.patterns['quarter']['pattern'], text):
                text_info['quarter'] = int(match.group(1))
            
            # Extract time
            if match := re.search(self.patterns['time']['pattern'], text):
                text_info['time'] = f"{match.group(1)}:{match.group(2)}"
            
            # Extract territory first (needed for yard line context)
            territory = None
            if match := re.search(self.patterns['territory']['pattern'], text):
                territory = match.group(1).upper()
                if self.patterns['territory']['validate'](territory):
                    text_info['territory'] = territory
            
            # Extract yard line with territory context
            if match := re.search(self.patterns['yard_line']['pattern'], text):
                yard_line = match.group(1)
                if self.patterns['yard_line']['validate'](yard_line, territory):
                    text_info['yard_line'] = self.patterns['yard_line']['format'](yard_line, territory)
        
        return text_info
    
    def _parse_text(self, text: str, region_type: str) -> Dict[str, Optional[str]]:
        """Parse raw text and extract game information."""
        text_info = {}
        text = text.lower()
        
        # Extract territory first for yard line context
        territory = None
        if match := re.search(self.patterns['territory']['pattern'], text):
            territory = match.group(1).upper()
            if self.patterns['territory']['validate'](territory):
                text_info['territory'] = territory
        
        # Apply pattern matching for the specific region type
        if region_type == 'yard_line':
            if match := re.search(self.patterns['yard_line']['pattern'], text):
                yard_line = match.group(1)
                if self.patterns['yard_line']['validate'](yard_line, territory):
                    text_info['yard_line'] = self.patterns['yard_line']['format'](yard_line, territory)
        elif pattern_info := self.patterns.get(region_type):
            if match := re.search(pattern_info['pattern'], text):
                if region_type == 'time':
                    text_info[region_type] = f"{match.group(1)}:{match.group(2)}"
                else:
                    text_info[region_type] = int(match.group(1))
        
        return text_info 

    def process_hud_region(self, hud_region: np.ndarray) -> Dict[str, Any]:
        """Process specific regions of HUD for optimal OCR performance."""
        results = {}
        h, w = hud_region.shape[:2]
        
        for region_name, coords in self.roi_regions.items():
            # Calculate absolute coordinates
            x1 = int(w * coords['x'][0])
            x2 = int(w * coords['x'][1])
            y1 = int(h * coords['y'][0])
            y2 = int(h * coords['y'][1])
            
            # Extract ROI
            roi = hud_region[y1:y2, x1:x2]
            
            # Apply preprocessing based on region type
            if 'score' in region_name or 'down' in region_name:
                # Optimize for number detection
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)[1]
            elif 'team' in region_name:
                # Optimize for text detection
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi = cv2.GaussianBlur(roi, (3,3), 0)
            
            # Perform OCR
            ocr_result = self.reader.readtext(roi)
            
            if ocr_result:
                text = ocr_result[0][1]
                conf = ocr_result[0][2]
                
                # Validate against expected patterns
                if region_name in ['left_team', 'right_team']:
                    if re.match(self.text_patterns['team_abbrev'], text):
                        results[region_name] = {'text': text, 'confidence': conf}
                elif 'score' in region_name:
                    if re.match(self.text_patterns['score'], text):
                        results[region_name] = {'text': int(text), 'confidence': conf}
                elif region_name == 'down_distance':
                    # Parse down and distance
                    down_match = re.search(r'([1-4])', text)
                    dist_match = re.search(r'(\d{1,3})', text)
                    if down_match and dist_match:
                        results['down'] = {'text': int(down_match.group(1)), 'confidence': conf}
                        results['distance'] = {'text': int(dist_match.group(1)), 'confidence': conf}
                elif region_name == 'yard_line':
                    if re.match(self.text_patterns['yard_line'], text):
                        results[region_name] = {'text': text, 'confidence': conf}
        
        return results

    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OCR results against known game constraints."""
        validated = results.copy()
        
        # Validate scores (0-99)
        for score_key in ['left_score', 'right_score']:
            if score_key in validated:
                score = validated[score_key]['text']
                if not (0 <= score <= 99):
                    validated.pop(score_key)
        
        # Validate down (1-4)
        if 'down' in validated:
            down = validated['down']['text']
            if not (1 <= down <= 4):
                validated.pop('down')
        
        # Validate distance (1-99)
        if 'distance' in validated:
            distance = validated['distance']['text']
            if not (1 <= distance <= 99):
                validated.pop('distance')
        
        # Validate yard line (1-50)
        if 'yard_line' in validated:
            try:
                yard = int(validated['yard_line']['text'])
                # Must be between 1-50 (no 0 yard line in football)
                if not (1 <= yard <= 50):
                    validated.pop('yard_line')
            except ValueError:
                validated.pop('yard_line')
        
        return validated 