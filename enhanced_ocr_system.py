#!/usr/bin/env python3
"""
Enhanced OCR System for SpygateAI
Significantly improves text extraction accuracy through:
- Advanced preprocessing pipelines
- Multi-engine OCR (EasyOCR + Tesseract)
- Sport-specific text validation
- Intelligent confidence scoring
- Game-specific HUD text patterns
- Production-grade error handling and recovery
"""

import cv2
import numpy as np
import re
import sys
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum

# OCR Engine availability flags
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Configure logging
logger = logging.getLogger(__name__)

class OCRError(Exception):
    """Base exception for OCR-related errors."""
    pass

class OCREngineError(OCRError):
    """Raised when OCR engine initialization or processing fails."""
    pass

class ValidationError(OCRError):
    """Raised when text validation fails critically."""
    pass

class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails."""
    pass

class EngineStatus(Enum):
    """Status of OCR engines."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    FALLBACK = "fallback"

@dataclass
class OCRResult:
    """Structured OCR result with comprehensive metadata."""
    text: str = ""
    confidence: float = 0.0
    engine: str = "none"
    preprocessing: str = "none"
    validation_score: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False
    raw_results: List[Dict] = field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """Check if OCR was successful."""
        return bool(self.text) and self.confidence > 0.0 and not self.error
    
    @property
    def final_confidence(self) -> float:
        """Get weighted confidence score including validation."""
        if self.validation_score > 0:
            return (self.confidence * 0.7) + (self.validation_score * 0.3)
        return self.confidence

class EnhancedOCRSystem:
    """
    Production-grade OCR system designed specifically for football game HUD text extraction.
    Features multiple preprocessing strategies, dual OCR engines, comprehensive error handling,
    and intelligent fallback mechanisms.
    """
    
    def __init__(self, gpu_enabled: bool = True, debug: bool = False, 
                 max_retries: int = 3, fallback_enabled: bool = True):
        """Initialize the enhanced OCR system with robust error handling."""
        self.debug = debug
        self.gpu_enabled = gpu_enabled
        self.max_retries = max_retries
        self.fallback_enabled = fallback_enabled
        
        # Engine status tracking
        self.engine_status = {
            'easyocr': EngineStatus.UNAVAILABLE,
            'tesseract': EngineStatus.UNAVAILABLE
        }
        
        # Performance metrics
        self.performance_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'engine_failures': {'easyocr': 0, 'tesseract': 0},
            'fallback_uses': 0,
            'average_confidence': 0.0
        }
        
        # Initialize OCR engines with error handling
        self.easyocr_reader = None
        self._initialize_ocr_engines()
        
        # Game-specific text patterns for validation
        self.text_patterns = self._initialize_text_patterns()
        
        # Common OCR errors and corrections
        self.ocr_corrections = self._initialize_ocr_corrections()
        
        self._log_initialization_status()
        
    def _initialize_ocr_engines(self):
        """Initialize both OCR engines with comprehensive error handling."""
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                logger.info("Initializing EasyOCR...")
                import time
                start_time = time.time()
                
                self.easyocr_reader = easyocr.Reader(['en'], gpu=self.gpu_enabled, verbose=False)
                
                # Test the reader with a simple image
                test_img = np.ones((50, 100, 3), dtype=np.uint8) * 255
                test_result = self.easyocr_reader.readtext(test_img)
                
                init_time = time.time() - start_time
                self.engine_status['easyocr'] = EngineStatus.AVAILABLE
                logger.info(f"EasyOCR initialized successfully in {init_time:.2f}s")
                
            except Exception as e:
                self.engine_status['easyocr'] = EngineStatus.FAILED
                logger.error(f"EasyOCR initialization failed: {e}")
                if self.debug:
                    traceback.print_exc()
                self.easyocr_reader = None
        else:
            logger.warning("EasyOCR not available - install with: pip install easyocr")
        
        # Test Tesseract
        if TESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                self.engine_status['tesseract'] = EngineStatus.AVAILABLE
                logger.info(f"Tesseract {version} available")
                
                # Test with simple extraction
                test_img = np.ones((50, 100), dtype=np.uint8) * 255
                test_result = pytesseract.image_to_string(test_img)
                
            except Exception as e:
                self.engine_status['tesseract'] = EngineStatus.FAILED
                logger.error(f"Tesseract test failed: {e}")
                if self.debug:
                    traceback.print_exc()
        else:
            logger.warning("Tesseract not available - install with: pip install pytesseract")
    
    def _log_initialization_status(self):
        """Log the initialization status."""
        available_engines = [name for name, status in self.engine_status.items() 
                           if status == EngineStatus.AVAILABLE]
        
        print(f"🚀 Enhanced OCR System initialized")
        print(f"   • EasyOCR: {'✅' if self.engine_status['easyocr'] == EngineStatus.AVAILABLE else '❌'}")
        print(f"   • Tesseract: {'✅' if self.engine_status['tesseract'] == EngineStatus.AVAILABLE else '❌'}")
        print(f"   • GPU Enabled: {'✅' if self.gpu_enabled else '❌'}")
        print(f"   • Available Engines: {len(available_engines)}")
        
        if not available_engines:
            logger.critical("No OCR engines available! System will use fallback methods.")
        elif len(available_engines) == 1:
            logger.warning(f"Only {available_engines[0]} available. Consider installing both engines for redundancy.")

    def _initialize_text_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize sport-specific text validation patterns."""
        return {
            'down_distance': [
                re.compile(r'^([1-4])(?:st|nd|rd|th)?\s*[&\s]+\s*(\d{1,2})$', re.IGNORECASE),
                re.compile(r'^([1-4])\s*&\s*(\d+)$', re.IGNORECASE),
                re.compile(r'^(\d+)\s*[&]\s*(\d+)$', re.IGNORECASE),
            ],
            'score': [
                re.compile(r'^(\d{1,2})\s*[-:]\s*(\d{1,2})$'),
                re.compile(r'^(\d{1,2})\s+(\d{1,2})$'),
                re.compile(r'^HOME\s*(\d+)\s*[-]\s*AWAY\s*(\d+)$', re.IGNORECASE),
            ],
            'time': [
                re.compile(r'^(\d{1,2}):(\d{2})$'),
                re.compile(r'^(\d{1,2})\.(\d{2})$'),
                re.compile(r'^(\d{1,2}):(\d{2})\s*(Q[1-4]|OT)$', re.IGNORECASE),
            ],
            'quarter': [
                re.compile(r'^(Q[1-4]|OT|1st|2nd|3rd|4th)$', re.IGNORECASE),
                re.compile(r'^([1-4])(?:st|nd|rd|th)?\s*QTR$', re.IGNORECASE),
            ],
            'team_names': [
                re.compile(r'^[A-Z]{2,4}$'),  # Team abbreviations
                re.compile(r'^[A-Z][a-z]+$'),  # Team names
            ]
        }
    
    def _initialize_ocr_corrections(self) -> Dict[str, str]:
        """Common OCR misreads and their corrections."""
        return {
            # Common character mistakes
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'S': '5', 's': '5',
            'B': '8', 'G': '6',
            'Z': '2', 'z': '2',
            # Common down/distance mistakes
            '1st': '1ST', '2nd': '2ND', '3rd': '3RD', '4th': '4TH',
            'lst': '1ST', '2na': '2ND', '3ra': '3RD', '4tn': '4TH',
            'Ist': '1ST', 'znd': '2ND', 'Srd': '3RD',
            # Special characters
            'a': '&', 'e': '&', '@': '&',
            # Common team name fixes
            'HOME': 'HOME', 'AWAY': 'AWAY',
        }

    def enhance_image_for_ocr(self, image: np.ndarray, enhancement_type: str = "adaptive") -> List[np.ndarray]:
        """Enhance image with multiple preprocessing strategies for OCR."""
        try:
            # Validate and sanitize input image
            if image is None or image.size == 0:
                logger.error("Critical error in image enhancement: Input image is empty or None")
                raise ImageProcessingError("Input image is empty or None")
                
            # Handle corrupted or invalid image data
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                logger.warning("Image contains NaN or infinite values, cleaning...")
                image = np.nan_to_num(image, nan=127, posinf=255, neginf=0)
            
            # Ensure proper data type and range
            if image.dtype != np.uint8:
                logger.warning(f"Converting image from {image.dtype} to uint8")
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Validate image shape
            if len(image.shape) not in [2, 3] or image.shape[0] == 0 or image.shape[1] == 0:
                raise ImageProcessingError(f"Invalid image shape: {image.shape}")
            
            # Handle corrupted or invalid data types
            try:
                # Convert to proper format if needed
                if image.dtype not in [np.uint8, np.float32, np.float64]:
                    # Try to convert to uint8
                    if np.issubdtype(image.dtype, np.integer):
                        image = np.clip(image, 0, 255).astype(np.uint8)
                    else:
                        # For float types, normalize to 0-255 range
                        image_min, image_max = np.nanmin(image), np.nanmax(image)
                        if np.isfinite(image_min) and np.isfinite(image_max) and image_max > image_min:
                            image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                        else:
                            raise ImageProcessingError("Image contains invalid values (NaN/Inf) that cannot be normalized")
                
                # Handle special values
                if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                    # Replace NaN and Inf with valid values
                    image = np.nan_to_num(image, nan=0, posinf=255, neginf=0).astype(np.uint8)
                
                # Ensure values are in valid range
                image = np.clip(image, 0, 255).astype(np.uint8)
                
            except Exception as e:
                logger.error(f"Failed to sanitize image data: {e}")
                raise ImageProcessingError(f"Cannot process corrupted image data: {e}")
            
            # Ensure image has valid dimensions
            if len(image.shape) not in [2, 3]:
                logger.error(f"Invalid image dimensions: {image.shape}")
                raise ImageProcessingError(f"Unsupported image dimensions: {image.shape}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    logger.warning(f"Color conversion failed, using alternative method: {e}")
                    # Alternative grayscale conversion for edge cases
                    if image.shape[2] >= 3:
                        gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
                    else:
                        gray = image[:, :, 0]  # Use first channel
            else:
                gray = image
            
            # Quick quality check
            if gray.std() < 5:  # Very low contrast
                logger.warning("Very low contrast image detected")
            
            # Return simple grayscale for fallback cases or specific request
            if enhancement_type == "simple":
                return [gray]
            
            enhanced_images = []
            
            # Strategy 1: Original grayscale
            enhanced_images.append(gray)
            
            # Strategy 2: Contrast enhancement
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                contrast_enhanced = clahe.apply(gray)
                enhanced_images.append(contrast_enhanced)
            except Exception as e:
                logger.warning(f"Contrast enhancement failed: {e}")
            
            # Strategy 3: Adaptive thresholding
            try:
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                enhanced_images.append(adaptive_thresh)
            except Exception as e:
                logger.warning(f"Adaptive thresholding failed: {e}")
            
            # Strategy 4: Otsu's thresholding
            try:
                _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                enhanced_images.append(otsu_thresh)
            except Exception as e:
                logger.warning(f"Otsu thresholding failed: {e}")
            
            # Strategy 5: Morphological operations
            try:
                kernel = np.ones((2, 2), np.uint8)
                morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                enhanced_images.append(morphed)
            except Exception as e:
                logger.warning(f"Morphological operations failed: {e}")
            
            return enhanced_images if enhanced_images else [gray]
            
        except ImageProcessingError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"Critical error in image enhancement: {e}")
            # Last resort - try to return basic grayscale conversion
            try:
                if image is not None and image.size > 0:
                    return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]
                else:
                    raise ImageProcessingError("Cannot process image: Input is empty or None")
            except:
                raise ImageProcessingError(f"Cannot process image: {e}")

    def extract_text_multi_engine(self, image: np.ndarray, text_type: Optional[str] = None) -> List[OCRResult]:
        """
        Extract text using multiple OCR engines and preprocessing strategies with comprehensive error handling.
        
        Args:
            image: Input image
            text_type: Expected text type for validation (optional)
            
        Returns:
            List of OCR results sorted by confidence
        """
        results = []
        self.performance_stats['total_extractions'] += 1
        
        try:
            # Get multiple preprocessed versions
            enhanced_images = self.enhance_image_for_ocr(image, "all")
            
            for i, processed_img in enumerate(enhanced_images):
                preprocessing_method = ["original", "binary", "adaptive", "otsu", "enhanced", "denoised", "morphology"][i]
                
                # Try EasyOCR
                if self.engine_status['easyocr'] == EngineStatus.AVAILABLE:
                    for attempt in range(self.max_retries):
                        easyocr_result = self._extract_with_easyocr(processed_img, preprocessing_method)
                        if easyocr_result and easyocr_result.is_successful:
                            results.append(easyocr_result)
                            break
                        elif attempt == self.max_retries - 1:
                            self.performance_stats['engine_failures']['easyocr'] += 1
                
                # Try Tesseract
                if self.engine_status['tesseract'] == EngineStatus.AVAILABLE:
                    for attempt in range(self.max_retries):
                        tesseract_result = self._extract_with_tesseract(processed_img, preprocessing_method, text_type)
                        if tesseract_result and tesseract_result.is_successful:
                            results.append(tesseract_result)
                            break
                        elif attempt == self.max_retries - 1:
                            self.performance_stats['engine_failures']['tesseract'] += 1
            
            # Sort by final confidence
            results = sorted(results, key=lambda x: x.final_confidence, reverse=True)
            
            # Apply text validation if type specified
            if text_type and text_type in self.text_patterns:
                validated_results = []
                for result in results:
                    try:
                        validation_score = self._validate_text(result.text, text_type)
                        result.validation_score = validation_score
                        validated_results.append(result)
                    except ValidationError as e:
                        logger.warning(f"Validation failed for '{result.text}': {e}")
                        result.validation_score = 0.0
                        validated_results.append(result)
                
                results = sorted(validated_results, key=lambda x: x.final_confidence, reverse=True)
            
            if results:
                self.performance_stats['successful_extractions'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in multi-engine extraction: {e}")
            if self.debug:
                traceback.print_exc()
            return [OCRResult(error=f"Multi-engine extraction failed: {e}")]

    def _extract_with_easyocr(self, image: np.ndarray, preprocessing_method: str) -> Optional[OCRResult]:
        """Extract text using EasyOCR with comprehensive error handling."""
        if self.engine_status['easyocr'] != EngineStatus.AVAILABLE:
            return None
        
        try:
            import time
            start_time = time.time()
            
            ocr_results = self.easyocr_reader.readtext(image, detail=1, paragraph=False)
            processing_time = time.time() - start_time
            
            if not ocr_results:
                return OCRResult(
                    engine='EasyOCR',
                    preprocessing=preprocessing_method,
                    processing_time=processing_time,
                    error="No text detected"
                )
            
            # Combine all text and calculate average confidence
            texts = []
            confidences = []
            
            for bbox, text, confidence in ocr_results:
                if confidence > 0.1:  # Very low threshold for inclusion
                    texts.append(text.strip())
                    confidences.append(confidence)
            
            if not texts:
                return OCRResult(
                    engine='EasyOCR',
                    preprocessing=preprocessing_method,
                    processing_time=processing_time,
                    error="No high-confidence text found"
                )
            
            combined_text = " ".join(texts)
            avg_confidence = np.mean(confidences)
            
            # Apply text corrections
            corrected_text = self._apply_corrections(combined_text)
            
            return OCRResult(
                text=corrected_text,
                confidence=avg_confidence,
                engine='EasyOCR',
                preprocessing=preprocessing_method,
                processing_time=processing_time,
                raw_results=[{
                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                    'text': text,
                    'confidence': conf
                } for bbox, text, conf in ocr_results]
            )
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            if self.debug:
                traceback.print_exc()
            
            # Mark engine as failed if consistent failures
            self.performance_stats['engine_failures']['easyocr'] += 1
            if self.performance_stats['engine_failures']['easyocr'] > 10:
                self.engine_status['easyocr'] = EngineStatus.FALLBACK
                logger.warning("EasyOCR marked as fallback due to repeated failures")
            
            return OCRResult(
                engine='EasyOCR',
                preprocessing=preprocessing_method,
                error=f"EasyOCR failed: {e}"
            )

    def _extract_with_tesseract(self, image: np.ndarray, preprocessing_method: str, 
                               text_type: Optional[str] = None) -> Optional[OCRResult]:
        """Extract text using Tesseract with optimized configurations and error handling."""
        if self.engine_status['tesseract'] != EngineStatus.AVAILABLE:
            return None
        
        try:
            import time
            start_time = time.time()
            
            # Configure Tesseract based on text type
            configs = self._get_tesseract_configs(text_type)
            
            best_result = None
            best_confidence = 0
            
            for config_name, config in configs.items():
                try:
                    # Extract text
                    text = pytesseract.image_to_string(image, config=config).strip()
                    
                    if not text:
                        continue
                    
                    # Get confidence data
                    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [float(conf) for conf in data['conf'] if conf != '-1' and float(conf) > 0]
                    
                    if not confidences:
                        continue
                    
                    avg_confidence = np.mean(confidences)
                    
                    # Apply corrections
                    corrected_text = self._apply_corrections(text)
                    
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        processing_time = time.time() - start_time
                        best_result = OCRResult(
                            text=corrected_text,
                            confidence=avg_confidence / 100.0,  # Normalize to 0-1
                            engine='Tesseract',
                            preprocessing=preprocessing_method,
                            processing_time=processing_time,
                            raw_results=[{
                                'config': config_name,
                                'raw_text': text,
                                'confidences': confidences
                            }]
                        )
                
                except Exception as e:
                    logger.warning(f"Tesseract config {config_name} failed: {e}")
                    continue
            
            if best_result:
                return best_result
            else:
                return OCRResult(
                    engine='Tesseract',
                    preprocessing=preprocessing_method,
                    processing_time=time.time() - start_time,
                    error="No valid text extracted with any configuration"
                )
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            if self.debug:
                traceback.print_exc()
                
            # Mark engine as failed if consistent failures
            self.performance_stats['engine_failures']['tesseract'] += 1
            if self.performance_stats['engine_failures']['tesseract'] > 10:
                self.engine_status['tesseract'] = EngineStatus.FALLBACK
                logger.warning("Tesseract marked as fallback due to repeated failures")
            
            return OCRResult(
                engine='Tesseract',
                preprocessing=preprocessing_method,
                error=f"Tesseract failed: {e}"
            )

    def _get_tesseract_configs(self, text_type: Optional[str] = None) -> Dict[str, str]:
        """Get optimized Tesseract configurations for different text types."""
        
        base_configs = {
            'default': '--oem 3 --psm 8',
            'single_line': '--oem 3 --psm 7',
            'single_word': '--oem 3 --psm 8',
            'digits_only': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
        }
        
        if text_type == 'down_distance':
            base_configs.update({
                'down_distance': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789THNDRS&',
                'down_distance_simple': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789&',
            })
        elif text_type == 'score':
            base_configs.update({
                'score': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-:',
                'score_simple': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            })
        elif text_type == 'time':
            base_configs.update({
                'time': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:.',
            })
        elif text_type == 'team_names':
            base_configs.update({
                'team_names': '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            })
        
        return base_configs

    def _apply_corrections(self, text: str) -> str:
        """Apply common OCR error corrections with validation."""
        try:
            corrected = text
            
            # Apply character-level corrections
            for wrong, right in self.ocr_corrections.items():
                corrected = corrected.replace(wrong, right)
            
            # Clean up whitespace
            corrected = re.sub(r'\s+', ' ', corrected).strip()
            
            return corrected
            
        except Exception as e:
            logger.warning(f"Text correction failed: {e}")
            return text  # Return original if correction fails

    def _validate_text(self, text: str, text_type: str) -> float:
        """
        Validate extracted text against expected patterns.
        
        Returns:
            Validation score between 0.0 and 1.0
        """
        try:
            if text_type not in self.text_patterns:
                return 0.5  # Neutral score if no patterns defined
            
            patterns = self.text_patterns[text_type]
            
            # Check if text matches any pattern
            for pattern in patterns:
                if pattern.match(text.strip()):
                    return 1.0  # Perfect match
            
            # Partial scoring based on similarity
            cleaned_text = re.sub(r'[^\w\d&:-]', '', text.upper())
            
            if text_type == 'down_distance':
                # Check for down (1-4) and distance (0-99)
                if re.search(r'[1-4]', cleaned_text) and re.search(r'\d{1,2}', cleaned_text):
                    return 0.7
            elif text_type == 'score':
                # Check for two numbers
                if len(re.findall(r'\d+', cleaned_text)) >= 2:
                    return 0.7
            elif text_type == 'time':
                # Check for time format
                if re.search(r'\d{1,2}[\:\.]?\d{2}', cleaned_text):
                    return 0.7
            
            return 0.2  # Low score but not zero
            
        except Exception as e:
            logger.warning(f"Text validation failed for '{text}' as {text_type}: {e}")
            return 0.0

    def extract_text_from_region(self, frame: np.ndarray, bbox: List[int], 
                                text_type: Optional[str] = None, 
                                padding: int = 5) -> OCRResult:
        """
        Extract text from a specific region with enhanced accuracy and comprehensive error handling.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            text_type: Expected text type for validation
            padding: Padding around the region
            
        Returns:
            Best extraction result with metadata
        """
        try:
            # Validate inputs
            if frame is None or frame.size == 0:
                return OCRResult(error="Input frame is empty or None")
            
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return OCRResult(error=f"Invalid bbox format: {bbox}")
            
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Validate and clamp coordinates
            if x1 >= x2 or y1 >= y2:
                return OCRResult(error=f"Invalid bbox coordinates: {bbox}")
            
            # Apply padding and ensure valid coordinates
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Final validation
            if x1 >= x2 or y1 >= y2:
                return OCRResult(error=f"Invalid bbox after padding: [{x1}, {y1}, {x2}, {y2}]")
            
            # Extract region
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return OCRResult(error='Empty region after extraction')
            
            # Extract text using multiple engines
            results = self.extract_text_multi_engine(roi, text_type)
            
            if not results:
                return OCRResult(error='No OCR results from any engine')
            
            # Filter successful results
            successful_results = [r for r in results if r.is_successful]
            
            if not successful_results:
                # Use fallback if no successful results
                if self.fallback_enabled:
                    self.performance_stats['fallback_uses'] += 1
                    fallback_result = self._fallback_text_extraction(roi, text_type)
                    fallback_result.fallback_used = True
                    return fallback_result
                else:
                    # Return best failed result with error info
                    best_failed = max(results, key=lambda x: x.confidence)
                    best_failed.error = f"All engines failed. Best confidence: {best_failed.confidence:.3f}"
                    return best_failed
            
            # Return best successful result
            best_result = successful_results[0]
            best_result.raw_results = [r.__dict__ for r in results[:3]]  # Keep top 3 for comparison
            
            if self.debug:
                print(f"🔍 Text extraction for {text_type or 'unknown'}:")
                print(f"   • Best: '{best_result.text}' (conf: {best_result.final_confidence:.3f}, {best_result.engine})")
                for i, result in enumerate(successful_results[1:4], 1):
                    print(f"   • Alt {i}: '{result.text}' (conf: {result.final_confidence:.3f}, {result.engine})")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Critical error in text extraction from region: {e}")
            if self.debug:
                traceback.print_exc()
            return OCRResult(error=f"Critical extraction failure: {e}")

    def _fallback_text_extraction(self, roi: np.ndarray, text_type: Optional[str] = None) -> OCRResult:
        """
        Fallback text extraction using simple methods when primary engines fail.
        """
        try:
            logger.info("Using fallback text extraction")
            
            # Simple threshold + basic pattern matching
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Template matching for common patterns
            if text_type == 'down_distance':
                # Look for digit patterns
                digit_patterns = ['1ST', '2ND', '3RD', '4TH']
                for pattern in digit_patterns:
                    if self._simple_template_match(binary, pattern):
                        return OCRResult(
                            text=pattern + " & 10",  # Default assumption
                            confidence=0.3,
                            engine='fallback',
                            preprocessing='template_match',
                            fallback_used=True
                        )
            
            # Generic fallback - return empty result
            return OCRResult(
                text="",
                confidence=0.0,
                engine='fallback',
                preprocessing='simple_threshold',
                fallback_used=True,
                error="Fallback extraction found no text"
            )
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return OCRResult(
                error=f"Fallback extraction failed: {e}",
                engine='fallback',
                fallback_used=True
            )

    def _simple_template_match(self, image: np.ndarray, pattern: str) -> bool:
        """Simple template matching for fallback extraction."""
        try:
            # This is a placeholder for actual template matching
            # In a real implementation, you'd have pre-created templates
            return False
        except Exception as e:
            logger.warning(f"Template matching failed: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_extractions'] > 0:
            stats['success_rate'] = stats['successful_extractions'] / stats['total_extractions']
        else:
            stats['success_rate'] = 0.0
        
        stats['engine_status'] = {k: v.value for k, v in self.engine_status.items()}
        
        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'engine_failures': {'easyocr': 0, 'tesseract': 0},
            'fallback_uses': 0,
            'average_confidence': 0.0
        }
        logger.info("Performance statistics reset")

    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the OCR system."""
        health = {
            'status': 'healthy',
            'engines': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check engine status
        available_engines = 0
        for engine, status in self.engine_status.items():
            health['engines'][engine] = {
                'status': status.value,
                'failures': self.performance_stats['engine_failures'].get(engine, 0)
            }
            
            if status == EngineStatus.AVAILABLE:
                available_engines += 1
            elif status == EngineStatus.FAILED:
                health['issues'].append(f"{engine} engine failed")
        
        # Overall health assessment
        if available_engines == 0:
            health['status'] = 'critical'
            health['issues'].append("No OCR engines available")
            health['recommendations'].append("Install EasyOCR or Tesseract")
        elif available_engines == 1:
            health['status'] = 'warning'
            health['recommendations'].append("Install additional OCR engine for redundancy")
        
        # Performance assessment
        stats = self.get_performance_stats()
        if stats['total_extractions'] > 0:
            if stats['success_rate'] < 0.5:
                health['status'] = 'warning' if health['status'] == 'healthy' else health['status']
                health['issues'].append(f"Low success rate: {stats['success_rate']:.1%}")
            
            if stats['fallback_uses'] > stats['total_extractions'] * 0.3:
                health['issues'].append("High fallback usage - primary engines may be failing")
        
        health['performance'] = stats
        
        return health


if __name__ == "__main__":
    # Basic test
    print("🧪 Enhanced OCR System Test")
    
    try:
        ocr = EnhancedOCRSystem(debug=True)
        print(f"✅ Initialization successful")
        
        # Health check
        health = ocr.health_check()
        print(f"🏥 Health Status: {health['status']}")
        
        if health['issues']:
            print("⚠️ Issues:")
            for issue in health['issues']:
                print(f"   • {issue}")
        
        if health['recommendations']:
            print("💡 Recommendations:")
            for rec in health['recommendations']:
                print(f"   • {rec}")
                
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        traceback.print_exc() 