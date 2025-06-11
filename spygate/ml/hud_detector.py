"""YOLOv8-based HUD element detection for gameplay clips with enhanced OCR processing."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback
import time

import cv2
import numpy as np

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

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from .yolov8_model import UI_CLASSES, EnhancedYOLOv8, OptimizationConfig

# Import our enhanced OCR system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from enhanced_ocr_system import EnhancedOCRSystem, OCRResult
    ENHANCED_OCR_AVAILABLE = True
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
    print("⚠️ Enhanced OCR system not available - using fallback methods")

# Configure logging
logger = logging.getLogger(__name__)

class HUDDetectionError(Exception):
    """Base exception for HUD detection errors."""
    pass

class ModelInitializationError(HUDDetectionError):
    """Raised when model initialization fails."""
    pass

class OCRProcessingError(HUDDetectionError):
    """Raised when OCR processing fails."""
    pass

class ValidationError(HUDDetectionError):
    """Raised when result validation fails."""
    pass


class EnhancedHUDDetector:
    """YOLOv8-based HUD element detector with advanced OCR processing and hardware optimization.
    
    Uses the new 5-class fresh model system:
    - hud (Main HUD bar)
    - possession_triangle_area (Left triangle region)
    - territory_triangle_area (Right triangle region)  
    - preplay_indicator (Pre-play state indicator)
    - play_call_screen (Play call screen overlay)
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the enhanced HUD detector.

        Args:
            model_path: Path to a custom YOLOv8 model. If None, will use the fresh model.
        """
        self.initialized = False
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'ocr_failures': 0,
            'model_failures': 0,
            'average_detection_time': 0.0,
            'average_ocr_time': 0.0
        }
        
        # Error handling
        self.max_retries = 3
        self.retry_count = 0
        
        # Model configuration with fresh 5-class system
        self.classes = UI_CLASSES  # Fresh model classes
        self.model_path = model_path
        
        # Initialize components with error handling
        try:
            self._initialize_model()
            self._initialize_ocr()
            self.initialized = True
            logger.info("Enhanced HUD detector initialized successfully")
        except Exception as e:
            logger.error(f"HUD detector initialization failed: {e}")
            raise ModelInitializationError(f"Failed to initialize HUD detector: {e}")

    def _initialize_model(self):
        """Initialize the YOLOv8 model with error handling."""
        try:
            if self.model_path:
                if not Path(self.model_path).exists():
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")
                logger.info(f"Loading custom model: {self.model_path}")
            else:
                # Use fresh model as default
                fresh_model_path = Path("best_fresh.pt")
                if fresh_model_path.exists():
                    self.model_path = str(fresh_model_path)
                    logger.info("Using trained fresh model: best_fresh.pt")
                else:
                    logger.warning("Fresh model not found, using default configuration")

            # Get hardware-optimized configuration
            config = self.optimizer.get_model_config("yolov8")
            
            self.model = EnhancedYOLOv8(
                model_path=self.model_path,
                classes=self.classes,
                config=config
            )
            
            # Validate model
            if not self.model.initialize():
                raise ModelInitializationError("YOLOv8 model initialization failed")
                
            logger.info(f"Fresh 5-class model loaded successfully: {self.classes}")
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise ModelInitializationError(f"Failed to initialize YOLOv8 model: {e}")

    def _initialize_ocr(self):
        """Initialize OCR system with enhanced capabilities."""
        try:
            if ENHANCED_OCR_AVAILABLE:
                self.ocr_system = EnhancedOCRSystem(
                    gpu_enabled=self.hardware.has_gpu(),
                    debug=False,
                    max_retries=2,
                    fallback_enabled=True
                )
                logger.info("Enhanced OCR system initialized")
            else:
                # Fallback to basic OCR
                self.ocr_system = None
                self._initialize_basic_ocr()
                logger.warning("Using basic OCR fallback")
                
        except Exception as e:
            logger.error(f"OCR initialization error: {e}")
            # Continue with basic OCR as fallback
            self.ocr_system = None
            self._initialize_basic_ocr()

    def _initialize_basic_ocr(self):
        """Initialize basic OCR as fallback."""
        try:
            if EASYOCR_AVAILABLE:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=self.hardware.has_gpu(), verbose=False)
                logger.info("Basic EasyOCR initialized as fallback")
            else:
                self.easyocr_reader = None
                logger.warning("No OCR engines available")
        except Exception as e:
            logger.error(f"Basic OCR initialization failed: {e}")
            self.easyocr_reader = None

    def detect_hud_elements(self, frame: np.ndarray) -> dict[str, Any]:
        """Detect HUD elements in a frame using the fresh 5-class YOLOv8 model.

        Args:
            frame: Input frame as numpy array

        Returns:
            Dict containing:
            - detections: List of detected elements with their locations and text
            - metadata: Detection info and confidence scores
        """
        if not self.initialized:
            raise RuntimeError("Enhanced HUD detector not initialized")

        start_time = time.time()
        self.detection_stats['total_detections'] += 1

        try:
            # Validate input
            if frame is None or frame.size == 0:
                raise ValueError("Input frame is empty or None")

            # Use the optimized YOLOv8 model for detection with fresh 5-class system
            detection_results = self.model.detect_hud_elements(frame)

            # Extract detection data using fresh model classes
            detections = []
            game_state = {}
            total_confidence = 0
            detection_count = 0

            for detection in detection_results.get('detections', []):
                try:
                    element_type = detection.get('class')
                    bbox = detection.get('bbox', [])
                    confidence = detection.get('confidence', 0.0)

                    if not element_type or not bbox or len(bbox) != 4:
                        continue

                    # Process different element types with enhanced OCR
                    text = ""
                    ocr_confidence = 0.0
                    
                    if element_type in ['hud', 'possession_triangle_area', 'territory_triangle_area']:
                        text, ocr_confidence = self._extract_text_with_error_handling(
                            frame, bbox, element_type
                        )

                    detection_item = {
                        'type': element_type,
                        'bbox': bbox,
                        'confidence': confidence,
                        'text': text,
                        'ocr_confidence': ocr_confidence
                    }
                    detections.append(detection_item)

                    # Update game state based on fresh model classes
                    if element_type == "hud" and text:
                        game_state.update(self._parse_hud_text(text))
                    elif element_type == "possession_triangle_area" and text:
                        possession = self._parse_possession_triangle(text)
                        if possession:
                            game_state["possession_indicator"] = possession
                    elif element_type == "territory_triangle_area" and text:
                        territory = self._parse_territory_triangle(text)
                        if territory:
                            game_state["territory_indicator"] = territory
                    elif element_type == "preplay_indicator":
                        game_state["preplay_active"] = True
                    elif element_type == "play_call_screen":
                        game_state["play_call_active"] = True

                    total_confidence += confidence
                    detection_count += 1

                except Exception as e:
                    logger.warning(f"Error processing detection {element_type}: {e}")
                    continue

            # Calculate overall confidence
            if detection_count > 0:
                game_state["confidence"] = total_confidence / detection_count

            detection_time = time.time() - start_time
            self._update_performance_stats(detection_time, True)

            result = {
                'detections': detections,
                'game_state': game_state,
                'metadata': {
                    'detection_count': detection_count,
                    'processing_time': detection_time,
                    'model_classes': self.classes,
                    'fresh_model': True
                }
            }

            self.detection_stats['successful_detections'] += 1
            return result

        except Exception as e:
            logger.error(f"HUD detection failed: {e}")
            self.detection_stats['model_failures'] += 1
            
            # Return error result
            return {
                'detections': [],
                'game_state': {},
                'metadata': {
                    'error': str(e),
                    'processing_time': time.time() - start_time,
                    'detection_count': 0
                }
            }

    def _extract_text_with_error_handling(self, frame: np.ndarray, bbox: List[int], 
                                         element_type: str) -> Tuple[str, float]:
        """Extract text from detected region with comprehensive error handling."""
        try:
            ocr_start_time = time.time()
            
            # Determine text type for validation
            text_type_mapping = {
                'hud': None,  # General HUD text
                'possession_triangle_area': 'team_names',
                'territory_triangle_area': 'team_names'
            }
            text_type = text_type_mapping.get(element_type)

            # Use enhanced OCR if available
            if self.ocr_system:
                try:
                    result = self.ocr_system.extract_text_from_region(
                        frame, bbox, text_type, padding=5
                    )
                    
                    ocr_time = time.time() - ocr_start_time
                    self._update_ocr_stats(ocr_time, result.is_successful)
                    
                    if result.is_successful:
                        return result.text, result.final_confidence
                    else:
                        logger.warning(f"Enhanced OCR failed for {element_type}: {result.error}")
                        # Fall through to basic OCR
                        
                except Exception as e:
                    logger.warning(f"Enhanced OCR error for {element_type}: {e}")
                    # Fall through to basic OCR

            # Fallback to basic OCR
            return self._basic_text_extraction(frame, bbox, element_type)

        except Exception as e:
            logger.error(f"Text extraction completely failed for {element_type}: {e}")
            self.detection_stats['ocr_failures'] += 1
            return "", 0.0

    def _basic_text_extraction(self, frame: np.ndarray, bbox: List[int], 
                              element_type: str) -> Tuple[str, float]:
        """Basic text extraction fallback."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Apply padding and clamp coordinates
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return "", 0.0
            
            # Basic preprocessing
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Try EasyOCR if available
            if self.easyocr_reader:
                try:
                    results = self.easyocr_reader.readtext(gray, detail=1)
                    if results:
                        # Combine all text
                        texts = [text for _, text, conf in results if conf > 0.1]
                        if texts:
                            combined_text = " ".join(texts).strip()
                            avg_conf = np.mean([conf for _, _, conf in results if conf > 0.1])
                            return combined_text, avg_conf
                except Exception as e:
                    logger.warning(f"Basic EasyOCR failed: {e}")
            
            # Last resort - return empty
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Basic text extraction failed: {e}")
            return "", 0.0

    def _parse_hud_text(self, text: str) -> Dict[str, Any]:
        """Parse general HUD text for game state information."""
        try:
            import re
            game_info = {}
            
            # Try to extract score (pattern: XX-YY or XX YY)
            score_pattern = r"(\d{1,2})\s*[-:]\s*(\d{1,2})"
            score_match = re.search(score_pattern, text)
            if score_match:
                game_info["score"] = {
                    "home": int(score_match.group(1)),
                    "away": int(score_match.group(2))
                }
            
            # Try to extract down and distance
            down_dist_pattern = r"([1-4])(?:st|nd|rd|th)?\s*[&\s]+\s*(\d{1,2})"
            dd_match = re.search(down_dist_pattern, text, re.IGNORECASE)
            if dd_match:
                game_info["down"] = int(dd_match.group(1))
                game_info["distance"] = int(dd_match.group(2))
            
            # Try to extract time
            time_pattern = r"(\d{1,2}):(\d{2})"
            time_match = re.search(time_pattern, text)
            if time_match:
                game_info["time"] = f"{time_match.group(1)}:{time_match.group(2)}"
            
            return game_info
            
        except Exception as e:
            logger.warning(f"HUD text parsing failed: {e}")
            return {}

    def _parse_possession_triangle(self, text: str) -> Optional[str]:
        """Parse possession triangle indicator (left side of HUD)."""
        try:
            import re
            
            # Look for team indicators or directional arrows
            # The possession triangle points to the team that HAS the ball
            
            # Common patterns: "HOME", "AWAY", team abbreviations
            team_pattern = r"(HOME|AWAY|[A-Z]{2,4})"
            match = re.search(team_pattern, text.upper())
            
            if match:
                team = match.group(1)
                return "→" if team in ["HOME", "LEFT"] else "←"
            
            # Look for arrow characters
            if "→" in text or ">" in text or "►" in text:
                return "→"
            elif "←" in text or "<" in text or "◄" in text:
                return "←"
            
            return None
            
        except Exception as e:
            logger.warning(f"Possession triangle parsing failed: {e}")
            return None

    def _parse_territory_triangle(self, text: str) -> Optional[str]:
        """Parse territory triangle indicator (right side of HUD)."""
        try:
            import re
            
            # Territory indicator shows field position context
            # ▲ = in opponent's territory (good field position)
            # ▼ = in own territory (poor field position)
            
            # Look for explicit territory indicators
            if "▲" in text or "UP" in text.upper():
                return "▲"
            elif "▼" in text or "DOWN" in text.upper():
                return "▼"
            
            # Fallback to text patterns: "OWN", "OPP"
            pattern = r"(OWN|OPP)"
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                territory = match.group(1).upper()
                return "▲" if territory == "OPP" else "▼"

            return None
            
        except Exception as e:
            logger.warning(f"Territory triangle parsing failed: {e}")
            return None

    def _update_performance_stats(self, detection_time: float, success: bool):
        """Update performance tracking statistics."""
        try:
            # Update average detection time
            total_time = (self.detection_stats['average_detection_time'] * 
                         (self.detection_stats['total_detections'] - 1) + detection_time)
            self.detection_stats['average_detection_time'] = (
                total_time / self.detection_stats['total_detections']
            )
            
        except Exception as e:
            logger.warning(f"Performance stats update failed: {e}")

    def _update_ocr_stats(self, ocr_time: float, success: bool):
        """Update OCR performance statistics."""
        try:
            if not hasattr(self, 'ocr_stats'):
                self.ocr_stats = {'total_ocr': 0, 'average_ocr_time': 0.0}
            
            self.ocr_stats['total_ocr'] += 1
            total_time = (self.ocr_stats['average_ocr_time'] * 
                         (self.ocr_stats['total_ocr'] - 1) + ocr_time)
            self.ocr_stats['average_ocr_time'] = total_time / self.ocr_stats['total_ocr']
            
        except Exception as e:
            logger.warning(f"OCR stats update failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.detection_stats.copy()
        
        if stats['total_detections'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_detections']
            stats['ocr_failure_rate'] = stats['ocr_failures'] / stats['total_detections']
            stats['model_failure_rate'] = stats['model_failures'] / stats['total_detections']
        else:
            stats['success_rate'] = 0.0
            stats['ocr_failure_rate'] = 0.0
            stats['model_failure_rate'] = 0.0
        
        # Add OCR stats if available
        if hasattr(self, 'ocr_stats'):
            stats.update(self.ocr_stats)
        
        # Add system health
        stats['system_health'] = self.health_check()
        
        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            'status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check model status
            health['components']['model'] = {
                'status': 'available' if self.model and self.initialized else 'failed',
                'path': self.model_path,
                'classes': len(self.classes)
            }
            
            # Check OCR status
            if self.ocr_system:
                ocr_health = self.ocr_system.health_check()
                health['components']['ocr'] = ocr_health
                if ocr_health['status'] != 'healthy':
                    health['issues'].extend(ocr_health['issues'])
            else:
                health['components']['ocr'] = {
                    'status': 'basic_fallback',
                    'engines': {'easyocr': self.easyocr_reader is not None}
                }
                health['issues'].append("Using basic OCR fallback")
            
            # Check performance
            stats = self.get_performance_stats()
            if stats['total_detections'] > 0:
                if stats['success_rate'] < 0.7:
                    health['status'] = 'warning'
                    health['issues'].append(f"Low detection success rate: {stats['success_rate']:.1%}")
                
                if stats['ocr_failure_rate'] > 0.3:
                    health['issues'].append(f"High OCR failure rate: {stats['ocr_failure_rate']:.1%}")
            
            # Overall status
            if health['issues']:
                if health['status'] == 'healthy':
                    health['status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def reset_stats(self):
        """Reset performance statistics."""
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'ocr_failures': 0,
            'model_failures': 0,
            'average_detection_time': 0.0,
            'average_ocr_time': 0.0
        }
        
        if hasattr(self, 'ocr_stats'):
            del self.ocr_stats
        
        if self.ocr_system:
            self.ocr_system.reset_performance_stats()
        
        logger.info("HUD detector statistics reset")


# Backward compatibility alias
HUDDetector = EnhancedHUDDetector
