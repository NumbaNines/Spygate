#!/usr/bin/env python3
"""
OCR Accuracy Enhancer for SpygateAI
Practical enhancements to existing OCR systems without major refactoring.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRAccuracyEnhancer:
    """
    Practical enhancements for existing OCR systems.
    Can be integrated into current workflow without major changes.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Sport-specific patterns for validation
        self.patterns = {
            'down_distance': [
                re.compile(r'([1-4])(?:st|nd|rd|th)?\s*[&\s]+\s*(\d{1,2})', re.IGNORECASE),
                re.compile(r'([1-4])[snrd]+\s*[&\s]\s*(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)\s*[&]\s*(\d+)', re.IGNORECASE),
            ],
            'score': [
                re.compile(r'(\d{1,2})\s*[-:]\s*(\d{1,2})'),
                re.compile(r'(\d{1,2})\s+(\d{1,2})'),
            ],
            'time': [
                re.compile(r'(\d{1,2}):(\d{2})'),
                re.compile(r'(\d{1,2})\.(\d{2})'),
            ]
        }
        
        # Common OCR error corrections
        self.corrections = {
            'lst': '1st', '2na': '2nd', '3ra': '3rd', '4tn': '4th',
            'O': '0', 'I': '1', 'S': '5', 'B': '8',
            'a': '&', 'e': '&', 'G': '6', 'g': '9'
        }
    
    def enhance_roi_preprocessing(self, roi: np.ndarray, method: str = "adaptive") -> List[np.ndarray]:
        """
        Enhanced preprocessing with multiple strategies.
        
        Args:
            roi: Region of interest
            method: "single" for one method, "multi" for multiple versions
            
        Returns:
            List of enhanced images
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        enhanced_versions = []
        
        if method == "multi":
            # Strategy 1: Original with contrast enhancement
            enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=20)
            enhanced_versions.append(enhanced)
            
            # Strategy 2: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            enhanced_versions.append(adaptive)
            
            # Strategy 3: Otsu's threshold
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_versions.append(otsu)
            
            # Strategy 4: Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            _, morph_thresh = cv2.threshold(morph, 127, 255, cv2.THRESH_BINARY)
            enhanced_versions.append(morph_thresh)
            
            # Strategy 5: Denoising + threshold
            denoised = cv2.fastNlMeansDenoising(gray)
            _, denoised_thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
            enhanced_versions.append(denoised_thresh)
            
        else:
            # Single best strategy - adaptive threshold with enhancement
            enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)
            adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            enhanced_versions.append(adaptive)
        
        # Scale up small regions
        final_versions = []
        for img in enhanced_versions:
            h, w = img.shape
            if h < 25 or w < 25:
                scale_factor = max(2, 25 // min(h, w))
                scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
                final_versions.append(scaled)
            else:
                final_versions.append(img)
        
        return final_versions
    
    def apply_text_corrections(self, text: str) -> str:
        """Apply common OCR error corrections."""
        corrected = text
        
        # Apply corrections
        for wrong, right in self.corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Clean whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    def validate_extracted_text(self, text: str, expected_type: str) -> float:
        """
        Validate text against expected patterns.
        
        Returns:
            Confidence score 0.0-1.0
        """
        if not text or expected_type not in self.patterns:
            return 0.0
        
        patterns = self.patterns[expected_type]
        
        # Check exact matches
        for pattern in patterns:
            if pattern.match(text.strip()):
                return 1.0
        
        # Partial validation
        cleaned = re.sub(r'[^\w\d&:-]', '', text.upper())
        
        if expected_type == 'down_distance':
            if re.search(r'[1-4]', cleaned) and re.search(r'\d{1,2}', cleaned):
                return 0.7
        elif expected_type == 'score':
            numbers = re.findall(r'\d+', cleaned)
            if len(numbers) >= 2:
                return 0.7
        elif expected_type == 'time':
            if re.search(r'\d{1,2}[\:\.]?\d{2}', cleaned):
                return 0.7
        
        return 0.2
    
    def enhanced_easyocr_extraction(self, roi: np.ndarray, reader, confidence_threshold: float = 0.3) -> List[Dict]:
        """Enhanced EasyOCR extraction with multiple preprocessing."""
        results = []
        
        # Get multiple preprocessed versions
        enhanced_rois = self.enhance_roi_preprocessing(roi, "multi")
        
        for i, enhanced_roi in enumerate(enhanced_rois):
            try:
                ocr_results = reader.readtext(enhanced_roi, detail=1, paragraph=False)
                
                for bbox, text, confidence in ocr_results:
                    if confidence > confidence_threshold:
                        corrected_text = self.apply_text_corrections(text)
                        
                        results.append({
                            'text': corrected_text,
                            'confidence': confidence,
                            'preprocessing': f"method_{i}",
                            'engine': 'EasyOCR',
                            'bbox': bbox
                        })
            except Exception as e:
                if self.debug:
                    print(f"EasyOCR preprocessing {i} failed: {e}")
                continue
        
        return results
    
    def enhanced_tesseract_extraction(self, roi: np.ndarray, text_type: Optional[str] = None) -> List[Dict]:
        """Enhanced Tesseract extraction with optimized configs."""
        results = []
        
        # Get configs based on text type
        configs = self._get_tesseract_configs(text_type)
        enhanced_rois = self.enhance_roi_preprocessing(roi, "multi")
        
        for preprocessing_idx, enhanced_roi in enumerate(enhanced_rois):
            for config_name, config in configs.items():
                try:
                    text = pytesseract.image_to_string(enhanced_roi, config=config).strip()
                    
                    if not text:
                        continue
                    
                    # Get confidence
                    data = pytesseract.image_to_data(enhanced_roi, config=config, 
                                                   output_type=pytesseract.Output.DICT)
                    confidences = [float(conf) for conf in data['conf'] if conf != '-1' and float(conf) > 0]
                    
                    if confidences:
                        avg_confidence = np.mean(confidences) / 100.0  # Normalize
                        corrected_text = self.apply_text_corrections(text)
                        
                        results.append({
                            'text': corrected_text,
                            'confidence': avg_confidence,
                            'preprocessing': f"method_{preprocessing_idx}",
                            'config': config_name,
                            'engine': 'Tesseract'
                        })
                
                except Exception as e:
                    if self.debug:
                        print(f"Tesseract config {config_name} failed: {e}")
                    continue
        
        return results
    
    def _get_tesseract_configs(self, text_type: Optional[str]) -> Dict[str, str]:
        """Get optimized Tesseract configs."""
        base_configs = {
            'default': '--oem 3 --psm 8',
            'single_line': '--oem 3 --psm 7',
        }
        
        if text_type == 'down_distance':
            base_configs.update({
                'down_distance': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789THNDRS&',
                'simple_nums': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789&',
            })
        elif text_type == 'score':
            base_configs.update({
                'score': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789-:',
                'digits': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
            })
        elif text_type == 'time':
            base_configs.update({
                'time': '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:.',
            })
        
        return base_configs
    
    def extract_text_enhanced(self, roi: np.ndarray, text_type: Optional[str] = None, 
                             easyocr_reader=None) -> Dict[str, Any]:
        """
        Enhanced text extraction combining multiple engines and strategies.
        
        Args:
            roi: Region of interest
            text_type: Expected text type for validation
            easyocr_reader: Initialized EasyOCR reader (optional)
            
        Returns:
            Best extraction result
        """
        all_results = []
        
        # Try EasyOCR if available
        if easyocr_reader and EASYOCR_AVAILABLE:
            easyocr_results = self.enhanced_easyocr_extraction(roi, easyocr_reader)
            all_results.extend(easyocr_results)
        
        # Try Tesseract if available
        if TESSERACT_AVAILABLE:
            tesseract_results = self.enhanced_tesseract_extraction(roi, text_type)
            all_results.extend(tesseract_results)
        
        if not all_results:
            return {'text': '', 'confidence': 0.0, 'engine': 'none'}
        
        # Sort by confidence
        all_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
        
        # Apply validation boost if text type specified
        if text_type:
            for result in all_results:
                validation_score = self.validate_extracted_text(result['text'], text_type)
                # Boost confidence for validated text
                result['confidence'] = (result['confidence'] * 0.7) + (validation_score * 0.3)
            
            # Re-sort after validation
            all_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
        
        # Return best result
        best_result = all_results[0]
        best_result['all_candidates'] = [r['text'] for r in all_results[:3]]
        
        if self.debug:
            print(f"🔍 Enhanced OCR for {text_type or 'unknown'}:")
            print(f"   • Best: '{best_result['text']}' (conf: {best_result['confidence']:.3f})")
            for i, result in enumerate(all_results[1:3], 1):
                print(f"   • Alt {i}: '{result['text']}' (conf: {result['confidence']:.3f})")
        
        return best_result


def upgrade_existing_ocr_processor():
    """
    Example of how to upgrade existing OCR processor with enhancements.
    This shows integration without major refactoring.
    """
    
    print("🚀 Upgrading OCR with Accuracy Enhancements...")
    
    # Initialize enhancer
    enhancer = OCRAccuracyEnhancer(debug=True)
    
    # Initialize EasyOCR if available
    easyocr_reader = None
    if EASYOCR_AVAILABLE:
        try:
            easyocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print("✅ EasyOCR initialized")
        except:
            print("❌ EasyOCR failed to initialize")
    
    # Example integration with existing code
    def enhanced_extract_text(frame, bbox, text_type=None):
        """Enhanced version of existing extract_text function."""
        
        # Extract ROI (same as before)
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "", 0.0
        
        # Use enhanced extraction
        result = enhancer.extract_text_enhanced(roi, text_type, easyocr_reader)
        
        return result['text'], result['confidence']
    
    # Test on sample data if available
    test_images_dir = Path("training_data/sample_images")
    if test_images_dir.exists():
        sample_files = list(test_images_dir.glob("*.png"))[:3]
        
        print(f"\n🧪 Testing enhanced OCR on {len(sample_files)} samples...")
        
        for img_path in sample_files:
            print(f"\n📸 Testing: {img_path.name}")
            
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Test different HUD regions
            h, w = frame.shape[:2]
            test_regions = [
                {'bbox': [50, 20, 250, 80], 'type': 'down_distance', 'name': 'Down & Distance'},
                {'bbox': [w-200, 20, w-50, 80], 'type': 'score', 'name': 'Score'},
                {'bbox': [w//2-100, 20, w//2+100, 80], 'type': 'time', 'name': 'Time'},
            ]
            
            for region in test_regions:
                text, confidence = enhanced_extract_text(
                    frame, region['bbox'], region['type']
                )
                print(f"   • {region['name']}: '{text}' (confidence: {confidence:.3f})")
    
    else:
        print("⚠️ No test images found. Testing with synthetic data...")
        
        # Create synthetic test image
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "3RD & 7", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        text, confidence = enhanced_extract_text(test_img, [30, 20, 200, 80], 'down_distance')
        print(f"Synthetic test: '{text}' (confidence: {confidence:.3f})")
    
    print("\n✅ OCR enhancement demonstration complete!")
    return enhancer, enhanced_extract_text


if __name__ == "__main__":
    upgrade_existing_ocr_processor() 