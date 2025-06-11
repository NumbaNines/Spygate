#!/usr/bin/env python3
"""
EasyOCR Enhanced System for SpygateAI
Focused on maximizing EasyOCR accuracy through advanced preprocessing and validation.
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import time

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

class EasyOCREnhancer:
    """Enhanced EasyOCR system with advanced preprocessing and validation."""
    
    def __init__(self, gpu_enabled: bool = True, debug: bool = False):
        self.debug = debug
        self.gpu_enabled = gpu_enabled
        
        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            print("🔧 Initializing Enhanced EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=gpu_enabled, verbose=False)
            print("✅ Enhanced EasyOCR ready")
        else:
            raise ImportError("EasyOCR is required. Install with: pip install easyocr")
        
        # Sport-specific validation patterns
        self.patterns = {
            'down_distance': [
                re.compile(r'([1-4])(?:st|nd|rd|th)?\s*[&\s]+\s*(\d{1,2})', re.IGNORECASE),
                re.compile(r'([1-4])\s*&\s*(\d+)', re.IGNORECASE),
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
        
        # OCR error corrections
        self.corrections = {
            'lst': '1st', '2na': '2nd', '3ra': '3rd', '4tn': '4th',
            'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6',
        }
        
    def create_enhanced_versions(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """Create multiple enhanced versions of the input image."""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        enhanced_versions = []
        
        # Version 1: High contrast with adaptive threshold
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=2.5, beta=30)
        adaptive_thresh = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        enhanced_versions.append((adaptive_thresh, "contrast_adaptive"))
        
        # Version 2: Otsu threshold with enhancement
        enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=15)
        _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_versions.append((otsu_thresh, "enhanced_otsu"))
        
        # Version 3: Denoised version
        denoised = cv2.fastNlMeansDenoising(gray)
        denoised_enhanced = cv2.convertScaleAbs(denoised, alpha=2.0, beta=20)
        enhanced_versions.append((denoised_enhanced, "denoised_enhanced"))
        
        # Version 4: Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        morph_enhanced = cv2.convertScaleAbs(morph_cleaned, alpha=2.2, beta=25)
        enhanced_versions.append((morph_enhanced, "morph_enhanced"))
        
        # Version 5: Inverted if predominantly dark
        mean_val = np.mean(gray)
        if mean_val < 127:  # Dark image - try inverted
            inverted = 255 - gray
            inv_enhanced = cv2.convertScaleAbs(inverted, alpha=1.5, beta=10)
            enhanced_versions.append((inv_enhanced, "inverted_enhanced"))
        
        # Scale up small images
        final_versions = []
        for img, name in enhanced_versions:
            h, w = img.shape
            if h < 30 or w < 30:
                scale_factor = max(2, 30 // min(h, w))
                scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
                final_versions.append((scaled, f"{name}_scaled"))
            else:
                final_versions.append((img, name))
        
        return final_versions
    
    def apply_corrections(self, text: str) -> str:
        """Apply sport-specific OCR corrections."""
        corrected = text
        
        # Apply corrections
        for wrong, right in self.corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Clean whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        return corrected
    
    def validate_text(self, text: str, expected_type: Optional[str]) -> float:
        """Validate text against expected patterns. Returns confidence boost 0.0-1.0."""
        if not text or not expected_type or expected_type not in self.patterns:
            return 0.0
        
        patterns = self.patterns[expected_type]
        
        # Check exact matches
        for pattern in patterns:
            if pattern.match(text.strip()):
                return 1.0  # Perfect match - boost confidence
        
        # Partial validation
        cleaned = re.sub(r'[^\w\d&:-]', '', text.upper())
        
        if expected_type == 'down_distance':
            if re.search(r'[1-4]', cleaned) and re.search(r'\d{1,2}', cleaned):
                return 0.5
        elif expected_type == 'score':
            numbers = re.findall(r'\d+', cleaned)
            if len(numbers) >= 2:
                return 0.5
        elif expected_type == 'time':
            if re.search(r'\d{1,2}[\:\.]?\d{2}', cleaned):
                return 0.5
        
        return 0.0
    
    def extract_text_enhanced(self, image: np.ndarray, expected_type: Optional[str] = None, 
                             confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Enhanced text extraction with multiple preprocessing strategies.
        
        Args:
            image: Input image region
            expected_type: Expected text type ('down_distance', 'score', 'time')
            confidence_threshold: Minimum confidence for OCR results
            
        Returns:
            Best extraction result with metadata
        """
        
        # Get enhanced versions
        enhanced_versions = self.create_enhanced_versions(image)
        
        all_results = []
        
        # Test each enhanced version
        for enhanced_img, version_name in enhanced_versions:
            try:
                # Run EasyOCR
                ocr_results = self.reader.readtext(enhanced_img, detail=1, paragraph=False)
                
                # Process results
                for bbox, text, confidence in ocr_results:
                    if confidence > confidence_threshold:
                        # Apply corrections
                        corrected_text = self.apply_corrections(text)
                        
                        # Apply validation boost if type specified
                        validation_boost = self.validate_text(corrected_text, expected_type)
                        final_confidence = confidence + (validation_boost * 0.2)  # Boost validated text
                        
                        all_results.append({
                            'text': corrected_text,
                            'confidence': min(1.0, final_confidence),  # Cap at 1.0
                            'original_confidence': confidence,
                            'validation_boost': validation_boost,
                            'preprocessing': version_name,
                            'bbox': bbox
                        })
                
            except Exception as e:
                if self.debug:
                    print(f"OCR failed on {version_name}: {e}")
                continue
        
        if not all_results:
            return {
                'text': '',
                'confidence': 0.0,
                'preprocessing': 'none',
                'error': 'no_text_detected'
            }
        
        # Sort by final confidence
        all_results = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
        
        # Get best result
        best_result = all_results[0]
        
        # Add metadata
        best_result['total_candidates'] = len(all_results)
        best_result['all_candidates'] = [r['text'] for r in all_results[:3]]
        
        if self.debug:
            print(f"🔍 Enhanced OCR Results for {expected_type or 'unknown'}:")
            print(f"   • Best: '{best_result['text']}' (conf: {best_result['confidence']:.3f})")
            for i, result in enumerate(all_results[1:3], 1):
                print(f"   • Alt {i}: '{result['text']}' (conf: {result['confidence']:.3f})")
        
        return best_result
    
    def process_hud_regions(self, frame: np.ndarray, regions: List[Dict]) -> Dict[str, Any]:
        """Process multiple HUD regions with enhanced OCR."""
        
        results = {}
        
        for region in regions:
            region_name = region['name']
            bbox = region['bbox']
            expected_type = region.get('type')
            
            # Extract ROI
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                results[region_name] = {'text': '', 'confidence': 0.0, 'error': 'empty_region'}
                continue
            
            # Extract text
            result = self.extract_text_enhanced(roi, expected_type)
            result['region_name'] = region_name
            result['bbox'] = bbox
            
            results[region_name] = result
        
        return results

def test_enhanced_easyocr():
    """Test the enhanced EasyOCR system."""
    
    print("🚀 Testing Enhanced EasyOCR System")
    print("=" * 50)
    
    # Initialize enhancer
    enhancer = EasyOCREnhancer(debug=True)
    
    # Test with synthetic images first
    print("\n🧪 Synthetic Test Cases:")
    
    test_cases = [
        {'text': '3RD & 7', 'type': 'down_distance'},
        {'text': '1ST & 10', 'type': 'down_distance'},
        {'text': '14 - 21', 'type': 'score'},
        {'text': '07:45', 'type': 'time'},
        {'text': '2ND & 15', 'type': 'down_distance'},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n🔬 Test {i+1}: '{case['text']}'")
        
        # Create test image with noise
        img = np.ones((80, 200, 3), dtype=np.uint8) * 245
        cv2.putText(img, case['text'], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
        
        # Add realistic noise
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Test extraction
        start_time = time.time()
        result = enhancer.extract_text_enhanced(img, case['type'])
        processing_time = time.time() - start_time
        
        print(f"   🎯 Result: '{result['text']}'")
        print(f"   📈 Confidence: {result['confidence']:.3f}")
        print(f"   ⏱️ Time: {processing_time:.3f}s")
        print(f"   🔧 Method: {result.get('preprocessing', 'unknown')}")
        
        # Check accuracy
        original_clean = case['text'].replace(' ', '').replace('&', '').lower()
        result_clean = result['text'].replace(' ', '').replace('&', '').lower()
        
        if original_clean in result_clean or result_clean in original_clean:
            print(f"   ✅ ACCURATE!")
        else:
            print(f"   ⚠️ Needs improvement")
    
    # Test with real images if available
    print(f"\n📸 Real Image Tests:")
    
    test_dirs = [Path("training_data/sample_images"), Path("test_images")]
    test_images = []
    
    for test_dir in test_dirs:
        if test_dir.exists():
            test_images.extend(list(test_dir.glob("*.png"))[:3])
    
    if test_images:
        for img_path in test_images[:2]:  # Test 2 real images
            print(f"\n🖼️ Testing: {img_path.name}")
            
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            # Define HUD regions
            regions = [
                {
                    'name': 'down_distance',
                    'bbox': [50, 20, 300, 100],
                    'type': 'down_distance'
                },
                {
                    'name': 'score',
                    'bbox': [w-250, 20, w-50, 100],
                    'type': 'score'
                }
            ]
            
            # Process regions
            results = enhancer.process_hud_regions(frame, regions)
            
            for region_name, result in results.items():
                print(f"   • {region_name}: '{result['text']}' (conf: {result['confidence']:.3f})")
    
    else:
        print("   ⚠️ No real test images found")
    
    print(f"\n✅ Enhanced EasyOCR testing complete!")

def compare_basic_vs_enhanced_easyocr():
    """Compare basic EasyOCR vs enhanced version."""
    
    print("\n🔄 Basic vs Enhanced EasyOCR Comparison")
    print("=" * 50)
    
    # Initialize both
    basic_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    enhancer = EasyOCREnhancer(debug=False)
    
    # Create challenging test cases
    test_cases = [
        "3RD & 7",
        "1ST & 10", 
        "14 - 21",
        "07:45"
    ]
    
    for text in test_cases:
        print(f"\n📝 Testing: '{text}'")
        
        # Create challenging image
        img = np.ones((60, 180, 3), dtype=np.uint8) * 220
        cv2.putText(img, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        
        # Add blur and noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Test basic EasyOCR
        basic_results = basic_reader.readtext(img, detail=1)
        basic_text = ""
        basic_conf = 0.0
        if basic_results:
            basic_text = " ".join([r[1] for r in basic_results if r[2] > 0.3])
            basic_conf = max([r[2] for r in basic_results if r[2] > 0.3], default=0.0)
        
        # Test enhanced EasyOCR
        enhanced_result = enhancer.extract_text_enhanced(img, 'down_distance')
        enhanced_text = enhanced_result['text']
        enhanced_conf = enhanced_result['confidence']
        
        print(f"   🟡 Basic: '{basic_text}' (conf: {basic_conf:.3f})")
        print(f"   🟢 Enhanced: '{enhanced_text}' (conf: {enhanced_conf:.3f})")
        
        # Determine winner
        original_clean = text.replace(' ', '').replace('&', '').lower()
        basic_clean = basic_text.replace(' ', '').replace('&', '').lower()
        enhanced_clean = enhanced_text.replace(' ', '').replace('&', '').lower()
        
        basic_match = original_clean in basic_clean or basic_clean in original_clean
        enhanced_match = original_clean in enhanced_clean or enhanced_clean in original_clean
        
        if enhanced_match and not basic_match:
            print(f"   🏆 Enhanced WINS!")
        elif basic_match and not enhanced_match:
            print(f"   🏆 Basic WINS")
        elif enhanced_match and basic_match:
            if enhanced_conf > basic_conf:
                print(f"   🏆 Enhanced WINS (higher confidence)")
            else:
                print(f"   🤝 TIE (both accurate)")
        else:
            print(f"   😐 Both need improvement")

if __name__ == "__main__":
    if not EASYOCR_AVAILABLE:
        print("❌ EasyOCR not available. Install with: pip install easyocr")
        exit(1)
    
    test_enhanced_easyocr()
    compare_basic_vs_enhanced_easyocr() 