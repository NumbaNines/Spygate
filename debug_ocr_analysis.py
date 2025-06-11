#!/usr/bin/env python3
"""
Debug OCR Analysis for SpygateAI
Helps us understand what text OCR is detecting in HUD regions
"""

import cv2
import numpy as np
import easyocr
from pathlib import Path
from hybrid_triangle_detector_final import FinalHybridTriangleDetector

def debug_ocr_on_image(image_path: str):
    """Debug OCR extraction on a specific image."""
    
    print(f"🔍 Debugging OCR on: {image_path}")
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not load {image_path}")
        return
    
    # Initialize detectors
    triangle_detector = FinalHybridTriangleDetector()
    ocr_reader = easyocr.Reader(['en'], gpu=True)
    
    # Get triangle detection results
    triangle_results = triangle_detector.process_frame(frame)
    
    # Find HUD regions
    hud_regions = [r for r in triangle_results['regions'] if r['class_name'] == 'hud']
    
    print(f"📊 Found {len(hud_regions)} HUD regions")
    
    if not hud_regions:
        print("❌ No HUD regions detected")
        return
    
    # Process each HUD region
    for i, hud_region in enumerate(hud_regions):
        print(f"\n🎯 HUD Region {i+1}:")
        print(f"   Confidence: {hud_region['confidence']:.3f}")
        print(f"   Bbox: {hud_region['bbox']}")
        
        x1, y1, x2, y2 = hud_region['bbox']
        
        # Extract HUD region with various padding sizes
        for padding in [10, 20, 50]:
            print(f"\n   📋 Testing with {padding}px padding:")
            
            # Extract region
            hud_crop = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                            max(0, x1-padding):min(frame.shape[1], x2+padding)]
            
            if hud_crop.size == 0:
                print(f"      ❌ Invalid crop with {padding}px padding")
                continue
            
            # Save the cropped region for inspection
            crop_filename = f"debug_hud_crop_{i+1}_pad{padding}.png"
            cv2.imwrite(crop_filename, hud_crop)
            print(f"      💾 Saved crop: {crop_filename}")
            
            # Test different preprocessing approaches
            preprocessed_images = {
                'original': hud_crop,
                'grayscale': cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY),
                'contrast_enhanced': cv2.convertScaleAbs(cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY), alpha=2.0, beta=0),
                'threshold': cv2.threshold(cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],
                'adaptive_threshold': cv2.adaptiveThreshold(cv2.cvtColor(hud_crop, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            }
            
            for preprocess_name, processed_img in preprocessed_images.items():
                try:
                    # Save preprocessed image
                    if len(processed_img.shape) == 2:  # Grayscale
                        debug_filename = f"debug_{preprocess_name}_pad{padding}.png"
                        cv2.imwrite(debug_filename, processed_img)
                    
                    # Run OCR
                    ocr_results = ocr_reader.readtext(processed_img)
                    
                    if ocr_results:
                        print(f"      ✅ {preprocess_name.upper()} OCR Results:")
                        for result in ocr_results:
                            bbox, text, confidence = result
                            if confidence > 0.2:  # Lower threshold for debugging
                                print(f"         📝 '{text}' (conf: {confidence:.3f})")
                        
                        # Combine all text
                        all_text = " ".join([result[1] for result in ocr_results if result[2] > 0.2])
                        print(f"      🔗 Combined text: '{all_text}'")
                    else:
                        print(f"      ❌ {preprocess_name}: No text detected")
                        
                except Exception as e:
                    print(f"      ⚠️ {preprocess_name} error: {e}")
    
    # Also try OCR on the entire frame
    print(f"\n🖼️ Full Frame OCR Analysis:")
    try:
        full_ocr_results = ocr_reader.readtext(frame)
        if full_ocr_results:
            print(f"   ✅ Found {len(full_ocr_results)} text elements in full frame:")
            for result in full_ocr_results:
                bbox, text, confidence = result
                if confidence > 0.3:
                    print(f"      📝 '{text}' (conf: {confidence:.3f}) at {bbox}")
        else:
            print("   ❌ No text detected in full frame")
    except Exception as e:
        print(f"   ⚠️ Full frame OCR error: {e}")

def debug_multiple_images():
    """Debug OCR on multiple test images."""
    
    test_images = [
        "fresh_test_2_monitor3_screenshot_20250611_031353_61.png",
        "fresh_test_6_monitor3_screenshot_20250611_032339_26.png", 
        "fresh_test_10_monitor3_screenshot_20250611_031328_56.png"
    ]
    
    print("🔬 OCR Debug Analysis")
    print("="*60)
    
    for img_path in test_images:
        if Path(img_path).exists():
            debug_ocr_on_image(img_path)
            print("\n" + "="*60)
        else:
            print(f"❌ Image not found: {img_path}")

if __name__ == "__main__":
    debug_multiple_images() 