"""
Side-by-side comparison test for OCR improvements.
Shows original vs enhanced OCR results for yard line detection.
"""

import cv2
import numpy as np
from pathlib import Path
from src.spygate.ml.enhanced_ocr import EnhancedOCR
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_rgb(img):
    """Convert image to RGB if it's grayscale."""
    if len(img.shape) == 2:  # Grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def create_side_by_side_comparison(original_img, processed_img, original_text, new_text, filename):
    """Create a side by side comparison image with detection results."""
    # Ensure both images are RGB
    original_img = ensure_rgb(original_img)
    processed_img = ensure_rgb(processed_img)
    
    # Create a white background for text
    text_height = 60
    h, w = original_img.shape[:2]
    text_bg = np.ones((text_height, w * 2 + 10, 3), dtype=np.uint8) * 255
    
    # Combine images horizontally with a gap
    combined = np.hstack([original_img, np.ones((h, 10, 3), dtype=np.uint8) * 255, processed_img])
    
    # Add text background
    final_img = np.vstack([combined, text_bg])
    
    # Add detection results as text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_img, f"Original: {original_text}", (10, h + 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(final_img, f"Enhanced: {new_text}", (w + 20, h + 30), font, 0.7, (0, 0, 0), 2)
    
    # Save the comparison
    output_dir = "ocr_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"comparison_{filename}"), final_img)
    return final_img

def run_comparison_test():
    """Run OCR comparison test on test images."""
    # Initialize OCR
    ocr = EnhancedOCR()
    
    # Get test images from comprehensive test folder
    test_dir = Path("comprehensive_spygate_results")
    
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} test images")
    
    results = []
    for img_path in image_files:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
            
        # Process with enhanced OCR
        processed_img = ocr.preprocess_image(img.copy())
        ocr_results = ocr.process_region(processed_img)
        
        # Extract yard line info
        yard_line = ocr_results.get('yard_line', 'N/A')
        territory = ocr_results.get('territory', '')
        confidence = ocr_results.get('confidence', 0) if isinstance(ocr_results.get('confidence'), (int, float)) else 0
        
        # Format yard line text with territory context
        yard_line_text = f"{territory} {yard_line}" if territory and yard_line != 'N/A' else str(yard_line)
        
        # Create comparison
        comparison = create_side_by_side_comparison(
            img, 
            processed_img,
            "No yard line detected" if yard_line == 'N/A' else f"Yard line: {yard_line_text}",
            "No yard line detected" if yard_line == 'N/A' else f"Yard line: {yard_line_text} ({confidence:.2f})",
            img_path.name
        )
        
        results.append({
            'filename': img_path.name,
            'yard_line': yard_line_text,
            'confidence': confidence
        })
        
        logger.info(f"Processed {img_path.name}: {yard_line_text} ({confidence:.2f})")
        
    # Print summary
    logger.info("\nResults Summary:")
    for result in results:
        logger.info(f"{result['filename']}: Yard line = {result['yard_line']}, Confidence = {result['confidence']:.2f}")
    
    logger.info(f"\nComparison images saved in ocr_comparison_results/")

if __name__ == "__main__":
    run_comparison_test() 