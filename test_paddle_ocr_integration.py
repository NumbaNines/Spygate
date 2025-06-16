"""
Test script to verify PaddleOCR integration in SpygateAI
"""

import cv2
import numpy as np
from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.core.hardware import HardwareDetector

def test_paddle_ocr_integration():
    """Test that PaddleOCR is properly integrated."""
    
    print("🧪 Testing PaddleOCR Integration...")
    
    try:
        # Initialize the analyzer
        print("1. Initializing EnhancedGameAnalyzer...")
        hardware = HardwareDetector()
        analyzer = EnhancedGameAnalyzer(hardware=hardware)
        print("✅ Analyzer initialized successfully")
        
        # Check OCR system
        print("\n2. Checking OCR system...")
        print(f"   OCR type: {type(analyzer.ocr).__name__}")
        print(f"   Has extract_text: {hasattr(analyzer.ocr, 'extract_text')}")
        print(f"   Has extract_down_distance: {hasattr(analyzer.ocr, 'extract_down_distance')}")
        print(f"   Has extract_game_clock: {hasattr(analyzer.ocr, 'extract_game_clock')}")
        print(f"   Has extract_play_clock: {hasattr(analyzer.ocr, 'extract_play_clock')}")
        
        # Test with a simple image
        print("\n3. Testing OCR extraction...")
        # Create a test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "1ST & 10", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test extract_text
        text_result = analyzer.ocr.extract_text(test_image)
        print(f"   extract_text result: '{text_result}'")
        
        # Test extract_down_distance
        down_distance_result = analyzer.ocr.extract_down_distance(test_image)
        print(f"   extract_down_distance result: '{down_distance_result}'")
        
        print("\n✅ PaddleOCR integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paddle_ocr_integration() 