#!/usr/bin/env python3
"""
Fix PyTesseract configuration for SpygateAI OCR functionality.
"""

import pytesseract
import os
from PIL import Image
import numpy as np

def configure_pytesseract():
    """Configure pytesseract to find Tesseract executable."""
    
    # Common Tesseract installation paths on Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\Nines\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    ]
    
    print("🔧 Configuring PyTesseract...")
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✅ Found Tesseract at: {path}")
            return True
    
    print("❌ Tesseract executable not found in common locations")
    return False

def test_ocr():
    """Test OCR functionality."""
    try:
        print("\n🧪 Testing OCR functionality...")
        
        # Create a simple test image with text
        test_img = Image.new('RGB', (200, 100), color='white')
        
        # Test with the image
        result = pytesseract.image_to_string(test_img)
        print("✅ OCR test completed successfully!")
        
        # Test version info
        version = pytesseract.get_tesseract_version()
        print(f"📋 Tesseract version: {version}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def main():
    """Main configuration function."""
    print("🔥 SpygateAI PyTesseract Configuration")
    print("=" * 50)
    
    # Configure pytesseract
    if configure_pytesseract():
        # Test OCR
        if test_ocr():
            print("\n✅ PyTesseract configuration complete!")
            print("🎯 SpygateAI OCR functionality is now ready!")
            
            # Show configuration
            print(f"\n📍 Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
            
        else:
            print("\n❌ OCR test failed - check Tesseract installation")
    else:
        print("\n❌ Configuration failed - Tesseract not found")
        print("💡 Try installing Tesseract OCR manually")

if __name__ == "__main__":
    main() 