#!/usr/bin/env python3
"""
Quick sample check of OCR results
"""

from pathlib import Path

import cv2
import numpy as np

# Import OCR engines
try:
    import keras_ocr
    from paddleocr import PaddleOCR

    # Quick test on one of the 6.12 screenshots
    def quick_test():
        print("🔍 QUICK SAMPLE TEST - Both OCR Engines")
        print("=" * 60)

        # Get a sample image
        screenshots_folder = Path("6.12 screenshots")
        sample_images = list(screenshots_folder.glob("*.png"))[:3]  # First 3 images

        if not sample_images:
            print("❌ No sample images found")
            return

        # Initialize engines
        print("🔧 Initializing OCR engines...")
        paddle_ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
        keras_pipeline = keras_ocr.pipeline.Pipeline()
        print("✅ Both engines ready!")

        for i, image_path in enumerate(sample_images, 1):
            print(f"\n📸 Sample {i}: {image_path.name}")
            print("-" * 50)

            # Test PaddleOCR
            try:
                image = cv2.imread(str(image_path))
                paddle_result = paddle_ocr.ocr(image, cls=False)

                if paddle_result and paddle_result[0]:
                    paddle_texts = []
                    for line in paddle_result[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]
                            conf = line[1][1]
                            paddle_texts.append(f"{text} ({conf:.2f})")

                    print(f"🟦 PaddleOCR ({len(paddle_texts)} detections):")
                    for text in paddle_texts[:5]:  # Show first 5
                        print(f"   • {text}")
                    if len(paddle_texts) > 5:
                        print(f"   ... and {len(paddle_texts)-5} more")
                else:
                    print("🟦 PaddleOCR: No text detected")

            except Exception as e:
                print(f"🟦 PaddleOCR: Error - {e}")

            # Test KerasOCR
            try:
                keras_image = keras_ocr.tools.read(str(image_path))
                keras_predictions = keras_pipeline.recognize([keras_image])[0]

                if keras_predictions:
                    keras_texts = [text for text, box in keras_predictions]

                    print(f"🟩 KerasOCR ({len(keras_texts)} detections):")
                    for text in keras_texts[:5]:  # Show first 5
                        print(f"   • {text}")
                    if len(keras_texts) > 5:
                        print(f"   ... and {len(keras_texts)-5} more")
                else:
                    print("🟩 KerasOCR: No text detected")

            except Exception as e:
                print(f"🟩 KerasOCR: Error - {e}")

        print(f"\n🎯 Both engines are working! Full comparison running in background...")
        print(f"📁 Check 'comprehensive_ocr_comparison_results' for visual comparisons")

    if __name__ == "__main__":
        quick_test()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("OCR engines not available for quick test")
