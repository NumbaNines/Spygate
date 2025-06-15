#!/usr/bin/env python3
"""
Quick progress checker for comprehensive OCR comparison
"""

import json
import os
from pathlib import Path


def check_progress():
    results_dir = Path("comprehensive_ocr_comparison_results")

    if not results_dir.exists():
        print("❌ Results directory not found - test may not have started yet")
        return

    # Count comparison images
    comparison_images = list(results_dir.glob("comparison_*.png"))
    total_screenshots = len(list(Path("6.12 screenshots").glob("*.png")))

    print(f"📊 PROGRESS UPDATE")
    print(f"=" * 50)
    print(f"🖼️  Processed: {len(comparison_images)}/{total_screenshots} images")
    print(f"📈 Progress: {(len(comparison_images)/total_screenshots)*100:.1f}%")

    # Check if detailed results exist
    results_file = results_dir / "detailed_results.json"
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                results = json.load(f)

            print(f"\n📋 CURRENT RESULTS")
            print(f"-" * 30)
            print(f"🟦 PaddleOCR Success: {results['paddle_success']}/{results['total_images']}")
            print(f"🟩 KerasOCR Success: {results['keras_success']}/{results['total_images']}")

            # Show some sample detections
            if results["detailed_results"]:
                print(f"\n🔍 SAMPLE DETECTIONS (Latest 3)")
                print(f"-" * 40)

                for result in results["detailed_results"][-3:]:
                    print(f"\n📸 {result['image_name']}")

                    if result["paddle_success"]:
                        paddle_texts = result["paddle_texts"][:3]  # First 3 texts
                        print(f"  🟦 PaddleOCR: {' | '.join(paddle_texts)}")
                    else:
                        print(f"  🟦 PaddleOCR: ❌ {result.get('paddle_error', 'Failed')}")

                    if result["keras_success"]:
                        keras_texts = result["keras_texts"][:3]  # First 3 texts
                        print(f"  🟩 KerasOCR:  {' | '.join(keras_texts)}")
                    else:
                        print(f"  🟩 KerasOCR:  ❌ {result.get('keras_error', 'Failed')}")

        except Exception as e:
            print(f"❌ Error reading results: {e}")

    print(f"\n📁 Visual comparisons available in: {results_dir}")
    print(f"🔍 Check the comparison_*.png files to see side-by-side OCR results!")

    # Show latest comparison file
    if comparison_images:
        latest_comparison = max(comparison_images, key=lambda x: x.stat().st_mtime)
        print(f"📸 Latest comparison: {latest_comparison.name}")


if __name__ == "__main__":
    check_progress()
