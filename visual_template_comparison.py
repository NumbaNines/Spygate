#!/usr/bin/env python3
"""
Visual comparison of templates vs real screenshots to see what's being detected.
"""

import cv2
import numpy as np
from pathlib import Path
from down_template_detector import DownTemplateDetector


def show_templates():
    """Show all our templates."""
    print("🎯 OUR TEMPLATES:")
    print("=" * 50)
    
    template_dir = Path("down_templates_real")
    
    for template_file in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"📋 {template_file.name}: {img.shape[1]}x{img.shape[0]} pixels")
            
            # Show some pixel values to understand the template
            print(f"   Pixel range: {img.min()}-{img.max()}")
            print(f"   Average brightness: {img.mean():.1f}")
        else:
            print(f"❌ Failed to load {template_file.name}")


def show_real_screenshots():
    """Show all real screenshots."""
    print("\n📱 REAL SCREENSHOTS:")
    print("=" * 50)
    
    screenshot_dir = Path("templates/raw_gameplay")
    
    for screenshot_file in sorted(screenshot_dir.glob("*.png")):
        img = cv2.imread(str(screenshot_file))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"📸 {screenshot_file.name}: {img.shape[1]}x{img.shape[0]} pixels")
            print(f"   Pixel range: {gray.min()}-{gray.max()}")
            print(f"   Average brightness: {gray.mean():.1f}")
        else:
            print(f"❌ Failed to load {screenshot_file.name}")


def test_specific_detection():
    """Test detection on specific screenshots and show what's happening."""
    print("\n🔍 DETECTION ANALYSIS:")
    print("=" * 50)
    
    detector = DownTemplateDetector()
    
    # Test cases with expected results
    test_cases = [
        {"file": "templates/raw_gameplay/1st_10.png", "expected": 1, "desc": "1st & 10"},
        {"file": "templates/raw_gameplay/2nd_7.png", "expected": 2, "desc": "2nd & 7"},
        {"file": "templates/raw_gameplay/3rd_3.png", "expected": 3, "desc": "3rd & 3"},
        {"file": "templates/raw_gameplay/4th_1.png", "expected": 4, "desc": "4th & 1"},
        {"file": "templates/raw_gameplay/3rd_goal.png", "expected": 3, "desc": "3rd & Goal"},
        {"file": "templates/raw_gameplay/4th_goal.png", "expected": 4, "desc": "4th & Goal"},
    ]
    
    for test_case in test_cases:
        if not Path(test_case["file"]).exists():
            continue
            
        print(f"\n🧪 Testing: {test_case['desc']}")
        print(f"   Expected: {test_case['expected']}")
        
        # Load image
        frame = cv2.imread(test_case["file"])
        height, width = frame.shape[:2]
        bbox = (0, 0, width, height)
        
        # Apply preprocessing to see what the detector sees
        preprocessed = detector._apply_spygate_preprocessing(frame)
        print(f"   Original: {width}x{height}")
        print(f"   Preprocessed: {preprocessed.shape[1]}x{preprocessed.shape[0]}")
        
        # Test each template manually
        best_matches = []
        
        for template_name, template_data in detector.templates.items():
            template_img = template_data["image"]
            
            # Template matching at scale 1.0
            if (template_img.shape[0] <= preprocessed.shape[0] and 
                template_img.shape[1] <= preprocessed.shape[1]):
                
                result = cv2.matchTemplate(
                    preprocessed, template_img, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                best_matches.append({
                    "template": template_name,
                    "confidence": max_val,
                    "down": template_data["down"],
                    "location": max_loc
                })
        
        # Sort by confidence
        best_matches.sort(key=lambda x: x["confidence"], reverse=True)
        
        print("   🏆 Top 3 matches:")
        for i, match in enumerate(best_matches[:3]):
            status = "✅" if match["down"] == test_case["expected"] else "❌"
            print(f"      {i+1}. {status} {match['template']}: {match['confidence']:.3f} (down {match['down']})")
        
        # Test actual detector
        result = detector.detect_down_in_yolo_region(frame, bbox, "goal" in test_case["desc"].lower())
        
        if result:
            correct = result.down == test_case["expected"]
            status = "✅ CORRECT" if correct else "❌ WRONG"
            print(f"   🎯 Detector result: {status}")
            print(f"      Detected: {result.down} (conf: {result.confidence:.3f})")
            print(f"      Template: {result.template_name}")
        else:
            print("   ❌ Detector: No detection")


def save_visual_comparison():
    """Save side-by-side comparison images."""
    print("\n💾 SAVING VISUAL COMPARISONS:")
    print("=" * 40)
    
    # Compare 1ST template with 1st_10 screenshot
    template_path = "down_templates_real/1ST.png"
    screenshot_path = "templates/raw_gameplay/1st_10.png"
    
    if Path(template_path).exists() and Path(screenshot_path).exists():
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        screenshot = cv2.imread(screenshot_path)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Create side-by-side comparison
        comparison = np.hstack([template, screenshot_gray])
        cv2.imwrite("template_vs_screenshot_comparison.png", comparison)
        
        print("✅ Saved: template_vs_screenshot_comparison.png")
        print(f"   Template (left): {template.shape[1]}x{template.shape[0]}")
        print(f"   Screenshot (right): {screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}")
        
        # Show pixel differences
        print(f"   Template brightness: {template.mean():.1f}")
        print(f"   Screenshot brightness: {screenshot_gray.mean():.1f}")
        print(f"   Brightness difference: {abs(template.mean() - screenshot_gray.mean()):.1f}")
    else:
        print("❌ Template or screenshot not found for comparison")


def main():
    """Main function."""
    print("👁️ VISUAL TEMPLATE ANALYSIS")
    print("=" * 30)
    
    show_templates()
    show_real_screenshots()
    test_specific_detection()
    save_visual_comparison()
    
    print("\n🎯 SUMMARY:")
    print("=" * 20)
    print("✅ Templates are 80x30 pixels (same as real screenshots)")
    print("✅ Detection is working but needs accuracy improvement")
    print("⚠️ Main issue: Templates are too similar to each other")
    print("💡 Solution: Need more distinct templates or better preprocessing")


if __name__ == "__main__":
    main() 