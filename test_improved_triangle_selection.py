#!/usr/bin/env python3
"""
Test Improved Triangle Selection
Tests our new advanced scoring system that should pick better triangles
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# Add project path
sys.path.append('.')

def main():
    print("🎯 TESTING IMPROVED TRIANGLE SELECTION")
    print("=" * 60)
    
    try:
        # Import our modules
        from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
        from src.spygate.ml.template_triangle_detector import YOLOIntegratedTriangleDetector
        from src.spygate.core.hardware import HardwareDetector
        print("✅ Successfully imported SpygateAI modules")
        
        # Initialize our detection system
        hardware = HardwareDetector()
        analyzer = EnhancedGameAnalyzer(hardware=hardware)
        
        # Initialize template detector with debug output
        debug_dir = Path("debug_improved_selection")
        debug_dir.mkdir(exist_ok=True)
        
        template_detector = YOLOIntegratedTriangleDetector(
            game_analyzer=analyzer,
            debug_output_dir=debug_dir
        )
        
        print(f"🔧 Hardware Tier: {hardware.detect_tier().name}")
        print(f"🎯 Using 5-class YOLO model with improved triangle selection")
        print()
        
        # Test on 5 random images to see the improvement
        screenshots_dir = Path("6.12 screenshots")
        image_files = list(screenshots_dir.glob("*.png"))
        
        if not image_files:
            print("❌ No screenshots found in '6.12 screenshots' directory")
            return
        
        # Select 5 random images for focused testing
        test_images = random.sample(image_files, min(5, len(image_files)))
        
        print(f"🧪 Testing improved selection on {len(test_images)} images...")
        print()
        
        for i, img_path in enumerate(test_images, 1):
            print(f"📸 Testing Image {i}: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"   ❌ Could not load image")
                continue
            
            try:
                # Run our improved detection
                matches = template_detector.detect_triangles_in_yolo_regions(image)
                
                print(f"   🎯 Final Results: {len(matches)} triangles selected")
                
                for match in matches:
                    triangle_type = match.triangle_type.value
                    direction = match.direction.value
                    conf = match.confidence
                    size = f"{match.bounding_box[2]}x{match.bounding_box[3]}"
                    scale = match.scale_factor
                    template = match.template_name
                    
                    print(f"   ✅ {triangle_type.upper()} {direction}: {template}")
                    print(f"      📊 Confidence: {conf:.3f} | Size: {size} | Scale: {scale:.2f}x")
                
                # Create visualization
                vis_image = image.copy()
                
                for match in matches:
                    x, y, w, h = match.bounding_box
                    
                    # Color based on triangle type
                    if match.triangle_type.value == "possession":
                        color = (0, 255, 255)  # Yellow for possession
                        label_color = (0, 0, 0)  # Black text
                    else:
                        color = (255, 0, 255)  # Magenta for territory
                        label_color = (255, 255, 255)  # White text
                    
                    # Draw thick bounding box for selected triangles
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
                    
                    # Draw label with detailed info
                    label = f"{match.triangle_type.value.upper()} {match.direction.value}"
                    detail = f"conf:{match.confidence:.2f} scale:{match.scale_factor:.1f}x"
                    
                    # Background for text
                    cv2.rectangle(vis_image, (x, y-40), (x + 200, y), color, -1)
                    cv2.putText(vis_image, label, (x + 5, y - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
                    cv2.putText(vis_image, detail, (x + 5, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
                
                # Save visualization
                vis_filename = f"improved_selection_{i:02d}_{img_path.stem}.jpg"
                cv2.imwrite(vis_filename, vis_image)
                print(f"   🖼️  Saved: {vis_filename}")
                
            except Exception as e:
                print(f"   ❌ Error processing image: {e}")
            
            print()
        
        print("=" * 60)
        print("🏆 IMPROVED SELECTION TEST COMPLETE")
        print("=" * 60)
        print("✨ KEY IMPROVEMENTS:")
        print("   🎯 Advanced scoring system with 6 factors")
        print("   📏 Smart size scoring (optimal ranges, not just bigger)")
        print("   🏷️  Template quality scoring (prefers Madden-specific)")
        print("   📐 Aspect ratio validation")
        print("   🔍 Reasonable scale factor preferences")
        print("   📍 Position-aware scoring")
        print()
        print("📁 Check these files:")
        print("   🖼️  improved_selection_*.jpg - Visual results")
        print("   🔍 debug_improved_selection/ - Debug output")
        print()
        print("🔥 Should now select BETTER triangles even when false positives are larger!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 