#!/usr/bin/env python3
"""
Triangle Detection Test - 25 Random Images
Tests our enhanced YOLO + template matching system on random screenshots
"""

import os
import sys
import cv2
import numpy as np
import json
import random
from pathlib import Path

# Add project path
sys.path.append('.')

def main():
    print("🎯 SPYGATE TRIANGLE TEMPLATE MATCHING TEST")
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
        
        # Initialize template triangle detector
        template_detector = YOLOIntegratedTriangleDetector(
            game_analyzer=analyzer,
            debug_output_dir=Path("debug_template_matching")
        )
        
        print(f"🔧 Hardware Tier: {hardware.detect_tier().name}")
        print(f"🎯 Using 5-class YOLO model with template matching")
        print(f"📁 Debug output: debug_template_matching/")
        
        # Get random images from 6.12 screenshots
        screenshots_dir = Path("6.12 screenshots")
        if not screenshots_dir.exists():
            print(f"❌ Screenshots directory not found: {screenshots_dir}")
            return
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(screenshots_dir.glob(ext))
        
        if len(image_files) < 25:
            print(f"⚠️ Only found {len(image_files)} images, using all available")
            selected_images = image_files
        else:
            selected_images = random.sample(image_files, 25)
        
        print(f"🖼️ Testing {len(selected_images)} random images...")
        print("-" * 60)
        
        results = []
        total_template_matches = 0
        total_yolo_detections = 0
        
        for i, img_path in enumerate(selected_images, 1):
            print(f"\n📸 Image {i:2d}: {img_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"   ❌ Failed to load image")
                    continue
                
                # Run YOLO detection first
                yolo_detections = analyzer.model.detect(image)
                yolo_count = len(yolo_detections) if yolo_detections else 0
                total_yolo_detections += yolo_count
                
                print(f"   🎯 YOLO detected {yolo_count} objects")
                
                # Run template matching within YOLO regions
                template_matches = template_detector.detect_triangles_in_yolo_regions(image)
                template_count = len(template_matches)
                total_template_matches += template_count
                
                print(f"   🔍 Template matching found {template_count} triangles:")
                
                if template_matches:
                    for match in template_matches:
                        print(f"      - {match.triangle_type.value} {match.direction.value}: {match.confidence:.3f} (scale: {match.scale_factor:.1f}x)")
                        print(f"        Template: {match.template_name}")
                        print(f"        Position: ({match.position[0]}, {match.position[1]})")
                
                # Create comprehensive visualization
                vis_image = image.copy()
                
                # Draw YOLO detections in colored boxes
                yolo_colors = {
                    "hud": (0, 255, 0),  # Green
                    "possession_triangle_area": (255, 0, 0),  # Blue  
                    "territory_triangle_area": (0, 0, 255),  # Red
                    "preplay_indicator": (255, 255, 0),  # Cyan
                    "play_call_screen": (255, 0, 255),  # Magenta
                }
                
                if yolo_detections:
                    for detection in yolo_detections:
                        class_name = detection['class']
                        bbox = detection['bbox']
                        conf = detection['confidence']
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        color = yolo_colors.get(class_name, (128, 128, 128))
                        
                        # Draw YOLO bounding box
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis_image, f"YOLO: {class_name} {conf:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw template matches with different style
                for match in template_matches:
                    x, y, w, h = match.bounding_box
                    
                    # Use bright colors for template matches
                    template_color = (0, 255, 255)  # Yellow for template matches
                    
                    # Draw template match bounding box with different style
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), template_color, 3)
                    
                    # Add template match label
                    label = f"TEMPLATE: {match.triangle_type.value} {match.direction.value} {match.confidence:.2f}"
                    cv2.putText(vis_image, label, (x, y + h + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, template_color, 1)
                
                # Save visualization
                vis_filename = f"template_test_result_{i:02d}_{img_path.stem}.jpg"
                cv2.imwrite(vis_filename, vis_image)
                
                results.append({
                    "image": img_path.name,
                    "yolo_detections": yolo_count,
                    "template_matches": template_count,
                    "template_details": [
                        {
                            "type": match.triangle_type.value,
                            "direction": match.direction.value,
                            "confidence": match.confidence,
                            "template": match.template_name,
                            "scale": match.scale_factor
                        } for match in template_matches
                    ]
                })
                
            except Exception as e:
                print(f"   ❌ Error processing image: {e}")
                continue
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("🏆 TEMPLATE MATCHING TEST SUMMARY")
        print("=" * 60)
        print(f"📊 Total Images Processed: {len(results)}")
        print(f"🎯 Total YOLO Detections: {total_yolo_detections}")
        print(f"🔍 Total Template Matches: {total_template_matches}")
        print(f"📈 Template Match Rate: {total_template_matches/len(results):.1f} per image")
        
        # Analyze template match types
        possession_matches = sum(1 for r in results for t in r['template_details'] if t['type'] == 'possession')
        territory_matches = sum(1 for r in results for t in r['template_details'] if t['type'] == 'territory')
        
        print(f"\n🔍 Template Match Breakdown:")
        print(f"   📍 Possession Triangles: {possession_matches}")
        print(f"   🗺️  Territory Triangles: {territory_matches}")
        
        # Show confidence distribution
        all_confidences = [t['confidence'] for r in results for t in r['template_details']]
        if all_confidences:
            print(f"\n📊 Template Confidence Stats:")
            print(f"   Average: {np.mean(all_confidences):.3f}")
            print(f"   Maximum: {np.max(all_confidences):.3f}")
            print(f"   Minimum: {np.min(all_confidences):.3f}")
        
        # Save detailed results
        with open("template_matching_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: template_matching_results.json")
        print(f"🖼️ Visualizations saved as: template_test_result_*.jpg")
        print(f"🔍 Debug output in: debug_template_matching/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 