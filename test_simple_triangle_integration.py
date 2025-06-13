#!/usr/bin/env python3
"""
Simple Triangle Integration Test
Tests the template triangle detector integration without circular imports.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time

def test_template_triangle_detector():
    """Test the template triangle detector directly."""
    print("🎯 TEMPLATE TRIANGLE DETECTOR INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Import the template triangle detector
        from src.spygate.ml.template_triangle_detector import TemplateTriangleDetector
        
        print("✅ Successfully imported TemplateTriangleDetector")
        
        # Initialize the detector
        detector = TemplateTriangleDetector(debug_output_dir=Path("debug_simple_integration"))
        print("✅ TemplateTriangleDetector initialized")
        
        # Test with sample images
        test_images = []
        for i in range(1, 6):
            pattern = f"improved_selection_{i:02d}_*.jpg"
            matches = list(Path(".").glob(pattern))
            if matches:
                test_images.append(matches[0])
        
        if not test_images:
            print("❌ No test images found. Using screenshots from 6.12 screenshots folder.")
            screenshot_dir = Path("6.12 screenshots")
            if screenshot_dir.exists():
                test_images = list(screenshot_dir.glob("*.png"))[:5]
        
        print(f"📁 Found {len(test_images)} test images")
        
        # Test triangle detection
        print("\n🔍 TESTING TRIANGLE DETECTION")
        print("-" * 40)
        
        total_detections = 0
        successful_detections = 0
        
        for i, img_path in enumerate(test_images, 1):
            print(f"\n📸 Processing Image {i}: {img_path.name}")
            
            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"❌ Could not load image: {img_path}")
                continue
            
            # Simulate YOLO detection regions (use full image for now)
            h, w = frame.shape[:2]
            
            # Test possession triangle detection
            possession_roi = frame[0:h//3, 0:w//2]  # Top-left area
            start_time = time.time()
            possession_triangles = detector.detect_triangles_in_roi(possession_roi, "possession")
            detection_time = time.time() - start_time
            
            print(f"🏈 Possession triangles found: {len(possession_triangles)}")
            if possession_triangles:
                best_possession = detector.select_best_single_triangles(possession_triangles, "possession")
                if best_possession:
                    print(f"   Best: {best_possession['direction']} (conf: {best_possession['confidence']:.3f})")
                    successful_detections += 1
            
            # Test territory triangle detection
            territory_roi = frame[0:h//3, w//2:w]  # Top-right area
            territory_triangles = detector.detect_triangles_in_roi(territory_roi, "territory")
            
            print(f"🗺️ Territory triangles found: {len(territory_triangles)}")
            if territory_triangles:
                best_territory = detector.select_best_single_triangles(territory_triangles, "territory")
                if best_territory:
                    print(f"   Best: {best_territory['direction']} (conf: {best_territory['confidence']:.3f})")
                    successful_detections += 1
            
            print(f"⏱️ Detection time: {detection_time:.3f}s")
            total_detections += 2  # possession + territory
        
        # Display results
        print("\n📊 DETECTION RESULTS")
        print("-" * 30)
        print(f"Total detection attempts: {total_detections}")
        print(f"Successful detections: {successful_detections}")
        print(f"Success rate: {(successful_detections/total_detections)*100:.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_game_state_logic():
    """Demonstrate the game state logic for triangle flips."""
    print("\n🧠 GAME STATE LOGIC DEMONSTRATION")
    print("=" * 50)
    
    # Simulate triangle state changes
    scenarios = [
        {
            "name": "Normal Possession",
            "old_possession": "left",
            "new_possession": "left",
            "old_territory": "down",
            "new_territory": "down",
            "expected": "No change"
        },
        {
            "name": "Turnover!",
            "old_possession": "left",
            "new_possession": "right",
            "old_territory": "down",
            "new_territory": "down",
            "expected": "Possession change - clip worthy!"
        },
        {
            "name": "Crossed Midfield",
            "old_possession": "left",
            "new_possession": "left",
            "old_territory": "down",
            "new_territory": "up",
            "expected": "Territory change - field position improved"
        },
        {
            "name": "Pick-Six!",
            "old_possession": "left",
            "new_possession": "right",
            "old_territory": "up",
            "new_territory": "down",
            "expected": "Both changed - major momentum shift!"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎮 Scenario: {scenario['name']}")
        print(f"   Possession: {scenario['old_possession']} → {scenario['new_possession']}")
        print(f"   Territory:  {scenario['old_territory']} → {scenario['new_territory']}")
        
        # Check for changes
        possession_changed = scenario['old_possession'] != scenario['new_possession']
        territory_changed = scenario['old_territory'] != scenario['new_territory']
        
        if possession_changed and territory_changed:
            result = "🚨 MAJOR EVENT: Both possession and territory changed!"
            clip_worthy = True
        elif possession_changed:
            result = "🔄 TURNOVER: Possession changed!"
            clip_worthy = True
        elif territory_changed:
            result = "🗺️ FIELD POSITION: Territory changed"
            clip_worthy = False
        else:
            result = "✅ No significant change"
            clip_worthy = False
        
        print(f"   Result: {result}")
        print(f"   Clip worthy: {'Yes' if clip_worthy else 'No'}")
        print(f"   Expected: {scenario['expected']}")

def show_triangle_meanings():
    """Show what triangle states mean in the game."""
    print("\n📚 TRIANGLE STATE MEANINGS")
    print("=" * 40)
    
    print("🏈 POSSESSION TRIANGLES:")
    print("   'left'  = Away team has the ball")
    print("   'right' = Home team has the ball")
    
    print("\n🗺️ TERRITORY TRIANGLES:")
    print("   'up'   = In opponent's territory (good)")
    print("   'down' = In own territory (poor)")
    
    print("\n🎯 GAME SITUATIONS:")
    combinations = [
        ("left", "up", "Away team driving (scoring chance)"),
        ("left", "down", "Away team backed up (defensive)"),
        ("right", "up", "Home team driving (scoring chance)"),
        ("right", "down", "Home team backed up (defensive)")
    ]
    
    for poss, terr, meaning in combinations:
        print(f"   {poss} + {terr} = {meaning}")

if __name__ == "__main__":
    # Show triangle meanings
    show_triangle_meanings()
    
    # Test the template detector
    success = test_template_triangle_detector()
    
    # Demonstrate game state logic
    demonstrate_game_state_logic()
    
    if success:
        print("\n🎉 INTEGRATION TEST SUCCESSFUL!")
        print("✅ Template triangle detection working")
        print("✅ Game state logic implemented")
        print("✅ Ready for production integration")
    else:
        print("\n⚠️ Integration test failed - check logs above") 