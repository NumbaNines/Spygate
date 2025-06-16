#!/usr/bin/env python3
"""
Test actual GOAL vs normal template positioning.
Compare 3RD (GOAL) template against 1ST/2ND/4TH (normal) templates.
"""

import cv2
import numpy as np
from pathlib import Path
import json

def compare_goal_vs_normal():
    """Compare GOAL template positioning vs normal templates."""
    
    templates_dir = Path("down_templates_real")
    metadata_path = templates_dir / "templates_metadata.json"
    
    if not metadata_path.exists():
        print("❌ No metadata found!")
        return
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("🎯 SpygateAI GOAL vs Normal Template Analysis")
    print("=" * 50)
    
    # Analyze positioning
    print("📍 Template Positions (X coordinate):")
    positions = {}
    
    for down_type, data in metadata.items():
        x_start = data['crop_coords'][0]
        positions[down_type] = x_start
        status = "🥅 GOAL" if down_type == "3RD" else "📏 Normal"
        print(f"   {down_type}: X={x_start:4d} {status}")
    
    # Calculate shifts
    print("\n📊 Position Analysis:")
    normal_avg = np.mean([positions['1ST'], positions['2ND'], positions['4TH']])
    goal_pos = positions['3RD']
    shift = normal_avg - goal_pos
    
    print(f"   Normal average X: {normal_avg:.1f}")
    print(f"   GOAL position X:  {goal_pos}")
    print(f"   GOAL shift:       {shift:.1f} pixels LEFT")
    
    # Load actual templates for visual comparison
    print("\n🔍 Template Matching Test:")
    
    # Load templates
    templates = {}
    for down_type in ['1ST', '2ND', '3RD', '4TH']:
        template_path = templates_dir / f"{down_type}.png"
        if template_path.exists():
            templates[down_type] = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            print(f"   ✅ Loaded {down_type}: {templates[down_type].shape[1]}x{templates[down_type].shape[0]}px")
    
    # Test cross-matching (normal templates vs GOAL-positioned image)
    if '1ST' in templates and '3RD' in templates:
        print(f"\n🧪 Cross-Template Matching Test:")
        
        # Create test scenario: 1ST template at GOAL position vs normal position
        test_width = 400
        test_height = 200
        
        # Test 1ST template at normal position
        normal_test = np.zeros((test_height, test_width), dtype=np.uint8)
        normal_x = test_width // 2 - templates['1ST'].shape[1] // 2
        normal_y = test_height // 2 - templates['1ST'].shape[0] // 2
        
        normal_test[normal_y:normal_y + templates['1ST'].shape[0],
                   normal_x:normal_x + templates['1ST'].shape[1]] = templates['1ST']
        
        # Test 1ST template at GOAL-shifted position (25px left)
        goal_test = np.zeros((test_height, test_width), dtype=np.uint8)
        goal_x = normal_x - int(shift)  # Shift left by GOAL amount
        
        if goal_x >= 0:
            goal_test[normal_y:normal_y + templates['1ST'].shape[0],
                     goal_x:goal_x + templates['1ST'].shape[1]] = templates['1ST']
            
            # Match 1ST template against both scenarios
            normal_result = cv2.matchTemplate(normal_test, templates['1ST'], cv2.TM_CCOEFF_NORMED)
            goal_result = cv2.matchTemplate(goal_test, templates['1ST'], cv2.TM_CCOEFF_NORMED)
            
            normal_confidence = np.max(normal_result)
            goal_confidence = np.max(goal_result)
            
            confidence_loss = (normal_confidence - goal_confidence) / normal_confidence * 100
            
            print(f"   1ST template at normal position: {normal_confidence:.3f} confidence")
            print(f"   1ST template at GOAL position:   {goal_confidence:.3f} confidence")
            print(f"   Confidence loss: {confidence_loss:.1f}%")
            
            # Evaluation
            if confidence_loss < 10:
                print("   ✅ Minimal impact - current templates should work")
            elif confidence_loss < 30:
                print("   ⚠️ Moderate impact - consider GOAL-specific templates")
            else:
                print("   ❌ Significant impact - GOAL templates recommended")
        else:
            print("   ❌ Shift too large for test image")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    if shift > 20:
        print("   📝 CREATE GOAL-SPECIFIC TEMPLATES:")
        print("   - 1ST_GOAL, 2ND_GOAL, 3RD_GOAL, 4TH_GOAL")
        print("   - Use 3rd.png (your GOAL screenshot) for all down types")
        print("   - Crop each down type from GOAL situations")
        print("   - This ensures perfect positioning accuracy")
    elif shift > 10:
        print("   📝 CONSIDER dual template system:")
        print("   - Current templates for normal situations")
        print("   - Additional GOAL templates for red zone")
    else:
        print("   📝 Current templates should work fine")
    
    return shift, metadata

if __name__ == "__main__":
    compare_goal_vs_normal() 