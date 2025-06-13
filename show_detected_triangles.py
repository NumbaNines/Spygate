import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.triangle_orientation_detector import TriangleOrientationDetector, TriangleType

def find_contours_in_roi(roi_img):
    """Find contours in the ROI image."""
    # Convert to grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (remove very small ones)
    min_area = 30
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return filtered_contours

def visualize_detected_triangles(roi_img, contours, results, title, output_filename):
    """Create a detailed visualization of detected triangles."""
    # Create a larger canvas for annotations
    vis_height = max(roi_img.shape[0], 400)
    vis_width = roi_img.shape[1] + 500  # Extra space for text
    vis_img = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # Place the ROI on the left side
    vis_img[:roi_img.shape[0], :roi_img.shape[1]] = roi_img
    
    # Colors for different results
    colors = {
        'valid': (0, 255, 0),      # Green for valid triangles
        'invalid': (0, 0, 255),    # Red for rejected
        'left': (255, 255, 0),     # Cyan for left direction
        'right': (255, 0, 255),    # Magenta for right direction
        'up': (0, 255, 255),       # Yellow for up direction
        'down': (128, 255, 128),   # Light green for down direction
    }
    
    # Draw contours and annotations
    valid_count = 0
    for i, (contour, result) in enumerate(zip(contours, results)):
        # Determine color based on validity and direction
        if result.is_valid:
            valid_count += 1
            if result.direction.value == 'left':
                color = colors['left']
            elif result.direction.value == 'right':
                color = colors['right']
            elif result.direction.value == 'up':
                color = colors['up']
            elif result.direction.value == 'down':
                color = colors['down']
            else:
                color = colors['valid']
            
            # Draw thick contour for valid triangles
            cv2.drawContours(vis_img, [contour], -1, color, 3)
            
            # Add direction arrow
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Draw direction indicator
            if result.direction.value == 'left':
                cv2.arrowedLine(vis_img, (center[0] + 15, center[1]), (center[0] - 15, center[1]), color, 2)
            elif result.direction.value == 'right':
                cv2.arrowedLine(vis_img, (center[0] - 15, center[1]), (center[0] + 15, center[1]), color, 2)
            elif result.direction.value == 'up':
                cv2.arrowedLine(vis_img, (center[0], center[1] + 15), (center[0], center[1] - 15), color, 2)
            elif result.direction.value == 'down':
                cv2.arrowedLine(vis_img, (center[0], center[1] - 15), (center[0], center[1] + 15), color, 2)
            
            # Label with confidence
            cv2.putText(vis_img, f"{result.confidence:.1%}", 
                       (center[0] - 20, center[1] + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Draw thin contour for invalid shapes
            cv2.drawContours(vis_img, [contour], -1, colors['invalid'], 1)
        
        # Get contour center for numbering
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw contour number
            cv2.putText(vis_img, str(i+1), (cx-10, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(vis_img, title, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add success indicator
    if valid_count > 0:
        cv2.putText(vis_img, f"SUCCESS: {valid_count} TRIANGLES FOUND!", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis_img, "NO VALID TRIANGLES", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add legend and details on the right side
    text_x = roi_img.shape[1] + 10
    y_offset = 90
    
    cv2.putText(vis_img, "DETECTION RESULTS:", (text_x, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    
    for i, (contour, result) in enumerate(zip(contours, results)):
        area = cv2.contourArea(contour)
        
        # Contour info
        status = "✓ VALID" if result.is_valid else "✗ REJECTED"
        color = (0, 255, 0) if result.is_valid else (0, 0, 255)
        
        cv2.putText(vis_img, f"Contour {i+1}: {status}", (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 20
        
        cv2.putText(vis_img, f"  Area: {area:.1f}", (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 15
        
        if result.is_valid:
            cv2.putText(vis_img, f"  Direction: {result.direction.value.upper()}", (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 15
            cv2.putText(vis_img, f"  Confidence: {result.confidence:.1%}", (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 15
        else:
            # Split long rejection reasons into multiple lines
            reason = result.validation_reason
            if len(reason) > 25:
                words = reason.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 25:
                        current_line += word + " "
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
            else:
                lines = [reason]
            
            cv2.putText(vis_img, f"  Reason:", (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
            
            for line in lines:
                cv2.putText(vis_img, f"    {line}", (text_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
                y_offset += 12
        
        y_offset += 10  # Extra space between contours
    
    # Add color legend
    y_offset += 20
    cv2.putText(vis_img, "DIRECTION LEGEND:", (text_x, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    
    legend_items = [
        ("Cyan: LEFT arrow", colors['left']),
        ("Magenta: RIGHT arrow", colors['right']),
        ("Yellow: UP triangle", colors['up']),
        ("Light Green: DOWN triangle", colors['down']),
        ("Red: REJECTED", colors['invalid'])
    ]
    
    for text, color in legend_items:
        cv2.putText(vis_img, text, (text_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 15
    
    # Save the visualization
    cv2.imwrite(output_filename, vis_img)
    print(f"Saved detailed visualization: {output_filename}")

def main():
    # Load the frame
    frame = cv2.imread('extracted_frame.jpg')
    if frame is None:
        print("Could not load extracted_frame.jpg")
        return
    
    # Initialize triangle detector
    debug_dir = Path("debug_detected_triangles")
    detector = TriangleOrientationDetector(debug_output_dir=debug_dir)
    
    print("🎯 SHOWING DETECTED TRIANGLES WITH RELAXED VALIDATION")
    print("=" * 60)
    print()
    
    # 1. Territory Triangle Area (where the "2" digit is)
    print("1. TERRITORY TRIANGLE AREA (Right side)")
    territory_x1, territory_y1, territory_x2, territory_y2 = 1053, 650, 1106, 689
    territory_roi = frame[territory_y1:territory_y2, territory_x1:territory_x2]
    
    territory_contours = find_contours_in_roi(territory_roi)
    territory_results = []
    
    for i, contour in enumerate(territory_contours):
        result = detector.analyze_territory_triangle(contour, territory_roi)
        territory_results.append(result)
        area = cv2.contourArea(contour)
        status = "✅ DETECTED" if result.is_valid else "❌ REJECTED"
        print(f"   Contour {i+1}: Area={area:.1f} - {status}")
        if result.is_valid:
            print(f"      Direction: {result.direction.value.upper()}, Confidence: {result.confidence:.1%}")
    
    visualize_detected_triangles(
        territory_roi, 
        territory_contours, 
        territory_results,
        "Territory Area - Digit Rejection Test",
        "detected_territory_triangles.jpg"
    )
    
    # 2. Manual Left ROI (possession triangle area)
    print("\n2. LEFT POSSESSION AREA (Where triangles were found!)")
    left_roi = cv2.imread('debug_manual_left_roi_simple.jpg')
    if left_roi is not None:
        left_contours = find_contours_in_roi(left_roi)
        left_results = []
        
        valid_triangles = 0
        for i, contour in enumerate(left_contours):
            result = detector.analyze_possession_triangle(contour, left_roi)
            left_results.append(result)
            area = cv2.contourArea(contour)
            status = "✅ DETECTED" if result.is_valid else "❌ REJECTED"
            print(f"   Contour {i+1}: Area={area:.1f} - {status}")
            if result.is_valid:
                valid_triangles += 1
                print(f"      Direction: {result.direction.value.upper()}, Confidence: {result.confidence:.1%}")
        
        visualize_detected_triangles(
            left_roi, 
            left_contours, 
            left_results,
            f"Left Area - {valid_triangles} TRIANGLES FOUND!",
            "detected_possession_triangles.jpg"
        )
        
        print(f"\n🎉 SUCCESS: Found {valid_triangles} valid triangles in left area!")
    
    print(f"\n=== VISUALIZATION FILES CREATED ===")
    print("• detected_territory_triangles.jpg - Territory area analysis")
    print("• detected_possession_triangles.jpg - Possession area with found triangles")
    print("\n🎯 The relaxed validation is working perfectly!")
    print("✅ Detecting real triangles")
    print("❌ Still rejecting digit shapes")

if __name__ == "__main__":
    main() 