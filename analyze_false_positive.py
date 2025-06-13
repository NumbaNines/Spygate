import cv2
import numpy as np
from pathlib import Path
from src.spygate.ml.triangle_orientation_detector import TriangleOrientationDetector

def analyze_false_positive():
    """Analyze why the '4' is being detected as a valid triangle."""
    
    print("=== ANALYZING FALSE POSITIVE: '4' DETECTED AS TRIANGLE ===")
    
    # Load the territory ROI (which contains the "4")
    print("Loading territory ROI...")
    territory_roi = cv2.imread('debug_output/territory_triangle_area_roi.png')
    
    if territory_roi is None:
        print("Could not load territory ROI")
        return
    
    print(f"Territory ROI shape: {territory_roi.shape}")
    
    # Convert to grayscale and find contours
    print("Converting to grayscale and finding contours...")
    gray = cv2.cvtColor(territory_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")
    
    # Analyze each contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        print(f"\nContour {i+1}:")
        print(f"  Area: {area}")
        
        if area > 50:  # Only analyze significant contours
            print(f"  Analyzing significant contour...")
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h != 0 else 0
            print(f"  Bounding rect: x={x}, y={y}, w={w}, h={h}")
            print(f"  Aspect ratio: {aspect:.3f}")
            
            # Check if this passes our triangle detector's basic validation
            detector = TriangleOrientationDetector()
            
            # Manually check the validation criteria
            print(f"  Validation checks:")
            print(f"    Points count: {len(contour)} (need >= 3)")
            print(f"    Area: {area} (need {detector.MIN_AREA}-{detector.MAX_AREA})")
            print(f"    Aspect: {aspect:.3f} (territory needs {detector.TERRITORY_MIN_ASPECT}-{detector.TERRITORY_MAX_ASPECT})")
            
            # Check if it's in the valid range for territory triangles
            area_valid = detector.MIN_AREA <= area <= detector.MAX_AREA
            aspect_valid = detector.TERRITORY_MIN_ASPECT <= aspect <= detector.TERRITORY_MAX_ASPECT
            points_valid = len(contour) >= 3
            
            print(f"    Area valid: {area_valid}")
            print(f"    Aspect valid: {aspect_valid}")
            print(f"    Points valid: {points_valid}")
            
            if area_valid and aspect_valid and points_valid:
                print(f"  *** This contour would PASS basic validation ***")
                
                # Let's see what our detector thinks
                print(f"  Running triangle detector...")
                result = detector.analyze_territory_triangle(contour, territory_roi)
                print(f"  Detector result:")
                print(f"    Valid: {result.is_valid}")
                print(f"    Direction: {result.direction}")
                print(f"    Confidence: {result.confidence:.3f}")
                
                # Create a visualization of this specific contour
                contour_vis = territory_roi.copy()
                cv2.drawContours(contour_vis, [contour], -1, (0, 255, 0), 2)
                
                # Mark the key points that the detector found
                if result.is_valid:
                    print(f"  Finding key points...")
                    # Find topmost and bottommost points
                    topmost = tuple(contour[contour[:,:,1].argmin()][0])
                    bottommost = tuple(contour[contour[:,:,1].argmax()][0])
                    
                    # Find base point (furthest from top-bottom line)
                    top_bottom_line = np.array([topmost, bottommost])
                    distances = []
                    for point in contour:
                        point = point[0]
                        dist = np.abs(np.cross(top_bottom_line[1] - top_bottom_line[0], point - top_bottom_line[0])) / np.linalg.norm(top_bottom_line[1] - top_bottom_line[0])
                        distances.append((dist, point))
                    base_point = max(distances, key=lambda x: x[0])[1]
                    
                    # Draw the key points
                    cv2.circle(contour_vis, topmost, 3, (255, 0, 0), -1)  # Blue for top
                    cv2.circle(contour_vis, bottommost, 3, (0, 0, 255), -1)  # Red for bottom
                    cv2.circle(contour_vis, tuple(base_point), 3, (255, 255, 0), -1)  # Cyan for base
                    cv2.circle(contour_vis, result.center, 3, (255, 0, 255), -1)  # Magenta for center
                    
                    print(f"    Key points:")
                    print(f"      Top: {topmost}")
                    print(f"      Bottom: {bottommost}")
                    print(f"      Base: {tuple(base_point)}")
                    print(f"      Center: {result.center}")
                
                # Save the visualization
                filename = f"debug_output/false_positive_contour_{i+1}.png"
                cv2.imwrite(filename, contour_vis)
                print(f"  Saved visualization: {filename}")
            else:
                print(f"  This contour would FAIL basic validation")
        else:
            print(f"  Skipping small contour (area: {area})")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("The issue is that the '4' digit has a shape that passes our")
    print("triangle validation criteria. We need to improve our validation")
    print("to better distinguish between actual triangles and digit shapes.")

if __name__ == "__main__":
    print("Starting false positive analysis...")
    analyze_false_positive()
    print("Analysis complete.") 