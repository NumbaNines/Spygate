import cv2
import numpy as np
from pathlib import Path
from src.spygate.ml.triangle_orientation_detector import TriangleOrientationDetector

def test_orientation_detection():
    """Test our triangle orientation detector on the detected regions."""
    
    print("=== TESTING TRIANGLE ORIENTATION DETECTION ===")
    
    # Initialize the triangle detector
    debug_dir = Path("debug_output")
    print(f"Debug directory: {debug_dir}")
    detector = TriangleOrientationDetector(debug_output_dir=debug_dir)
    print("Triangle detector initialized")
    
    # Load the ROI images
    possession_roi = cv2.imread('debug_output/possession_triangle_area_roi.png')
    territory_roi = cv2.imread('debug_output/territory_triangle_area_roi.png')
    
    print(f"Possession ROI loaded: {possession_roi is not None}")
    print(f"Territory ROI loaded: {territory_roi is not None}")
    
    if possession_roi is not None:
        print(f"\n=== POSSESSION TRIANGLE ANALYSIS ===")
        print(f"ROI Shape: {possession_roi.shape}")
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(possession_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        if contours:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"Largest contour area: {area}")
            
            # Test our triangle orientation detector
            try:
                print("Calling analyze_possession_triangle...")
                result = detector.analyze_possession_triangle(largest_contour, possession_roi)
                print(f"Triangle Detection Result:")
                print(f"  Valid: {result.is_valid}")
                print(f"  Direction: {result.direction}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Center: {result.center}")
                print(f"  Triangle Type: {result.triangle_type}")
                    
            except Exception as e:
                print(f"Error analyzing possession triangle: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No contours found in possession triangle ROI")
    
    if territory_roi is not None:
        print(f"\n=== TERRITORY TRIANGLE ANALYSIS ===")
        print(f"ROI Shape: {territory_roi.shape}")
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(territory_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        if contours:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"Largest contour area: {area}")
            
            # Test our triangle orientation detector
            try:
                print("Calling analyze_territory_triangle...")
                result = detector.analyze_territory_triangle(largest_contour, territory_roi)
                print(f"Triangle Detection Result:")
                print(f"  Valid: {result.is_valid}")
                print(f"  Direction: {result.direction}")
                print(f"  Confidence: {result.confidence:.3f}")
                print(f"  Center: {result.center}")
                print(f"  Triangle Type: {result.triangle_type}")
                    
            except Exception as e:
                print(f"Error analyzing territory triangle: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No contours found in territory triangle ROI")
    
    print(f"\n=== SUMMARY ===")
    print("This shows how our triangle orientation detector performs on the")
    print("regions that YOLO detected as 'triangle areas'. If YOLO detected")
    print("a '4' as a triangle, our orientation detector should reject it")
    print("as invalid based on shape analysis.")

if __name__ == "__main__":
    print("Starting test...")
    test_orientation_detection()
    print("Test complete.") 