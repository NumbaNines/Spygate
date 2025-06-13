#!/usr/bin/env python3
"""
Comprehensive Triangle Detection Test
Tests 25 random images from 6.12 screenshots using our enhanced detection system
with YOLO + template matching pipeline.
"""

import os
import sys
import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from src.spygate.core.hardware import HardwareDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TriangleDetectionTester:
    """Comprehensive triangle detection tester using our enhanced pipeline."""
    
    def __init__(self):
        """Initialize the tester with our enhanced detection system."""
        
        # Initialize hardware detection
        self.hardware = HardwareDetector()
        
        # Initialize our enhanced game analyzer
        self.analyzer = EnhancedGameAnalyzer(hardware=self.hardware)
        
        # Screenshot directory
        self.screenshots_dir = Path("6.12 screenshots")
        
        # Results directory
        self.results_dir = Path("triangle_detection_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.test_results = []
        
        print("🎯 SPYGATE TRIANGLE DETECTION TEST")
        print("=" * 60)
        print(f"🔧 Hardware Tier: {self.hardware.detect_tier().name}")
        print(f"📁 Screenshots Directory: {self.screenshots_dir}")
        print(f"📊 Results Directory: {self.results_dir}")
        print(f"🎲 Testing 25 random images with our enhanced detection system")
        print()

    def get_random_images(self, count: int = 25) -> List[Path]:
        """Get random images from the screenshots directory."""
        
        # Get all PNG files
        all_images = list(self.screenshots_dir.glob("*.png"))
        
        if len(all_images) < count:
            logger.warning(f"Only {len(all_images)} images available, using all of them")
            return all_images
        
        # Randomly select images
        selected = random.sample(all_images, count)
        
        print(f"🎲 Selected {len(selected)} random images:")
        for i, img_path in enumerate(selected, 1):
            print(f"   {i:2d}. {img_path.name}")
        print()
        
        return selected

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze a single image using our enhanced detection system."""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Could not load image: {image_path}"}
        
        print(f"🔍 Analyzing: {image_path.name}")
        print(f"   📐 Size: {image.shape[1]}x{image.shape[0]}")
        
        # Analyze frame using our enhanced analyzer
        try:
            game_state = self.analyzer.analyze_frame(image)
            
            # Get raw detection info
            detections = self.analyzer.model.predict(image, verbose=False)
            
            # Process detections for detailed results
            detection_details = self._process_detections(detections[0] if detections else None, image)
            
            result = {
                "image_name": image_path.name,
                "image_size": f"{image.shape[1]}x{image.shape[0]}",
                "game_state": {
                    "possession_team": game_state.possession_team if game_state else None,
                    "territory": game_state.territory if game_state else None,
                    "confidence": game_state.confidence if game_state else 0.0
                },
                "detections": detection_details,
                "status": "success"
            }
            
            # Create visualization
            vis_image = self._create_visualization(image, detection_details)
            vis_path = self.results_dir / f"vis_{image_path.name}"
            cv2.imwrite(str(vis_path), vis_image)
            
            print(f"   ✅ Analysis complete")
            print(f"   📊 Detections: {len(detection_details['detected_classes'])}")
            print(f"   🎯 Confidence: {result['game_state']['confidence']:.3f}")
            
            return result
            
        except Exception as e:
            error_result = {
                "image_name": image_path.name,
                "error": str(e),
                "status": "error"
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result

    def _process_detections(self, detection_result, image: np.ndarray) -> Dict[str, Any]:
        """Process YOLO detection results into detailed format."""
        
        details = {
            "detected_classes": [],
            "bounding_boxes": [],
            "confidences": [],
            "triangle_analysis": {
                "possession_triangles": [],
                "territory_triangles": []
            }
        }
        
        if detection_result is None or detection_result.boxes is None:
            return details
        
        # Process each detection
        boxes = detection_result.boxes
        for i in range(len(boxes)):
            # Get box info
            box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            # Get class name
            class_name = self.analyzer.ui_classes[class_id] if class_id < len(self.analyzer.ui_classes) else f"class_{class_id}"
            
            details["detected_classes"].append(class_name)
            details["bounding_boxes"].append([int(x) for x in box])
            details["confidences"].append(conf)
            
            # Special processing for triangle areas
            if class_name in ["possession_triangle_area", "territory_triangle_area"]:
                self._analyze_triangle_region(image, box, class_name, conf, details)
        
        return details

    def _analyze_triangle_region(self, image: np.ndarray, box: np.ndarray, 
                                class_name: str, confidence: float, details: Dict[str, Any]):
        """Analyze triangle regions in detail."""
        
        x1, y1, x2, y2 = [int(x) for x in box]
        
        # Extract region
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return
        
        # Basic triangle analysis
        triangle_info = {
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "region_size": f"{x2-x1}x{y2-y1}",
            "area": (x2-x1) * (y2-y1)
        }
        
        # Add to appropriate category
        if class_name == "possession_triangle_area":
            details["triangle_analysis"]["possession_triangles"].append(triangle_info)
        elif class_name == "territory_triangle_area":
            details["triangle_analysis"]["territory_triangles"].append(triangle_info)

    def _create_visualization(self, image: np.ndarray, detection_details: Dict[str, Any]) -> np.ndarray:
        """Create visualization of detections."""
        
        vis_image = image.copy()
        
        # Draw bounding boxes
        for i, (class_name, bbox, conf) in enumerate(zip(
            detection_details["detected_classes"],
            detection_details["bounding_boxes"], 
            detection_details["confidences"]
        )):
            x1, y1, x2, y2 = bbox
            
            # Get color for class
            color = self.analyzer.colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {conf:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image

    def run_test(self, count: int = 25):
        """Run the complete test on random images."""
        
        print("🚀 Starting comprehensive triangle detection test...")
        print()
        
        # Get random images
        images = self.get_random_images(count)
        
        # Test each image
        for i, image_path in enumerate(images, 1):
            print(f"[{i:2d}/{len(images)}] " + "="*50)
            
            result = self.analyze_image(image_path)
            self.test_results.append(result)
            
            print()
        
        # Generate summary report
        self._generate_summary_report()
        
        print("✅ Test completed!")
        print(f"📊 Results saved to: {self.results_dir}")

    def _generate_summary_report(self):
        """Generate a comprehensive summary report."""
        
        print("📊 GENERATING SUMMARY REPORT")
        print("=" * 60)
        
        # Count successes and errors
        successful = [r for r in self.test_results if r.get("status") == "success"]
        errors = [r for r in self.test_results if r.get("status") == "error"]
        
        print(f"🎯 Total Images Tested: {len(self.test_results)}")
        print(f"✅ Successful Analyses: {len(successful)}")
        print(f"❌ Errors: {len(errors)}")
        print()
        
        if successful:
            # Analyze detection patterns
            class_counts = {}
            confidence_scores = []
            
            for result in successful:
                # Count detected classes
                for class_name in result["detections"]["detected_classes"]:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Collect confidence scores
                confidence_scores.extend(result["detections"]["confidences"])
            
            print("🔍 DETECTION STATISTICS:")
            print(f"   📊 Total Detections: {sum(class_counts.values())}")
            print(f"   🎯 Average Confidence: {np.mean(confidence_scores):.3f}")
            print(f"   📈 Max Confidence: {np.max(confidence_scores):.3f}")
            print(f"   📉 Min Confidence: {np.min(confidence_scores):.3f}")
            print()
            
            print("🏷️  CLASS DETECTION COUNTS:")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(successful)) * 100
                print(f"   {class_name:25s}: {count:3d} ({percentage:5.1f}%)")
            print()
            
            # Triangle-specific analysis
            possession_count = 0
            territory_count = 0
            
            for result in successful:
                possession_count += len(result["detections"]["triangle_analysis"]["possession_triangles"])
                territory_count += len(result["detections"]["triangle_analysis"]["territory_triangles"])
            
            print("🔺 TRIANGLE ANALYSIS:")
            print(f"   ◄► Possession Triangles: {possession_count}")
            print(f"   ▲▼ Territory Triangles: {territory_count}")
            print()
        
        # Save detailed results
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Create summary file
        summary = {
            "test_summary": {
                "total_images": len(self.test_results),
                "successful": len(successful),
                "errors": len(errors),
                "hardware_tier": self.hardware.detect_tier().name
            },
            "detection_stats": {
                "class_counts": class_counts if successful else {},
                "avg_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                "triangle_counts": {
                    "possession": possession_count if successful else 0,
                    "territory": territory_count if successful else 0
                }
            }
        }
        
        summary_file = self.results_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Detailed results: {results_file}")
        print(f"📋 Summary report: {summary_file}")

def main():
    """Main execution function."""
    
    # Create tester instance
    tester = TriangleDetectionTester()
    
    # Run the test
    tester.run_test(25)

if __name__ == "__main__":
    main() 