#!/usr/bin/env python3
"""AI-Assisted Annotation Tool for SpygateAI - Pre-label images using your trained model."""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse

class AIAssistedAnnotator:
    """Use your trained YOLO model to suggest annotations."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.15):
        """Initialize with your trained model."""
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.class_names = [
            "hud", "qb_position", "left_hash_mark", "right_hash_mark", 
            "preplay", "playcall", "possession_indicator", "territory_indicator"
        ]
    
    def suggest_annotations(self, image_path: str) -> dict:
        """Generate annotation suggestions for an image."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Run detection
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        # Convert to LabelMe format
        suggestions = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": Path(image_path).name,
            "imageData": None,
            "imageHeight": image.shape[0],
            "imageWidth": image.shape[1]
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    if cls_id < len(self.class_names):
                        shape = {
                            "label": self.class_names[cls_id],
                            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                            "group_id": None,
                            "shape_type": "rectangle",
                            "flags": {},
                            "description": f"AI suggested (conf: {conf:.2f})"
                        }
                        suggestions["shapes"].append(shape)
        
        return suggestions
    
    def process_folder(self, images_folder: str, output_folder: str):
        """Process all images in a folder and generate suggestions."""
        images_path = Path(images_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        print(f"🤖 Processing {len(image_files)} images with AI assistance...")
        
        for image_file in image_files:
            print(f"📷 Processing: {image_file.name}")
            
            suggestions = self.suggest_annotations(str(image_file))
            if suggestions:
                output_file = output_path / f"{image_file.stem}.json"
                
                # Don't overwrite existing annotations, rename as suggestions
                if output_file.exists():
                    output_file = output_path / f"{image_file.stem}_ai_suggestions.json"
                
                with open(output_file, 'w') as f:
                    json.dump(suggestions, f, indent=2)
                
                detection_count = len(suggestions["shapes"])
                print(f"  ✅ Generated {detection_count} suggestions -> {output_file.name}")
        
        print(f"🎯 AI annotation complete! Check {output_folder} for suggestions.")

def main():
    parser = argparse.ArgumentParser(description="AI-Assisted Annotation Tool")
    parser.add_argument("--model", default="runs/detect/spygate_hud_detection_fast2/weights/best.pt", 
                       help="Path to trained model")
    parser.add_argument("--images", default="training_data/images", 
                       help="Path to images folder")
    parser.add_argument("--output", default="ai_suggestions", 
                       help="Output folder for suggestions")
    parser.add_argument("--conf", type=float, default=0.15, 
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    annotator = AIAssistedAnnotator(args.model, args.conf)
    annotator.process_folder(args.images, args.output)

if __name__ == "__main__":
    main() 