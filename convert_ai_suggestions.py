#!/usr/bin/env python3
"""Convert AI annotation suggestions to training labels based on confidence thresholds."""

import json
import argparse
from pathlib import Path
import shutil

class AISuggestionConverter:
    """Convert AI suggestions to training labels."""
    
    def __init__(self, confidence_thresholds=None):
        """Initialize with confidence thresholds for each class."""
        self.confidence_thresholds = confidence_thresholds or {
            "hud": 0.3,           # HUD is well-trained, use higher threshold
            "qb_position": 0.15,  # QB position needs more data, lower threshold
            "left_hash_mark": 0.15,
            "right_hash_mark": 0.15,
            "preplay": 0.1,       # New classes, very low threshold
            "playcall": 0.1,
            "possession_indicator": 0.15,
            "territory_indicator": 0.15,
        }
    
    def convert_suggestions(self, suggestions_dir: Path, output_dir: Path, 
                          min_detections: int = 1):
        """Convert AI suggestions to training labels."""
        suggestions_dir = Path(suggestions_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🤖 Converting AI suggestions from {suggestions_dir}")
        print(f"📂 Output directory: {output_dir}")
        print(f"🎯 Confidence thresholds: {self.confidence_thresholds}")
        
        converted_count = 0
        total_detections = 0
        class_counts = {}
        
        for suggestion_file in suggestions_dir.glob("*.json"):
            try:
                with open(suggestion_file, 'r') as f:
                    data = json.load(f)
                
                # Filter shapes by confidence threshold
                filtered_shapes = []
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    description = shape.get("description", "")
                    
                    # Extract confidence from description
                    if "conf:" in description:
                        conf_str = description.split("conf:")[1].strip().rstrip(")")
                        try:
                            confidence = float(conf_str)
                        except ValueError:
                            continue
                    else:
                        continue
                    
                    # Check if confidence meets threshold
                    threshold = self.confidence_thresholds.get(label, 0.2)
                    if confidence >= threshold:
                        filtered_shapes.append(shape)
                        class_counts[label] = class_counts.get(label, 0) + 1
                        total_detections += 1
                
                # Only save if we have enough detections
                if len(filtered_shapes) >= min_detections:
                    data["shapes"] = filtered_shapes
                    
                    output_file = output_dir / suggestion_file.name
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    converted_count += 1
                    print(f"  ✅ {suggestion_file.stem}: {len(filtered_shapes)} detections")
                
            except Exception as e:
                print(f"  ❌ Error processing {suggestion_file.name}: {e}")
        
        print(f"\n🎯 CONVERSION COMPLETE!")
        print(f"📝 Converted {converted_count} files")
        print(f"🔍 Total detections: {total_detections}")
        print(f"\n📊 Detections by class:")
        for class_name, count in sorted(class_counts.items()):
            print(f"   {class_name}: {count}")
        
        return converted_count, class_counts
    
    def copy_corresponding_images(self, suggestions_dir: Path, images_dir: Path, 
                                output_images_dir: Path):
        """Copy images that have corresponding AI suggestions."""
        suggestions_dir = Path(suggestions_dir)
        images_dir = Path(images_dir)
        output_images_dir = Path(output_images_dir)
        output_images_dir.mkdir(exist_ok=True)
        
        copied_count = 0
        
        for suggestion_file in suggestions_dir.glob("*.json"):
            # Find corresponding image
            image_name = suggestion_file.stem  # Remove .json extension
            
            # Try common image extensions
            for ext in [".png", ".jpg", ".jpeg"]:
                image_path = images_dir / (image_name + ext)
                if image_path.exists():
                    output_path = output_images_dir / (image_name + ext)
                    shutil.copy2(image_path, output_path)
                    copied_count += 1
                    break
        
        print(f"📷 Copied {copied_count} corresponding images")
        return copied_count


def main():
    parser = argparse.ArgumentParser(description="Convert AI annotation suggestions to training labels")
    parser.add_argument("--suggestions", default="ai_suggestions", 
                       help="Path to AI suggestions directory")
    parser.add_argument("--output", default="converted_labels", 
                       help="Output directory for converted labels")
    parser.add_argument("--images", default="training_data/images", 
                       help="Path to source images")
    parser.add_argument("--output-images", default="converted_images", 
                       help="Output directory for corresponding images")
    parser.add_argument("--min-detections", type=int, default=1,
                       help="Minimum detections required to save file")
    parser.add_argument("--hud-conf", type=float, default=0.3,
                       help="Confidence threshold for HUD class")
    parser.add_argument("--other-conf", type=float, default=0.15,
                       help="Confidence threshold for other classes")
    
    args = parser.parse_args()
    
    # Custom confidence thresholds
    confidence_thresholds = {
        "hud": args.hud_conf,
        "qb_position": args.other_conf,
        "left_hash_mark": args.other_conf,
        "right_hash_mark": args.other_conf,
        "preplay": 0.1,
        "playcall": 0.1,
        "possession_indicator": args.other_conf,
        "territory_indicator": args.other_conf,
    }
    
    converter = AISuggestionConverter(confidence_thresholds)
    
    # Convert suggestions
    converted_count, class_counts = converter.convert_suggestions(
        Path(args.suggestions), 
        Path(args.output),
        args.min_detections
    )
    
    # Copy corresponding images
    if converted_count > 0:
        converter.copy_corresponding_images(
            Path(args.suggestions),
            Path(args.images),
            Path(args.output_images)
        )
    
    print(f"\n🎉 Ready for training! Use:")
    print(f"   Labels: {args.output}")
    print(f"   Images: {args.output_images}")


if __name__ == "__main__":
    main() 