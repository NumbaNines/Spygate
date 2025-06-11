#!/usr/bin/env python3
"""
Fix the missing labels issue and retrain properly.
"""

import shutil
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def fix_missing_labels():
    """Copy the missing labels to train and val directories."""
    print("🔧 Fixing missing labels...")
    
    # Check train images and copy corresponding labels
    train_img_dir = Path("training_data/train/images")
    train_lbl_dir = Path("training_data/train/labels")
    val_img_dir = Path("training_data/val/images")
    val_lbl_dir = Path("training_data/val/labels")
    
    # Ensure label directories exist
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy training labels
    train_images = list(train_img_dir.glob("*.png"))
    print(f"📸 Found {len(train_images)} training images")
    
    labels_copied = 0
    for img_file in train_images:
        # Find corresponding label in main labels directory
        label_name = img_file.stem + ".txt"
        source_label = Path("training_data/labels") / label_name
        target_label = train_lbl_dir / label_name
        
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            labels_copied += 1
        else:
            print(f"   ⚠️  Missing label: {label_name}")
    
    print(f"✅ Copied {labels_copied} training labels")
    
    # Copy validation labels
    val_images = list(val_img_dir.glob("*.png"))
    print(f"📸 Found {len(val_images)} validation images")
    
    val_labels_copied = 0
    for img_file in val_images:
        label_name = img_file.stem + ".txt"
        source_label = Path("training_data/labels") / label_name
        target_label = val_lbl_dir / label_name
        
        if source_label.exists():
            shutil.copy2(source_label, target_label)
            val_labels_copied += 1
        else:
            print(f"   ⚠️  Missing validation label: {label_name}")
    
    print(f"✅ Copied {val_labels_copied} validation labels")
    
    return labels_copied, val_labels_copied

def create_final_config():
    """Create the final corrected dataset configuration."""
    print("\n⚙️ Creating final dataset configuration...")
    
    # Read correct classes
    classes_file = Path("training_data/classes.txt")
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    config = {
        'path': str(Path('training_data').absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    config_path = Path("training_data/final_dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Final config: {config_path}")
    print(f"   Classes ({len(class_names)}): {class_names}")
    
    return config_path

def final_training():
    """Run the final training with all fixes applied."""
    print("\n🚀 Starting FINAL training with fixed labels...")
    
    # Get the config
    config_path = create_final_config()
    
    # Initialize fresh model
    model = YOLO('yolov8n.pt')
    
    # Final training parameters
    training_args = {
        'data': str(config_path),
        'epochs': 30,  # More epochs since we have good data now
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'workers': 2,
        'patience': 15,
        'save_period': 5,
        'val': True,
        'plots': True,
        'verbose': True,
        'project': 'runs/train',
        'name': f'hud_FINAL_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        
        # Minimal augmentation for perfect training data
        'degrees': 0.0,
        'translate': 0.02,  # Very minimal
        'scale': 0.02,      # Very minimal
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.2,      # Reduced mosaic
        'mixup': 0.0,
        
        # Loss weights for HUD detection
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    print("📋 Final training parameters:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print("\n🏈 Starting FINAL training run...")
    
    # Start training
    results = model.train(**training_args)
    
    print(f"\n✅ FINAL training completed!")
    print(f"📁 Results: {results.save_dir}")
    
    return results

def test_final_model(results):
    """Test the final trained model."""
    print("\n🧪 Testing FINAL model...")
    
    # Load best model
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    print(f"📦 Model classes: {list(model.names.values())}")
    
    # Test on perfect training images
    test_images = [
        "training_data/train/images/monitor3_screenshot_20250608_021042_6_copy_0001.png",
        "training_data/train/images/monitor3_screenshot_20250608_021427_50_copy_0001.png",
        "training_data/train/images/monitor3_screenshot_20250608_021044_7_copy_0001.png"
    ]
    
    print(f"\n🖼️  Testing on perfect training examples:")
    for i, img_path in enumerate(test_images, 1):
        if not Path(img_path).exists():
            continue
            
        print(f"\n   Test {i}: {Path(img_path).name}")
        results_test = model(img_path, conf=0.3, verbose=False)
        
        if results_test and len(results_test) > 0:
            result = results_test[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"      ✅ {len(boxes)} detections:")
                for j, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    confidence = float(conf)
                    print(f"         {j+1}. {class_name}: {confidence:.3f}")
            else:
                print("      ⚠️  No detections")
    
    # Test on validation image
    val_images = list(Path("training_data/val/images").glob("*.png"))
    if val_images:
        test_val = val_images[0]
        print(f"\n🔍 Testing on validation: {test_val.name}")
        results_val = model(str(test_val), conf=0.3, verbose=False)
        
        if results_val and len(results_val) > 0:
            result = results_val[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   ✅ {len(boxes)} detections on validation:")
                for j, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    confidence = float(conf)
                    print(f"      {j+1}. {class_name}: {confidence:.3f}")
            else:
                print("   ⚠️  No detections on validation")
    
    return best_model_path

def main():
    """Main function to fix everything and retrain."""
    print("🏈 FINAL YOLOv8 HUD Training - Fix Labels & Retrain")
    print("=" * 60)
    
    try:
        # Fix missing labels
        train_labels, val_labels = fix_missing_labels()
        
        if train_labels == 0:
            print("❌ No training labels found! Check label files.")
            return
        
        print(f"\n✅ Labels fixed: {train_labels} train, {val_labels} val")
        
        # Run final training
        results = final_training()
        
        # Test final model
        model_path = test_final_model(results)
        
        print(f"\n🎉 SUCCESS! Final model trained and ready:")
        print(f"📁 {model_path}")
        print(f"\n🚀 This model should now work perfectly on your HUD images!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 