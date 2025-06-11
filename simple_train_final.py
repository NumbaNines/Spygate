#!/usr/bin/env python3
"""
Simple final training script using existing train/val data.
"""

import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def verify_data():
    """Verify that we have matching images and labels."""
    print("🔍 Verifying training data...")
    
    train_img_dir = Path("training_data/train/images")
    train_lbl_dir = Path("training_data/train/labels")
    val_img_dir = Path("training_data/val/images")
    val_lbl_dir = Path("training_data/val/labels")
    
    train_imgs = len(list(train_img_dir.glob("*.png")))
    train_lbls = len(list(train_lbl_dir.glob("*.txt")))
    val_imgs = len(list(val_img_dir.glob("*.png")))
    val_lbls = len(list(val_lbl_dir.glob("*.txt")))
    
    print(f"📊 Data counts:")
    print(f"   Training: {train_imgs} images, {train_lbls} labels")
    print(f"   Validation: {val_imgs} images, {val_lbls} labels")
    
    if train_imgs == train_lbls and val_imgs == val_lbls:
        print("   ✅ Perfect match! Ready to train.")
        return True
    else:
        print("   ❌ Mismatch found!")
        return False

def create_config():
    """Create dataset configuration."""
    print("\n⚙️ Creating dataset configuration...")
    
    # Read classes
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
    
    config_path = Path("training_data/simple_dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Config saved: {config_path}")
    print(f"   Classes ({len(class_names)}): {class_names}")
    
    return config_path

def train_model():
    """Train the YOLOv8 model."""
    print("\n🚀 Starting YOLOv8 training...")
    
    # Create config
    config_path = create_config()
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    training_args = {
        'data': str(config_path),
        'epochs': 30,
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'workers': 2,
        'patience': 15,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        'project': 'runs/train',
        'name': f'hud_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        
        # Minimal augmentation for perfect data
        'degrees': 0.0,
        'translate': 0.01,
        'scale': 0.01,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.1,
        'mixup': 0.0,
    }
    
    print("📋 Training parameters:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print("\n🏈 Starting training...")
    
    # Start training
    results = model.train(**training_args)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results: {results.save_dir}")
    
    return results

def test_model(results):
    """Test the trained model."""
    print("\n🧪 Testing trained model...")
    
    # Load best model
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    print(f"📦 Model classes: {list(model.names.values())}")
    
    # Test on training images
    test_images = [
        "training_data/train/images/monitor3_screenshot_20250608_021042_6_copy_0001.png",
        "training_data/train/images/monitor3_screenshot_20250608_021427_50_copy_0001.png",
        "training_data/train/images/monitor3_screenshot_20250608_021044_7_copy_0001.png"
    ]
    
    print(f"\n🖼️  Testing on training examples:")
    for i, img_path in enumerate(test_images, 1):
        if not Path(img_path).exists():
            continue
            
        print(f"\n   Test {i}: {Path(img_path).name}")
        results_test = model(img_path, conf=0.2, verbose=False)
        
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
    
    return best_model_path

def main():
    """Main function."""
    print("🏈 Simple YOLOv8 HUD Training")
    print("=" * 40)
    
    # Verify data
    if not verify_data():
        print("\n❌ Data verification failed!")
        return
    
    try:
        # Train model
        results = train_model()
        
        # Test model
        model_path = test_model(results)
        
        print(f"\n🎉 SUCCESS! Model trained:")
        print(f"📁 {model_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 