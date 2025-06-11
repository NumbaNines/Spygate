#!/usr/bin/env python3
"""
Fix dataset configuration and retrain YOLOv8 model with correct classes.
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def fix_dataset_config():
    """Fix the dataset configuration with correct classes."""
    print("🔧 Fixing dataset configuration...")
    
    # Read the correct classes from classes.txt
    classes_file = Path("training_data/classes.txt")
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Found {len(class_names)} classes: {class_names}")
    else:
        print("❌ classes.txt not found!")
        return None
    
    # Create correct configuration
    config = {
        'path': str(Path('training_data').absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save corrected config
    config_path = Path("training_data/corrected_dataset.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Corrected config saved: {config_path}")
    print(f"   Classes: {len(class_names)} - {class_names}")
    
    return config_path

def quick_retrain():
    """Retrain with corrected configuration."""
    print("\n🚀 Starting corrected training...")
    
    # Fix config first
    config_path = fix_dataset_config()
    if not config_path:
        return
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Fresh start
    
    # Quick training parameters for testing
    training_args = {
        'data': str(config_path),
        'epochs': 20,  # Fewer epochs for quick test
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'workers': 2,
        'patience': 10,
        'save_period': 5,
        'val': True,
        'plots': True,
        'verbose': True,
        'project': 'runs/train',
        'name': f'hud_corrected_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        
        # Minimal augmentation for perfect data
        'degrees': 0.0,
        'translate': 0.05,
        'scale': 0.05,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.3,
        'mixup': 0.0,
    }
    
    print("📋 Corrected training parameters:")
    for key, value in training_args.items():
        print(f"   {key}: {value}")
    
    print("\n🏈 Starting corrected training...")
    
    # Start training
    results = model.train(**training_args)
    
    print(f"\n✅ Corrected training completed!")
    print(f"📁 Results: {results.save_dir}")
    
    return results

def test_corrected_model(results):
    """Test the corrected model."""
    print("\n🧪 Testing corrected model...")
    
    # Load the best model
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    print(f"📦 Model classes: {list(model.names.values())}")
    
    # Test on a training image
    test_image = "training_data/train/images/monitor3_screenshot_20250608_021042_6_copy_0001.png"
    if Path(test_image).exists():
        print(f"🖼️  Testing: {Path(test_image).name}")
        
        results_test = model(test_image, conf=0.1, verbose=False)
        if results_test and len(results_test) > 0:
            result = results_test[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"   ✅ Found {len(boxes)} detections:")
                for j, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    confidence = float(conf)
                    print(f"      {j+1}. {class_name}: {confidence:.3f}")
            else:
                print("   ⚠️  No detections")
        else:
            print("   ❌ No results")
    
    return best_model_path

def main():
    """Main function."""
    print("🏈 YOLOv8 HUD Detection - Configuration Fix & Retrain")
    print("=" * 60)
    
    try:
        # Quick retrain with correct config
        results = quick_retrain()
        
        # Test the corrected model
        model_path = test_corrected_model(results)
        
        print(f"\n🎉 Success! Corrected model ready:")
        print(f"📁 {model_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 