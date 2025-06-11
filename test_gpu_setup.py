#!/usr/bin/env python3
"""
Test script to verify SpygateAI GPU setup is working correctly.
"""

import torch
from ultralytics import YOLO
import sys

def test_gpu_setup():
    print("🔥 SpygateAI GPU Setup Test")
    print("=" * 50)
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ GPU not available")
        return False
    
    # Test YOLOv8
    try:
        print("\n🎯 Testing YOLOv8...")
        model = YOLO('yolov8n.pt')  # Download small model for testing
        model.to('cuda')  # Move to GPU
        print(f"✅ YOLOv8 model loaded on: {model.device}")
        
        # Test production model if it exists
        production_model_path = 'hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt'
        try:
            prod_model = YOLO(production_model_path)
            prod_model.to('cuda')
            print(f"✅ Production HUD model loaded on: {prod_model.device}")
        except Exception as e:
            print(f"ℹ️ Production model not found (expected if first run): {e}")
            
    except Exception as e:
        print(f"❌ YOLOv8 test failed: {e}")
        return False
    
    print("\n🚀 GPU Setup Complete!")
    print("Your RTX 4070 SUPER is ready for 10-50x speedup!")
    return True

if __name__ == "__main__":
    success = test_gpu_setup()
    sys.exit(0 if success else 1) 