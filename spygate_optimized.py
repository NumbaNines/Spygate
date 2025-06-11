#!/usr/bin/env python3
"""
SpygateAI Optimized Launcher
============================

Auto-generated optimized launcher based on system analysis.
Generated: 2025-06-11 13:21:59

Optimizations Applied:
- GPU Memory: 85%
- Preferred Model: yolov8s.pt
- Batch Size: 2
- CPU Workers: 8
"""

import os
import sys
import torch
from pathlib import Path

# Optimization settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def optimize_pytorch():
    """Apply PyTorch optimizations"""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # Enable mixed precision
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        print(f"🎮 GPU optimized: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory limit: 85%")
    else:
        print("⚠️  No GPU detected - running on CPU")

def main():
    """Launch SpygateAI with optimizations"""
    print("⚡ SpygateAI Optimized Launcher")
    print("=" * 40)
    
    # Apply optimizations
    optimize_pytorch()
    
    # Launch main application
    try:
        from spygate_desktop_app_faceit_style import SpygateDesktop
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        window = SpygateDesktop()
        window.show()
        
        print("🚀 SpygateAI launched with optimizations")
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("💡 Try running: python spygate_desktop.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
