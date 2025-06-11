#!/usr/bin/env python3
"""
Check GPU status and provide optimization recommendations for RTX 4070 SUPER.
"""

import torch
import gc

def check_gpu_status():
    """Check current GPU status and capabilities."""
    print("🔧 GPU Status Check")
    print("=" * 50)
    
    # Basic CUDA info
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        # Get device info
        device = torch.cuda.current_device()
        print(f"Current Device: {device}")
        print(f"Device Name: {torch.cuda.get_device_name(device)}")
        
        # Memory info
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / (1024**3)  # Convert to GB
        print(f"Total GPU Memory: {total_memory:.1f} GB")
        
        # Current memory usage
        torch.cuda.empty_cache()  # Clear cache first
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        cached = torch.cuda.memory_reserved(device) / (1024**3)
        free_memory = total_memory - allocated
        
        print(f"Currently Allocated: {allocated:.2f} GB")
        print(f"Currently Cached: {cached:.2f} GB") 
        print(f"Available Memory: {free_memory:.2f} GB")
        
        # RTX 4070 SUPER specific optimizations
        if "4070" in torch.cuda.get_device_name(device):
            print("\n🚀 RTX 4070 SUPER Detected!")
            print("Recommended optimizations:")
            print("✅ Use batch_size=32 for training")
            print("✅ Enable mixed precision (FP16)")
            print("✅ Use YOLOv8s model size")
            print("✅ Enable CUDA memory optimization")
            print("✅ Use compile optimization")
            return {
                "device": "cuda",
                "batch_size": 32,
                "mixed_precision": True,
                "model_size": "s",
                "optimize": True,
                "compile": True
            }
        else:
            print("\n⚙️  Generic GPU optimizations:")
            return {
                "device": "cuda", 
                "batch_size": 16,
                "mixed_precision": True,
                "optimize": True
            }
    else:
        print("❌ CUDA not available - will use CPU")
        return {
            "device": "cpu",
            "batch_size": 4,
            "mixed_precision": False,
            "optimize": False
        }

def optimize_memory():
    """Optimize GPU memory for maximum performance."""
    if torch.cuda.is_available():
        print("\n🔧 Optimizing GPU Memory...")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Enable memory efficiency
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory growth for TensorFlow-style allocation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("✅ GPU memory optimized for maximum performance")
        print("✅ CuDNN benchmark enabled")
        print("✅ Memory allocation optimized")

if __name__ == "__main__":
    config = check_gpu_status()
    optimize_memory()
    
    print(f"\n🎯 Recommended Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}") 