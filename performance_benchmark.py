#!/usr/bin/env python3
"""
SpygateAI Performance Benchmarking System
=========================================

Comprehensive performance testing and optimization analysis.
"""

import time
import psutil
import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
from datetime import datetime

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.results = []
        self.gpu_available = torch.cuda.is_available()
        
        # Create benchmark directory
        self.benchmark_dir = Path("benchmarks")
        self.benchmark_dir.mkdir(exist_ok=True)
        
        print("🏁 SpygateAI Performance Benchmark Suite")
        print("=" * 50)
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system information"""
        cpu_cores = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"🖥️  CPU: {cpu_cores} cores")
        print(f"💾 RAM: {memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total")
        
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("🎮 GPU: Not available")
            
        print(f"🔥 PyTorch: {torch.__version__}")
        print()
    
    def benchmark_gpu_performance(self):
        """Test GPU performance"""
        print("🎮 GPU Performance Tests")
        print("-" * 30)
        
        if not self.gpu_available:
            print("   ❌ CUDA not available - skipping GPU tests")
            return
        
        # GPU memory test
        print("⏱️  Testing GPU Memory Allocation...")
        start_time = time.perf_counter()
        
        try:
            size = 1000
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            duration = (time.perf_counter() - start_time) * 1000
            fps = 1000 / duration
            print(f"   ✅ GPU Memory Test: {duration:.1f}ms ({fps:.1f} FPS)")
            
        except Exception as e:
            print(f"   ❌ GPU Memory Test failed: {e}")
        
        # GPU compute test  
        print("⏱️  Testing GPU Matrix Operations...")
        start_time = time.perf_counter()
        
        try:
            size = 2000
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            duration = (time.perf_counter() - start_time) * 1000
            fps = 1000 / duration
            print(f"   ✅ GPU Compute Test: {duration:.1f}ms ({fps:.1f} FPS)")
            
        except Exception as e:
            print(f"   ❌ GPU Compute Test failed: {e}")
    
    def benchmark_yolo_performance(self):
        """Test YOLO model performance"""
        print("🎯 YOLO Model Performance Tests")
        print("-" * 30)
        
        try:
            from ultralytics import YOLO
            
            # Test different model sizes
            models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
            
            for model_name in models:
                if Path(model_name).exists():
                    print(f"⏱️  Testing {model_name}...")
                    
                    try:
                        start_time = time.perf_counter()
                        model = YOLO(model_name)
                        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        results = model(dummy_image, verbose=False)
                        duration = (time.perf_counter() - start_time) * 1000
                        fps = 1000 / duration
                        
                        print(f"   ✅ {model_name}: {duration:.1f}ms ({fps:.1f} FPS)")
                        
                    except Exception as e:
                        print(f"   ❌ {model_name} failed: {e}")
                else:
                    print(f"   ⚠️  {model_name} not found - skipping")
                    
        except ImportError:
            print("   ❌ ultralytics not available - skipping YOLO tests")
    
    def benchmark_video_processing(self):
        """Test video processing performance"""
        print("🎬 Video Processing Performance Tests")
        print("-" * 30)
        
        # Frame processing test
        print("⏱️  Testing Frame Processing...")
        start_time = time.perf_counter()
        
        try:
            # Create dummy video frame
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            
            # Simulate typical processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(frame, (640, 640))
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            duration = (time.perf_counter() - start_time) * 1000
            fps = 1000 / duration
            print(f"   ✅ Frame Processing: {duration:.1f}ms ({fps:.1f} FPS)")
            
        except Exception as e:
            print(f"   ❌ Frame Processing failed: {e}")
    
    def generate_report(self):
        """Generate performance report"""
        print("\n📊 Performance Report Generated")
        print("=" * 50)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu_available": self.gpu_available,
            "gpu_name": torch.cuda.get_device_name(0) if self.gpu_available else "None",
            "pytorch_version": torch.__version__,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_benchmark_{timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Report saved to: {filepath}")
        
        # Print recommendations
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        return report
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.gpu_available:
            recommendations.append("GPU acceleration is available - ensure all operations use CUDA when possible")
            recommendations.append("Consider mixed precision training for better performance")
        else:
            recommendations.append("No GPU detected - consider upgrading for better performance")
        
        recommendations.append("Monitor GPU memory usage to prevent out-of-memory errors")
        recommendations.append("Use batch processing for multiple images/videos")
        recommendations.append("Consider model optimization (quantization, pruning) for production")
        
        return recommendations
    
    def run_benchmark(self):
        """Run all benchmarks"""
        print("🚀 Starting Performance Benchmark\n")
        
        self.benchmark_gpu_performance()
        print()
        self.benchmark_yolo_performance()
        print()
        self.benchmark_video_processing()
        
        # Generate report
        report = self.generate_report()
        return report

def main():
    """Main entry point"""
    benchmark = PerformanceBenchmark()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
