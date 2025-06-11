"""
SpygateAI Production Configuration
Optimized settings for maximum performance
"""
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GPUConfig:
    """GPU optimization settings"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    allow_growth: bool = True
    mixed_precision: bool = True  # Enable for RTX cards
    optimize_for_inference: bool = True
    batch_size: int = 1  # Video processing batch size
    

@dataclass
class ModelConfig:
    """Model loading and inference settings"""
    model_path: str = "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 1000
    image_size: int = 640
    half_precision: bool = True  # Use FP16 for speed
    

@dataclass
class OCRConfig:
    """OCR processing settings"""
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_config: str = "--oem 3 --psm 6"
    confidence_threshold: float = 30.0
    language: str = "eng"
    whitelist_chars: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz &:-"
    

@dataclass
class VideoConfig:
    """Video processing settings"""
    input_resolution: Tuple[int, int] = (1920, 1080)
    target_fps: int = 60
    processing_fps: int = 30  # Process every other frame for speed
    buffer_size: int = 5  # Frame buffer size
    output_format: str = "mp4"
    

@dataclass
class PerformanceConfig:
    """Performance monitoring settings"""
    enable_monitoring: bool = True
    log_interval: int = 30  # Log every 30 seconds
    alert_low_fps: float = 15.0
    alert_high_gpu_memory: float = 0.9  # 90% GPU memory usage
    

@dataclass
class ProductionConfig:
    """Complete production configuration"""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    

class ProductionOptimizer:
    """Production optimization utilities"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        
    def optimize_gpu(self):
        """Apply GPU optimizations"""
        if not torch.cuda.is_available():
            print("⚠️  GPU not available - running on CPU")
            return False
            
        # Set memory fraction
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(self.config.gpu.memory_fraction)
            
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Mixed precision setup
        if self.config.gpu.mixed_precision:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        print(f"✅ GPU optimized: {torch.cuda.get_device_name()}")
        print(f"   Memory fraction: {self.config.gpu.memory_fraction}")
        print(f"   Mixed precision: {self.config.gpu.mixed_precision}")
        
        return True
        
    def configure_model(self):
        """Configure model for production"""
        settings = {
            "conf": self.config.model.confidence_threshold,
            "iou": self.config.model.iou_threshold,
            "max_det": self.config.model.max_detections,
            "imgsz": self.config.model.image_size,
            "half": self.config.model.half_precision and torch.cuda.is_available()
        }
        
        print("✅ Model configuration:")
        for key, value in settings.items():
            print(f"   {key}: {value}")
            
        return settings
        
    def configure_ocr(self):
        """Configure OCR for production"""
        import pytesseract
        
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = self.config.ocr.tesseract_path
        
        print("✅ OCR configured:")
        print(f"   Path: {self.config.ocr.tesseract_path}")
        print(f"   Config: {self.config.ocr.tesseract_config}")
        print(f"   Language: {self.config.ocr.language}")
        
        return {
            "config": self.config.ocr.tesseract_config,
            "lang": self.config.ocr.language
        }
        
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets for monitoring"""
        return {
            "target_fps": self.config.video.target_fps,
            "processing_fps": self.config.video.processing_fps,
            "alert_low_fps": self.config.performance.alert_low_fps,
            "max_gpu_memory": self.config.performance.alert_high_gpu_memory
        }
        
    def print_system_info(self):
        """Print complete system information"""
        print("\n🚀 SpygateAI Production Configuration")
        print("=" * 50)
        
        # GPU Info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f}GB")
            print(f"   CUDA: {torch.version.cuda}")
        else:
            print("🎮 GPU: Not Available")
            
        # Model Info
        print(f"\n🤖 Model: {os.path.basename(self.config.model.model_path)}")
        print(f"   Confidence: {self.config.model.confidence_threshold}")
        print(f"   Image Size: {self.config.model.image_size}")
        
        # Video Info
        print(f"\n📺 Video Processing:")
        print(f"   Input Resolution: {self.config.video.input_resolution}")
        print(f"   Target FPS: {self.config.video.target_fps}")
        print(f"   Processing FPS: {self.config.video.processing_fps}")
        
        # Performance
        print(f"\n📊 Performance Monitoring:")
        print(f"   Enabled: {self.config.performance.enable_monitoring}")
        print(f"   Log Interval: {self.config.performance.log_interval}s")
        
        print("\n" + "=" * 50)


def create_production_config() -> ProductionConfig:
    """Create optimized production configuration"""
    
    # Detect RTX card for optimal settings
    gpu_config = GPUConfig()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name().lower()
        if "rtx" in gpu_name or "4070" in gpu_name:
            gpu_config.mixed_precision = True
            gpu_config.memory_fraction = 0.85  # RTX cards can handle more
            
    # Optimize for video resolution
    model_config = ModelConfig()
    model_config.image_size = 640  # Good balance of speed vs accuracy
    
    # OCR optimization
    ocr_config = OCRConfig()
    ocr_config.tesseract_config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz &:-"
    
    return ProductionConfig(
        gpu=gpu_config,
        model=model_config,
        ocr=ocr_config,
        environment="production"
    )


def apply_production_optimizations():
    """Apply all production optimizations"""
    print("🚀 Applying SpygateAI Production Optimizations...")
    
    config = create_production_config()
    optimizer = ProductionOptimizer(config)
    
    # Apply optimizations
    optimizer.optimize_gpu()
    model_settings = optimizer.configure_model()
    ocr_settings = optimizer.configure_ocr()
    
    # Print system info
    optimizer.print_system_info()
    
    return config, optimizer, model_settings, ocr_settings


if __name__ == "__main__":
    # Demo the production configuration
    config, optimizer, model_settings, ocr_settings = apply_production_optimizations()
    
    print("\n✅ Production configuration complete!")
    print("🎯 Ready for maximum performance video analysis") 