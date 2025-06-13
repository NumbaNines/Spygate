"""
YOLOv8 model implementation with hardware optimization and advanced features.
"""

import torch
import torch._dynamo

# Disable compilation to avoid Triton dependency on Windows
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch._dynamo.config.cache_size_limit = 1
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Disable torch.compile globally to prevent Triton warnings
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import os

try:
    import torch.nn as nn
    from ultralytics.engine.model import Model
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import ops

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    YOLO = None
    Model = None
    BasePredictor = None
    BaseTrainer = None
    ops = None

try:
    from ..core.gpu_memory_manager import AdvancedGPUMemoryManager, get_memory_manager
    from ..core.hardware import HardwareDetector, HardwareTier

    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False
    HardwareDetector = None
    HardwareTier = None
    AdvancedGPUMemoryManager = None
    get_memory_manager = None

logger = logging.getLogger(__name__)

# UI Classes for HUD detection - Essential elements for game situation analysis
UI_CLASSES = [
    "hud",  # Main HUD bar (dark/black bar at bottom containing all game info)
    "possession_triangle_area",  # Left triangle area between team abbreviations (shows ball possession)
    "territory_triangle_area",  # Right triangle area next to yard line (▲ = opponent territory, ▼ = own territory)
    "preplay_indicator",  # Bottom left indicator shown only pre-play (indicates play about to start)
    "play_call_screen",  # Play call screen overlay (indicates play has ended)
]

# Enhanced hardware-tier specific model configurations with optimization features
MODEL_CONFIGS = (
    {
        HardwareTier.ULTRA_LOW: {  # Integrated GPUs or CPU only
            "model_size": "n",      # YOLOv8n - nano for minimum requirements
            "img_size": 320,        # Smaller input size for speed
            "batch_size": 1,        # Single image processing
            "half": False,          # No FP16 for compatibility
            "device": "cpu",        # Force CPU for integrated graphics
            "max_det": 10,
            "conf": 0.4,            # Higher confidence for fewer detections
            "iou": 0.7,
            "optimize": True,
            "quantize": False,      # Disable for stability
            "compile": False,
            "warmup_epochs": 1,
            "patience": 5,
            "workers": 1
        },
        HardwareTier.LOW: {         # Entry GPUs (GTX 1650, RTX 3050)
            "model_size": "n",      # Still using nano model
            "img_size": 416,        # Slightly larger for better accuracy
            "batch_size": 2,
            "half": True,           # Enable FP16 for modern GPUs
            "device": "auto",
            "max_det": 20,
            "conf": 0.3,
            "iou": 0.6,
            "optimize": True,
            "quantize": True,       # Enable basic optimizations
            "compile": False,       # Skip compilation for stability
            "warmup_epochs": 2,
            "patience": 10,
            "workers": 2
        },
        HardwareTier.MEDIUM: {      # Mid-range GPUs (RTX 2060-3060)
            "model_size": "s",      # Small model for balance
            "img_size": 640,
            "batch_size": 4,
            "half": True,
            "device": "auto",
            "max_det": 50,
            "conf": 0.25,
            "iou": 0.5,
            "optimize": True,
            "quantize": True,
            "compile": False,  # Disabled to avoid Triton warnings
            "warmup_epochs": 3,
            "patience": 15,
            "workers": 4
        },
        HardwareTier.HIGH: {        # High-end GPUs (RTX 3070-4060)
            "model_size": "m",      # Medium model
            "img_size": 832,
            "batch_size": 6,        # Reduced from 12 for wider compatibility
            "half": True,
            "device": "cuda",
            "max_det": 100,
            "conf": 0.2,
            "iou": 0.45,
            "optimize": True,
            "quantize": True,
            "compile": False,  # Disabled to avoid Triton warnings
            "warmup_epochs": 3,
            "patience": 15,
            "workers": 6
        },
        HardwareTier.ULTRA: {       # Enthusiast GPUs (RTX 4070+)
            "model_size": "l",      # Large model
            "img_size": 1024,       # Reduced from 1280 for better performance
            "batch_size": 8,        # Reduced from 16 for stability
            "half": True,
            "device": "cuda",
            "max_det": 200,         # Reduced from 300
            "conf": 0.15,
            "iou": 0.4,
            "optimize": True,
            "quantize": True,
            "compile": True,
            "warmup_epochs": 5,     # Reduced from 10
            "patience": 20,         # Reduced from 30
            "workers": 8
        }
    }
    if TORCH_AVAILABLE and HardwareTier
    else {}
)


@dataclass
class PerformanceMetrics:
    """Performance tracking for model optimization."""

    inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    gpu_utilization: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_measurement(
        self,
        inference_time: float,
        memory_mb: float,
        batch_size: int,
        accuracy: Optional[float] = None,
        gpu_util: Optional[float] = None,
    ):
        """Add a performance measurement."""
        self.inference_times.append(inference_time)
        self.memory_usage.append(memory_mb)
        self.batch_sizes.append(batch_size)

        if accuracy is not None:
            self.accuracy_scores.append(accuracy)
        if gpu_util is not None:
            self.gpu_utilization.append(gpu_util)

    def get_average_metrics(self) -> dict:
        """Get average performance metrics."""
        return {
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0.0,
            "avg_batch_size": np.mean(self.batch_sizes) if self.batch_sizes else 0.0,
            "avg_accuracy": np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0,
            "avg_gpu_utilization": np.mean(self.gpu_utilization) if self.gpu_utilization else 0.0,
            "total_inferences": len(self.inference_times),
        }


@dataclass
class OptimizationConfig:
    """Configuration for model optimization features."""

    enable_dynamic_switching: bool = True
    enable_adaptive_batch_size: bool = True
    enable_performance_monitoring: bool = True
    enable_auto_optimization: bool = True

    # Performance thresholds for optimization
    max_inference_time: float = 1.0  # Maximum acceptable inference time (seconds)
    min_accuracy: float = 0.85  # Minimum acceptable accuracy
    max_memory_usage: float = 0.9  # Maximum GPU memory usage (fraction)

    # Optimization intervals
    optimization_interval: int = 50  # Optimize every N inferences
    monitoring_interval: int = 10  # Monitor performance every N inferences


@dataclass
class DetectionResult:
    """Structured detection result with enhanced metrics."""

    boxes: np.ndarray  # Bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # Confidence scores
    classes: np.ndarray  # Class indices
    class_names: list[str]  # Class names
    processing_time: float  # Time taken for detection
    memory_usage: Optional[dict] = None  # Memory usage statistics
    model_config: Optional[dict] = None  # Model configuration used
    optimization_applied: Optional[dict] = None  # Optimizations applied


class EnhancedYOLOv8:
    """Enhanced YOLOv8 model with hardware-adaptive features."""
    
    def __init__(self, 
                 model_path: str = "models/yolov8n.pt",
                 hardware_tier: Optional[HardwareTier] = None,
                 confidence: float = 0.25,
                 device: Optional[str] = None,
                 optimization_config: Optional[OptimizationConfig] = None):
        """Initialize the enhanced YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights
            hardware_tier: Hardware tier for adaptive settings
            confidence: Confidence threshold for detections
            device: Device to run model on ('cpu', 'cuda', etc.)
            optimization_config: Optional configuration for model optimization
        """
        self.hardware = hardware_tier or HardwareDetector().detect_tier()
        self.model = YOLO(model_path)
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # Configure model based on hardware tier
        if self.hardware == HardwareTier.ULTRA_LOW:
            self.model.conf = 0.4
            self.model.imgsz = 320
        elif self.hardware == HardwareTier.LOW:
            self.model.conf = 0.3
            self.model.imgsz = 416
        elif self.hardware == HardwareTier.MEDIUM:
            self.model.conf = confidence
            self.model.imgsz = 640
        elif self.hardware == HardwareTier.HIGH:
            self.model.conf = confidence
            self.model.imgsz = 832
        else:  # ULTRA
            self.model.conf = confidence
            self.model.imgsz = 1280
            
        # Set device
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to('cuda')
            
        # Enable model optimizations for higher tiers (disabled to avoid Triton warnings)
        # if self.hardware in [HardwareTier.HIGH, HardwareTier.ULTRA]:
        #     if hasattr(torch, 'compile'):
        #         self.model = torch.compile(self.model)
                
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics()
        self.inference_counter = 0
        
        # Initialize memory manager if available
        self.memory_manager = get_memory_manager() if MEMORY_MANAGER_AVAILABLE else None
        
        # Set optimal batch size based on hardware tier
        self.optimal_batch_size = MODEL_CONFIGS[self.hardware]["batch_size"] if self.hardware in MODEL_CONFIGS else 1
        
    def detect(self, frame) -> List[Dict]:
        """Run detection on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections with bounding boxes and classes
        """
        results = self.model(frame, verbose=False)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = UI_CLASSES[cls]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': class_name
                })
                
        return detections

    def predict_with_optimization(
        self,
        source,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        half: Optional[bool] = None,
        device: Optional[str] = None,
        max_det: Optional[int] = None,
        return_memory_stats: bool = False,
        **kwargs,
    ):
        """Enhanced predict method with optimization and performance tracking."""
        # Use optimized settings
        conf = conf or self.model.conf
        iou = iou or self.model.iou
        imgsz = imgsz or self.model.imgsz
        half = half if half is not None else self.model.half
        device = device or self.model.device
        max_det = max_det or self.model.max_det

        memory_stats = {}
        optimization_applied = {}

        # Record initial memory state
        if self.memory_manager and torch.cuda.is_available():
            memory_stats["initial"] = self.memory_manager.get_memory_stats()

        # Start timing
        start_time = time.time()

        try:
            # Apply adaptive batch sizing if enabled
            if self.optimization_config.enable_adaptive_batch_size:
                if hasattr(source, "__len__") and len(source) > self.optimal_batch_size:
                    # Process in optimal batch sizes
                    results = []
                    for i in range(0, len(source), self.optimal_batch_size):
                        batch = source[i : i + self.optimal_batch_size]
                        batch_results = self.model(
                            source=batch,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            half=half,
                            device=device,
                            max_det=max_det,
                            verbose=False,
                            **kwargs,
                        )
                        results.extend(batch_results)
                    optimization_applied["adaptive_batching"] = True
                else:
                    # Single prediction
                    results = self.model(
                        source=source,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        half=half,
                        device=device,
                        max_det=max_det,
                        verbose=False,
                        **kwargs,
                    )
            else:
                # Standard prediction
                results = self.model(
                    source=source,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    half=half,
                    device=device,
                    max_det=max_det,
                    verbose=False,
                    **kwargs,
                )

            processing_time = time.time() - start_time

            # Record performance metrics
            if self.optimization_config.enable_performance_monitoring:
                batch_size = 1
                if hasattr(source, "__len__"):
                    batch_size = len(source)
                elif hasattr(source, "shape") and len(source.shape) == 4:
                    batch_size = source.shape[0]

                memory_mb = 0
                if self.memory_manager and torch.cuda.is_available():
                    memory_stats_current = self.memory_manager.get_memory_stats()
                    memory_mb = memory_stats_current.get("allocated_memory_gb", 0) * 1024

                self.performance_metrics.add_measurement(
                    inference_time=processing_time, memory_mb=memory_mb, batch_size=batch_size
                )

            # Record batch performance
            if self.memory_manager:
                batch_size = 1
                if hasattr(source, "__len__"):
                    batch_size = len(source)
                elif hasattr(source, "shape"):
                    batch_size = source.shape[0] if len(source.shape) == 4 else 1

                self.memory_manager.record_batch_performance(
                    batch_size=batch_size, processing_time=processing_time, success=True
                )

            # Record final memory state
            if self.memory_manager and torch.cuda.is_available():
                memory_stats["final"] = self.memory_manager.get_memory_stats()
                memory_stats["processing_time"] = processing_time

            # Increment inference counter and check for optimization
            self.inference_counter += 1
            if (
                self.optimization_config.enable_auto_optimization
                and self.inference_counter % self.optimization_config.optimization_interval == 0
            ):
                self._check_performance_and_optimize()

            if return_memory_stats:
                return results, memory_stats, optimization_applied
            return results

        except Exception as e:
            processing_time = time.time() - start_time

            # Record failed batch performance
            if self.memory_manager:
                batch_size = 1
                if hasattr(source, "__len__"):
                    batch_size = len(source)
                elif hasattr(source, "shape"):
                    batch_size = source.shape[0] if len(source.shape) == 4 else 1

                self.memory_manager.record_batch_performance(
                    batch_size=batch_size, processing_time=0.0, success=False
                )

            logger.error(f"YOLOv8 prediction failed: {e}")
            raise

    # Alias for backward compatibility
    predict_with_memory_tracking = predict_with_optimization

    def detect_hud_elements(
        self, frame: np.ndarray, return_memory_stats: bool = False
    ) -> Union[DetectionResult, tuple[DetectionResult, dict]]:
        """Detect HUD elements in a frame with structured output and optimization."""
        try:
            # Run optimized prediction
            if return_memory_stats:
                results, memory_stats, optimization_applied = self.predict_with_optimization(
                    frame, return_memory_stats=True
                )
            else:
                results = self.predict_with_optimization(frame)
                memory_stats = None
                optimization_applied = None

            # Extract results from ultralytics format
            if results and len(results) > 0:
                result = results[0]  # Get first (and only) result

                # Extract detection data
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                else:
                    boxes = np.empty((0, 4))
                    scores = np.empty(0)
                    classes = np.empty(0, dtype=int)

                # Get class names
                class_names = [
                    UI_CLASSES[i] if i < len(UI_CLASSES) else f"class_{i}"
                    for i in classes
                ]

                detection_result = DetectionResult(
                    boxes=boxes,
                    scores=scores,
                    classes=classes,
                    class_names=class_names,
                    processing_time=(
                        memory_stats.get("processing_time", 0.0) if memory_stats else 0.0
                    ),
                    memory_usage=memory_stats,
                    model_config=self.model.model.model.state_dict(),
                    optimization_applied=optimization_applied,
                )

                if return_memory_stats:
                    return detection_result, memory_stats
                return detection_result

            else:
                # No detections
                detection_result = DetectionResult(
                    boxes=np.empty((0, 4)),
                    scores=np.empty(0),
                    classes=np.empty(0, dtype=int),
                    class_names=[],
                    processing_time=(
                        memory_stats.get("processing_time", 0.0) if memory_stats else 0.0
                    ),
                    memory_usage=memory_stats,
                    model_config=self.model.model.model.state_dict(),
                    optimization_applied=optimization_applied,
                )

                if return_memory_stats:
                    return detection_result, memory_stats
                return detection_result

        except Exception as e:
            logger.error(f"HUD element detection failed: {e}")
            # Return empty result
            detection_result = DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty(0),
                classes=np.empty(0, dtype=int),
                class_names=[],
                processing_time=0.0,
                memory_usage=None,
                model_config=self.model.model.model.state_dict(),
                optimization_applied=None,
            )

            if return_memory_stats:
                return detection_result, {}
            return detection_result

    def get_memory_optimized_batch_size(
        self, input_shape: tuple, safety_factor: float = 0.8
    ) -> int:
        """Calculate memory-optimized batch size for given input shape."""
        if not self.memory_manager:
            return self.model.max_det

        return self.memory_manager.get_optimal_batch_size(
            estimated_items=1, item_memory_mb=None, safety_factor=safety_factor
        )

    def get_memory_buffer(self, size: tuple, dtype=None):
        """Get a memory buffer with pool management."""
        if self.memory_manager:
            return self.memory_manager.get_buffer(size, dtype)
        else:
            return torch.zeros(
                size, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu"
            )

    def return_memory_buffer(self, tensor):
        """Return a memory buffer to the pool."""
        if self.memory_manager:
            self.memory_manager.return_buffer(tensor)
        else:
            del tensor

    def get_performance_report(self) -> dict:
        """Get comprehensive performance report."""
        metrics = self.performance_metrics.get_average_metrics()

        report = {
            "model_info": {
                "model_size": self.model.model.model.state_dict()['model.model.0.weight'].shape[1],
                "hardware_tier": self.hardware.name,
                "device": self.model.device,
            },
            "performance_metrics": metrics,
            "configuration": {
                "batch_size": self.model.max_det,
                "confidence_threshold": self.model.conf,
                "iou_threshold": self.model.iou,
                "input_size": self.model.imgsz,
                "half_precision": self.model.half,
            },
            "optimization_config": {
                "dynamic_switching_enabled": self.optimization_config.enable_dynamic_switching,
                "adaptive_batch_size_enabled": self.optimization_config.enable_adaptive_batch_size,
                "auto_optimization_enabled": self.optimization_config.enable_auto_optimization,
                "performance_monitoring_enabled": self.optimization_config.enable_performance_monitoring,
            },
            "total_inferences": self.inference_counter,
        }

        return report

    def benchmark_model(self, test_images: list[np.ndarray], iterations: int = 10) -> dict:
        """Benchmark model performance with given test images."""
        logger.info(
            f"Starting model benchmark with {len(test_images)} images, {iterations} iterations"
        )

        benchmark_results = {
            "inference_times": [],
            "memory_usage": [],
            "detection_counts": [],
            "model_config": self.model.model.model.state_dict(),
        }

        for _iteration in range(iterations):
            iteration_times = []
            iteration_memory = []
            iteration_detections = []

            for image in test_images:
                start_time = time.time()

                # Record memory before
                memory_before = 0
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()

                # Run detection
                result = self.detect_hud_elements(image)

                # Record metrics
                inference_time = time.time() - start_time
                memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_used = (memory_after - memory_before) / 1024 / 1024  # MB

                iteration_times.append(inference_time)
                iteration_memory.append(memory_used)
                iteration_detections.append(len(result.boxes))

            benchmark_results["inference_times"].append(np.mean(iteration_times))
            benchmark_results["memory_usage"].append(np.mean(iteration_memory))
            benchmark_results["detection_counts"].append(np.mean(iteration_detections))

        # Calculate statistics
        benchmark_results.update(
            {
                "avg_inference_time": np.mean(benchmark_results["inference_times"]),
                "std_inference_time": np.std(benchmark_results["inference_times"]),
                "avg_memory_usage": np.mean(benchmark_results["memory_usage"]),
                "avg_detections": np.mean(benchmark_results["detection_counts"]),
                "min_inference_time": np.min(benchmark_results["inference_times"]),
                "max_inference_time": np.max(benchmark_results["inference_times"]),
            }
        )

        logger.info(
            f"Benchmark completed. Avg inference time: {benchmark_results['avg_inference_time']:.3f}s"
        )
        return benchmark_results

    def save_performance_report(self, filepath: str):
        """Save performance report to file."""
        report = self.get_performance_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to: {filepath}")

    def log_capabilities(self):
        """Log model capabilities and optimizations."""
        logger.info("Enhanced YOLOv8 Model Capabilities:")
        logger.info(f"- Hardware Tier: {self.hardware.name}")
        try:
            state_dict = self.model.model.model.state_dict()
            if 'model.model.0.weight' in state_dict:
                logger.info(f"- Model Size: YOLOv8{state_dict['model.model.0.weight'].shape[1]}")
                logger.info(f"- Optimizations: {state_dict['model.model.0.weight'].shape[1] - state_dict['model.model.0.weight'].shape[0]}")
            else:
                logger.warning("Model key 'model.model.0.weight' not found in state_dict. Skipping model size/optimizations log.")
        except Exception as e:
            logger.warning(f"Could not log model size/optimizations: {e}")
        logger.info(f"- Input Size: {self.model.imgsz}px")
        batch_size = getattr(self.model, 'max_det', None)
        if batch_size is not None:
            logger.info(f"- Batch Size: {batch_size}")
        else:
            logger.warning("YOLO model has no attribute 'max_det'. Skipping batch size log.")
        logger.info(f"- Device: {self.model.device}")
        logger.info(f"- Half Precision: {self.model.half}")
        logger.info(f"- Classes: {len(UI_CLASSES)}")
        logger.info(f"- Dynamic Switching: {self.optimization_config.enable_dynamic_switching}")
        logger.info(f"- Adaptive Batching: {self.optimization_config.enable_adaptive_batch_size}")
        logger.info(
            f"- Performance Monitoring: {self.optimization_config.enable_performance_monitoring}"
        )
        if self.memory_manager:
            logger.info("- Advanced GPU Memory Management")

    def get_model_memory_stats(self) -> dict:
        """Get comprehensive model memory statistics."""
        stats = {
            "hardware_tier": self.hardware.name,
            "optimal_batch_size": self.model.max_det,
            "model_size": self.model.model.model.state_dict()['model.model.0.weight'].shape[1],
            "input_size": self.model.imgsz,
            "device": self.model.device,
            "half_precision": self.model.half,
            "available_variants": [],
            "optimizations_applied": self.model.model.model.state_dict()['model.model.0.weight'].shape[1] - self.model.model.model.state_dict()['model.model.0.weight'].shape[0],
        }

        if hasattr(self.model.model.model, "parameters"):
            stats["model_parameters"] = sum(p.numel() for p in self.model.model.model.parameters())

        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            stats.update(memory_stats)

        return stats

    def optimize_memory(self):
        """Optimize memory usage based on GPU capabilities."""
        if not torch.cuda.is_available():
            print("⚠️ Running on CPU - no GPU memory optimizations needed")
            return
            
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # Set memory fraction based on GPU memory
        memory_fraction = 0.8  # Default
        if gpu_memory >= 16:  # High-end GPUs (16GB+)
            memory_fraction = 0.85
        elif gpu_memory >= 8:  # Mid-range GPUs (8GB)
            memory_fraction = 0.80
        elif gpu_memory >= 6:  # Entry GPUs (6GB)
            memory_fraction = 0.75
        else:  # Low memory GPUs
            memory_fraction = 0.70
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Enable flash attention only for 8GB+ GPUs
        if hasattr(torch.nn.functional, 'flash_attention') and gpu_memory >= 8:
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable TF32 for Ampere+ GPUs (RTX 30 series and newer)
        device_name = torch.cuda.get_device_name().lower()
        if any(x in device_name for x in ['rtx 30', 'rtx 40', 'rtx 20']):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print(f"✅ GPU memory optimizations enabled ({memory_fraction*100:.0f}% of {gpu_memory:.1f}GB)")
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")


# Aliases for compatibility
CustomYOLOv8 = EnhancedYOLOv8
# Aliases for clarity and backward compatibility
YOLOv8Model = EnhancedYOLOv8
SpygateYOLO = EnhancedYOLOv8
# Legacy YOLO11 compatibility removed - use EnhancedYOLOv8 directly


def load_optimized_yolov8_model(
    model_path: Optional[str] = None,
    hardware_tier: Optional[HardwareTier] = None,
    conf: float = 0.25,
    device: Optional[str] = None,
) -> EnhancedYOLOv8:
    """Load and configure optimized YOLOv8 model with advanced features."""
    return EnhancedYOLOv8(
        model_path=model_path, hardware_tier=hardware_tier, confidence=conf, device=device
    )


def get_hardware_optimized_config(hardware_tier: HardwareTier) -> dict:
    """Get hardware-optimized configuration for YOLOv8."""
    return MODEL_CONFIGS.get(hardware_tier, MODEL_CONFIGS[HardwareTier.LOW])


def create_optimization_config(
    enable_dynamic_switching: bool = True,
    enable_adaptive_batch_size: bool = True,
    enable_performance_monitoring: bool = True,
    enable_auto_optimization: bool = True,
    max_inference_time: float = 1.0,
    min_accuracy: float = 0.85,
    max_memory_usage: float = 0.9,
) -> OptimizationConfig:
    """Create an optimization configuration with custom settings."""
    return OptimizationConfig(
        enable_dynamic_switching=enable_dynamic_switching,
        enable_adaptive_batch_size=enable_adaptive_batch_size,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_auto_optimization=enable_auto_optimization,
        max_inference_time=max_inference_time,
        min_accuracy=min_accuracy,
        max_memory_usage=max_memory_usage,
    )
