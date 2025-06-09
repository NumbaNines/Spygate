"""YOLOv8 model architecture for SpygateAI HUD element detection with ultralytics.

This is the canonical YOLOv8 implementation for SpygateAI as specified in the PRD.
Uses ultralytics YOLOv8 with enhanced optimization, dynamic switching, and performance monitoring.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
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
    "score_bug",  # Overall score display area within HUD (contains team info, scores, timeouts)
    "away_team",  # Away team abbreviation and score (leftmost team on screen)
    "home_team",  # Home team abbreviation and score (rightmost team abbreviation on screen)
    "down_distance",  # Down and distance indicator (e.g., "1st & 10", "4th")
    "game_clock",  # Game time remaining (e.g., "1:57")
    "play_clock",  # Play clock countdown (e.g., ":04") - visible PRE-SNAP only
    "yards_to_goal",  # Numeric yard line display in HUD next to territory indicator (e.g., "25", "3", "1")
    "qb_position",  # QB/ball position indicating hash mark placement (left hash, right hash, center)
    "left_hash_mark",  # Left hash mark line on field for positional analysis
    "right_hash_mark",  # Right hash mark line on field for positional analysis
    "possession_indicator",  # Triangle on LEFT side between team abbreviations (shows ball possession)
    "territory_indicator",  # Triangle on FAR RIGHT side (▲ = opponent territory, ▼ = own territory)
]

# Enhanced hardware-tier specific model configurations with optimization features
MODEL_CONFIGS = (
    {
        HardwareTier.ULTRA_LOW: {
            "model_size": "n",  # YOLOv8n - nano
            "img_size": 320,
            "batch_size": 1,
            "half": False,  # Disable FP16 for better compatibility
            "device": "cpu",
            "max_det": 10,
            "conf": 0.4,
            "iou": 0.7,
            "optimize": True,  # Enable optimization
            "quantize": False,  # Disable quantization for stability
            "compile": False,  # Disable model compilation
            "warmup_epochs": 1,
            "patience": 5,
            "workers": 1,
        },
        HardwareTier.LOW: {
            "model_size": "n",  # YOLOv8n - nano
            "img_size": 416,
            "batch_size": 2,
            "half": False,
            "device": "auto",
            "max_det": 20,
            "conf": 0.3,
            "iou": 0.6,
            "optimize": True,
            "quantize": False,
            "compile": False,
            "warmup_epochs": 2,
            "patience": 10,
            "workers": 2,
        },
        HardwareTier.MEDIUM: {
            "model_size": "s",  # YOLOv8s - small
            "img_size": 640,
            "batch_size": 4,
            "half": True,
            "device": "auto",
            "max_det": 50,
            "conf": 0.25,
            "iou": 0.5,
            "optimize": True,
            "quantize": True,  # Enable quantization for performance
            "compile": True,  # Enable model compilation
            "warmup_epochs": 3,
            "patience": 15,
            "workers": 4,
        },
        HardwareTier.HIGH: {
            "model_size": "m",  # YOLOv8m - medium
            "img_size": 832,
            "batch_size": 8,
            "half": True,
            "device": "auto",
            "max_det": 100,
            "conf": 0.2,
            "iou": 0.45,
            "optimize": True,
            "quantize": True,
            "compile": True,
            "warmup_epochs": 5,
            "patience": 20,
            "workers": 6,
        },
        HardwareTier.ULTRA: {
            "model_size": "l",  # YOLOv8l - large
            "img_size": 1280,
            "batch_size": 16,
            "half": True,
            "device": "auto",
            "max_det": 300,
            "conf": 0.15,
            "iou": 0.4,
            "optimize": True,
            "quantize": True,
            "compile": True,
            "warmup_epochs": 10,
            "patience": 30,
            "workers": 8,
        },
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


class EnhancedYOLOv8(YOLO if TORCH_AVAILABLE else object):
    """Enhanced YOLOv8 model with advanced optimization, dynamic switching, and performance monitoring."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        hardware: Optional[HardwareDetector] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize the Enhanced YOLOv8 model with advanced optimization features."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for YOLOv8 functionality")

        # Initialize hardware detection
        self.hardware = hardware or HardwareDetector()
        self.optimization_config = optimization_config or OptimizationConfig()

        # Initialize base configuration
        self.base_config = MODEL_CONFIGS.get(self.hardware.tier, MODEL_CONFIGS[HardwareTier.LOW])
        self.config = self.base_config.copy()

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.inference_counter = 0
        self.last_optimization = 0

        # Model variants for dynamic switching
        self.model_variants = {}
        self.current_variant = "default"

        # Threading for background optimization
        self.optimization_lock = threading.Lock()

        # Initialize the primary model
        self._initialize_primary_model(model_path)

        # Setup memory management
        self._setup_memory_management()

        # Configure device and optimizations
        self._setup_device()
        self._apply_optimizations()

        # Store class names
        self.class_names = UI_CLASSES

        # Start background monitoring if enabled
        if self.optimization_config.enable_performance_monitoring:
            self._start_background_monitoring()

        logger.info(
            f"Enhanced YOLOv8 model initialized for {self.hardware.tier.name} hardware with optimizations"
        )

    def _initialize_primary_model(self, model_path: Optional[str]):
        """Initialize the primary YOLO model."""
        # Determine model to load
        if model_path and Path(model_path).exists():
            model_to_load = model_path
            logger.info(f"Loading custom YOLOv8 model from: {model_path}")
        else:
            # Use pre-trained YOLOv8 model based on hardware tier
            model_size = self.config["model_size"]
            model_to_load = f"yolov8{model_size}.pt"
            logger.info(f"Loading pre-trained YOLOv8{model_size} model")

        # Initialize the YOLO model
        try:
            super().__init__(model_to_load)
            logger.info(f"Successfully loaded YOLOv8 model: {model_to_load}")

            # Store the default variant
            self.model_variants["default"] = self.model

        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # Fallback to smallest model
            super().__init__("yolov8n.pt")
            logger.info("Loaded fallback YOLOv8n model")
            self.model_variants["default"] = self.model

    def _setup_memory_management(self):
        """Set up advanced GPU memory management."""
        self.memory_manager = None
        self.optimal_batch_size = self.config["batch_size"]

        if MEMORY_MANAGER_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager()
                if self.memory_manager is None:
                    from ..core.gpu_memory_manager import initialize_memory_manager

                    self.memory_manager = initialize_memory_manager(self.hardware)

                # Get optimal batch size from memory manager
                if hasattr(self.memory_manager, "get_optimal_batch_size"):
                    self.optimal_batch_size = self.memory_manager.get_optimal_batch_size()

                logger.info(
                    f"GPU Memory Manager integrated. Optimal batch size: {self.optimal_batch_size}"
                )
            except ImportError:
                logger.warning("GPU Memory Manager not available, using basic memory management")

    def _setup_device(self):
        """Configure the device for inference with optimizations."""
        device_config = self.config["device"]

        if device_config == "auto":
            if torch.cuda.is_available() and self.hardware.has_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device_config

        # Move model to device
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        logger.info(f"YOLOv8 model configured for device: {self.device}")

    def _apply_optimizations(self):
        """Apply hardware-tier specific optimizations."""
        optimizations_applied = []

        try:
            # Model compilation optimization
            if (
                self.config.get("compile", False)
                and hasattr(torch, "compile")
                and self.device == "cuda"
            ):
                self.model = torch.compile(self.model)
                optimizations_applied.append("torch_compile")
                logger.info("Applied torch.compile optimization")

            # Half precision optimization
            if self.config.get("half", False) and self.device == "cuda":
                self.model.half()
                optimizations_applied.append("half_precision")
                logger.info("Applied half precision optimization")

            # Model optimization for inference
            if self.config.get("optimize", True):
                self.model.eval()
                if hasattr(self.model, "fuse"):
                    self.model.fuse()
                    optimizations_applied.append("fused_layers")
                    logger.info("Applied layer fusion optimization")

            # Set optimized inference parameters
            if hasattr(self.model, "model"):
                for m in self.model.model.modules():
                    if hasattr(m, "inplace"):
                        m.inplace = True

            self.applied_optimizations = optimizations_applied
            logger.info(f"Applied optimizations: {optimizations_applied}")

        except Exception as e:
            logger.warning(f"Some optimizations failed to apply: {e}")
            self.applied_optimizations = optimizations_applied

    def _start_background_monitoring(self):
        """Start background performance monitoring."""

        def monitor():
            while True:
                time.sleep(30)  # Monitor every 30 seconds
                try:
                    self._check_performance_and_optimize()
                except Exception as e:
                    logger.warning(f"Background monitoring error: {e}")

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info("Started background performance monitoring")

    def _check_performance_and_optimize(self):
        """Check performance metrics and apply optimizations if needed."""
        if not self.optimization_config.enable_auto_optimization:
            return

        with self.optimization_lock:
            metrics = self.performance_metrics.get_average_metrics()

            # Check if optimization is needed
            needs_optimization = False
            optimization_reason = []

            if metrics["avg_inference_time"] > self.optimization_config.max_inference_time:
                needs_optimization = True
                optimization_reason.append("high_inference_time")

            if (
                metrics["avg_memory_usage"] > self.optimization_config.max_memory_usage * 1000
            ):  # Convert to MB
                needs_optimization = True
                optimization_reason.append("high_memory_usage")

            if needs_optimization:
                logger.info(f"Auto-optimization triggered: {optimization_reason}")
                self._apply_dynamic_optimization(metrics)

    def _apply_dynamic_optimization(self, metrics: dict):
        """Apply dynamic optimizations based on performance metrics."""
        optimizations = []

        # Reduce batch size if memory usage is high
        if (
            metrics["avg_memory_usage"] > self.optimization_config.max_memory_usage * 800
        ):  # 80% threshold
            new_batch_size = max(1, int(self.optimal_batch_size * 0.8))
            self.optimal_batch_size = new_batch_size
            optimizations.append(f"reduced_batch_size_to_{new_batch_size}")

        # Adjust confidence threshold if inference time is high
        if metrics["avg_inference_time"] > self.optimization_config.max_inference_time * 0.8:
            new_conf = min(0.5, self.config["conf"] + 0.05)  # Increase confidence threshold
            self.config["conf"] = new_conf
            optimizations.append(f"increased_confidence_to_{new_conf:.2f}")

        # Switch to smaller model if performance is consistently poor
        if (
            metrics["avg_inference_time"] > self.optimization_config.max_inference_time * 1.5
            and self.config["model_size"] != "n"
        ):
            self._switch_to_smaller_model()
            optimizations.append("switched_to_smaller_model")

        if optimizations:
            logger.info(f"Applied dynamic optimizations: {optimizations}")

    def _switch_to_smaller_model(self):
        """Switch to a smaller model variant for better performance."""
        current_size = self.config["model_size"]
        size_hierarchy = ["l", "m", "s", "n"]

        try:
            current_idx = size_hierarchy.index(current_size)
            if current_idx < len(size_hierarchy) - 1:
                new_size = size_hierarchy[current_idx + 1]

                # Load smaller model if not already cached
                variant_key = f"yolov8{new_size}"
                if variant_key not in self.model_variants:
                    new_model = YOLO(f"yolov8{new_size}.pt")
                    new_model.to(self.device)
                    self.model_variants[variant_key] = new_model

                # Switch to smaller model
                self.model = self.model_variants[variant_key]
                self.config["model_size"] = new_size
                self.current_variant = variant_key

                logger.info(f"Switched to smaller model: YOLOv8{new_size}")

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")

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
        conf = conf or self.config["conf"]
        iou = iou or self.config["iou"]
        imgsz = imgsz or self.config["img_size"]
        half = half if half is not None else self.config["half"]
        device = device or self.device
        max_det = max_det or self.config["max_det"]

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
                        batch_results = super().predict(
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
                    results = super().predict(
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
                results = super().predict(
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
                    self.class_names[i] if i < len(self.class_names) else f"class_{i}"
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
                    model_config=self.config.copy(),
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
                    model_config=self.config.copy(),
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
                model_config=self.config.copy(),
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
            return self.config["batch_size"]

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
                "model_size": self.config["model_size"],
                "current_variant": self.current_variant,
                "hardware_tier": self.hardware.tier.name,
                "device": self.device,
                "optimizations_applied": getattr(self, "applied_optimizations", []),
            },
            "performance_metrics": metrics,
            "configuration": {
                "batch_size": self.optimal_batch_size,
                "confidence_threshold": self.config["conf"],
                "iou_threshold": self.config["iou"],
                "input_size": self.config["img_size"],
                "half_precision": self.config["half"],
            },
            "optimization_config": {
                "dynamic_switching_enabled": self.optimization_config.enable_dynamic_switching,
                "adaptive_batch_size_enabled": self.optimization_config.enable_adaptive_batch_size,
                "auto_optimization_enabled": self.optimization_config.enable_auto_optimization,
                "performance_monitoring_enabled": self.optimization_config.enable_performance_monitoring,
            },
            "model_variants_loaded": list(self.model_variants.keys()),
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
            "model_config": self.config.copy(),
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
        logger.info(f"- Hardware Tier: {self.hardware.tier.name}")
        logger.info(f"- Model Size: YOLOv8{self.config['model_size']}")
        logger.info(f"- Input Size: {self.config['img_size']}px")
        logger.info(f"- Batch Size: {self.optimal_batch_size}")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Half Precision: {self.config['half']}")
        logger.info(f"- Classes: {len(self.class_names)}")
        logger.info(f"- Optimizations: {getattr(self, 'applied_optimizations', [])}")
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
            "hardware_tier": self.hardware.tier.name,
            "optimal_batch_size": self.optimal_batch_size,
            "model_size": self.config["model_size"],
            "input_size": self.config["img_size"],
            "device": self.device,
            "half_precision": self.config["half"],
            "current_variant": self.current_variant,
            "available_variants": list(self.model_variants.keys()),
            "optimizations_applied": getattr(self, "applied_optimizations", []),
        }

        if hasattr(self.model, "model") and hasattr(self.model.model, "parameters"):
            stats["model_parameters"] = sum(p.numel() for p in self.model.model.parameters())

        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            stats.update(memory_stats)

        return stats


# Aliases for compatibility
CustomYOLOv8 = EnhancedYOLOv8
# Aliases for clarity and backward compatibility
YOLOv8Model = EnhancedYOLOv8
SpygateYOLO = EnhancedYOLOv8
CustomYOLO11 = EnhancedYOLOv8  # Legacy compatibility - remove in future versions


def load_optimized_yolov8_model(
    model_path: Optional[str] = None,
    hardware: Optional[HardwareDetector] = None,
    optimization_config: Optional[OptimizationConfig] = None,
) -> EnhancedYOLOv8:
    """Load and configure optimized YOLOv8 model with advanced features."""
    return EnhancedYOLOv8(
        model_path=model_path, hardware=hardware, optimization_config=optimization_config
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
