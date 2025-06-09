"""Customized YOLOv8 model architecture for HUD element detection with ultralytics and advanced GPU memory management."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from ultralytics import YOLO

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    YOLO = None

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

# UI Classes for HUD detection
UI_CLASSES = [
    "score_bug",  # Main score display with team names, scores, timeouts
    "down_distance",  # Down and distance indicator (e.g., "1st & 10")
    "game_clock",  # Game time remaining
    "play_clock",  # Play clock countdown
    "field_position",  # Yard line or field position marker
    "possession_indicator",  # Shows which team has possession
    "timeout_indicators",  # Team timeout indicators
    "flag_indicator",  # Penalty flag indicator
    "replay_indicator",  # Replay or review indicator
    "weather_info",  # Weather conditions (if visible)
]

# Hardware-tier specific model configurations
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
        },
    }
    if TORCH_AVAILABLE and HardwareTier
    else {}
)


@dataclass
class DetectionResult:
    """Structured detection result."""

    boxes: np.ndarray  # Bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # Confidence scores
    classes: np.ndarray  # Class indices
    class_names: list[str]  # Class names
    processing_time: float  # Time taken for detection
    memory_usage: Optional[dict] = None  # Memory usage statistics


class CustomYOLOv8(YOLO if TORCH_AVAILABLE else object):
    """Customized YOLOv8 model for HUD element and text detection with advanced GPU memory management."""

    def __init__(
        self, model_path: Optional[str] = None, hardware: Optional[HardwareDetector] = None
    ):
        """Initialize the CustomYOLOv8 model with hardware-aware settings."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for YOLOv8 functionality")

        # Initialize hardware detection
        self.hardware = hardware or HardwareDetector()
        self.config = MODEL_CONFIGS.get(self.hardware.tier, MODEL_CONFIGS[HardwareTier.LOW])

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
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # Fallback to smallest model
            super().__init__("yolov8n.pt")
            logger.info("Loaded fallback YOLOv8n model")

        # Setup memory management
        self._setup_memory_management()

        # Configure device
        self._setup_device()

        # Store class names
        self.class_names = UI_CLASSES

        logger.info(f"YOLOv8 model initialized for {self.hardware.tier.name} hardware")

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
        """Configure the device for inference."""
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


# For backward compatibility with existing imports
CustomYOLO11 = CustomYOLOv8


def load_yolov8_model(
    model_path: Optional[str] = None, hardware: Optional[HardwareDetector] = None
) -> CustomYOLOv8:
    """Load and configure YOLOv8 model with hardware-aware settings."""
    return CustomYOLOv8(model_path=model_path, hardware=hardware)


def get_hardware_optimized_config(hardware_tier: HardwareTier) -> dict:
    """Get hardware-optimized configuration for YOLOv8."""
    return MODEL_CONFIGS.get(hardware_tier, MODEL_CONFIGS[HardwareTier.LOW])
