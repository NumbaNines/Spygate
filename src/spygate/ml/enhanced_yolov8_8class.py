"""
Enhanced YOLOv8 implementation with 8-class support and 5-class fallback.
Backward compatible with existing 5-class model.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class EnhancedYOLOv8_8Class:
    """
    Enhanced YOLOv8 implementation supporting both 5-class and 8-class models.
    Automatically detects model type and provides appropriate fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        prefer_8_class: bool = True,
        fallback_to_5_class: bool = True,
    ):
        """
        Initialize Enhanced YOLOv8 with multi-class support.

        Args:
            model_path: Path to model weights
            device: Device to run on ('cuda', 'cpu', or 'auto')
            prefer_8_class: Try to load 8-class model first
            fallback_to_5_class: Fall back to 5-class if 8-class fails
        """
        self.device = self._setup_device(device)
        self.prefer_8_class = prefer_8_class
        self.fallback_to_5_class = fallback_to_5_class

        # Model state
        self.model = None
        self.model_type = None  # '5-class' or '8-class'
        self.class_info = None

        # Performance tracking
        self.inference_times = []
        self.detection_counts = []

        # Load model with automatic detection
        self._load_model_with_fallback(model_path)

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup and validate device."""
        if device == "auto" or device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model_with_fallback(self, model_path: Optional[str]):
        """Load model with automatic 8-class/5-class detection and fallback."""

        # Define model paths to try
        model_paths_to_try = []

        if model_path:
            model_paths_to_try.append(model_path)

        # Add default paths based on preference
        if self.prefer_8_class:
            model_paths_to_try.extend(
                [
                    "hud_region_training_8class/runs/best_8class/weights/best.pt",
                    "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt",
                ]
            )
        else:
            model_paths_to_try.extend(
                [
                    "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt",
                    "hud_region_training_8class/runs/best_8class/weights/best.pt",
                ]
            )

        # Try loading each model
        for path in model_paths_to_try:
            try:
                if self._try_load_model(path):
                    logger.info(f"Successfully loaded {self.model_type} model from {path}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                continue

        # If all fails, raise error
        raise RuntimeError("Could not load any YOLOv8 model (5-class or 8-class)")

    def _try_load_model(self, model_path: str) -> bool:
        """Try to load a specific model and detect its type."""
        if not os.path.exists(model_path):
            return False

        # Load model
        model = YOLO(model_path)
        model.to(self.device)

        # Detect model type based on number of classes
        num_classes = len(model.names)

        if num_classes == 8:
            self.model_type = "8-class"
            self.class_info = self._get_8_class_info()
        elif num_classes == 5:
            self.model_type = "5-class"
            self.class_info = self._get_5_class_info()
        else:
            logger.warning(f"Unknown model type with {num_classes} classes")
            return False

        self.model = model
        logger.info(f"Loaded {self.model_type} model with {num_classes} classes")
        return True

    def _get_8_class_info(self) -> dict:
        """Get 8-class information."""
        return {
            "classes": [
                "hud",
                "possession_triangle_area",
                "territory_triangle_area",
                "preplay_indicator",
                "play_call_screen",
                "down_distance_area",
                "game_clock_area",
                "play_clock_area",
            ],
            "count": 8,
            "enhanced": True,
        }

    def _get_5_class_info(self) -> dict:
        """Get 5-class information."""
        return {
            "classes": [
                "hud",
                "possession_triangle_area",
                "territory_triangle_area",
                "preplay_indicator",
                "play_call_screen",
            ],
            "count": 5,
            "enhanced": False,
        }

    def detect(
        self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45
    ) -> list[dict]:
        """
        Detect HUD elements in image.

        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        try:
            # Run inference
            results = self.model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

            # Process results
            detections = []
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.model.names[class_id]

                        detection = {
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "class": class_name,
                            "class_id": int(class_id),
                        }
                        detections.append(detection)

            # Track performance
            self.detection_counts.append(len(detections))

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def get_supported_classes(self) -> list[str]:
        """Get list of supported class names."""
        if self.class_info:
            return self.class_info["classes"]
        elif self.model:
            return list(self.model.names.values())
        else:
            return []

    def is_8_class_model(self) -> bool:
        """Check if current model is 8-class."""
        return self.model_type == "8-class"

    def is_5_class_model(self) -> bool:
        """Check if current model is 5-class."""
        return self.model_type == "5-class"

    def has_enhanced_classes(self) -> bool:
        """Check if model supports enhanced classes (down/distance, clocks)."""
        return self.is_8_class_model()

    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        return {
            "model_type": self.model_type,
            "num_classes": len(self.get_supported_classes()),
            "supported_classes": self.get_supported_classes(),
            "device": self.device,
            "has_enhanced_detection": self.has_enhanced_classes(),
            "class_info": self.class_info,
        }

    def detect_with_fallback_mapping(
        self, image: np.ndarray, target_classes: Optional[list[str]] = None, **kwargs
    ) -> list[dict]:
        """
        Detect with automatic class mapping for backward compatibility.

        Args:
            image: Input image
            target_classes: Specific classes to detect (None for all)
            **kwargs: Additional detection parameters

        Returns:
            List of detections with consistent class names
        """
        detections = self.detect(image, **kwargs)

        # Filter by target classes if specified
        if target_classes:
            detections = [d for d in detections if d["class"] in target_classes]

        # Add compatibility flags
        for detection in detections:
            detection["model_type"] = self.model_type
            detection["enhanced_detection"] = self.has_enhanced_classes()

        return detections


# Backward compatibility alias
EnhancedYOLOv8 = EnhancedYOLOv8_8Class
