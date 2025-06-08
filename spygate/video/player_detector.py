"""
Player detection module using various detection algorithms.

This module provides functionality to detect players in video frames using
different detection methods, including CNN-based models and traditional
computer vision techniques.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

from ..utils.tracking_hardware import TrackingHardwareManager, TrackingMode

logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    Detects players in video frames using various detection methods.

    This class provides multiple detection algorithms and automatically selects
    the best one based on hardware capabilities and performance requirements.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the player detector.

        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.hardware_manager = TrackingHardwareManager()

        # Initialize detection models based on hardware capabilities
        self.tracking_mode = self.hardware_manager.tracking_mode
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize detection models based on hardware capabilities."""
        # Always initialize HOG detector as fallback
        self.models["hog"] = cv2.HOGDescriptor()
        self.models["hog"].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Initialize deep learning models if hardware supports it
        if self.tracking_mode in [
            TrackingMode.STANDARD,
            TrackingMode.ADVANCED,
            TrackingMode.PROFESSIONAL,
        ]:
            try:
                # Initialize Faster R-CNN
                self.models["frcnn"] = fasterrcnn_resnet50_fpn(pretrained=True)
                if torch.cuda.is_available():
                    self.models["frcnn"] = self.models["frcnn"].cuda()
                self.models["frcnn"].eval()

                # Initialize YOLOv5 if available
                try:
                    import torch.hub

                    self.models["yolo"] = torch.hub.load(
                        "ultralytics/yolov5", "yolov5s", pretrained=True
                    )
                    if torch.cuda.is_available():
                        self.models["yolo"] = self.models["yolo"].cuda()
                except Exception as e:
                    logger.warning(f"Could not initialize YOLOv5: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize deep learning models: {e}")

    def detect_players(
        self, frame: np.ndarray, method: str = "auto"
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect players in a video frame.

        Args:
            frame: Input video frame (BGR format)
            method: Detection method ('hog', 'frcnn', 'yolo', or 'auto')

        Returns:
            List of dictionaries containing detection results:
            {
                'bbox': np.ndarray,  # [x1, y1, x2, y2]
                'confidence': float,  # Detection confidence score
                'class': str  # Object class (e.g., 'person')
            }
        """
        if method == "auto":
            method = self._select_detection_method()

        if method not in self.models:
            logger.warning(f"Detection method {method} not available. Using HOG.")
            method = "hog"

        if method == "hog":
            return self._detect_hog(frame)
        elif method == "frcnn":
            return self._detect_frcnn(frame)
        elif method == "yolo":
            return self._detect_yolo(frame)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _select_detection_method(self) -> str:
        """Select the best detection method based on hardware capabilities."""
        if self.tracking_mode in [TrackingMode.ADVANCED, TrackingMode.PROFESSIONAL]:
            return "yolo" if "yolo" in self.models else "frcnn"
        elif self.tracking_mode == TrackingMode.STANDARD:
            return "frcnn"
        else:
            return "hog"

    def _detect_hog(
        self, frame: np.ndarray
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect players using HOG detector.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detection results
        """
        # Convert to grayscale for HOG
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect people
        boxes, weights = self.models["hog"].detectMultiScale(
            gray, winStride=(8, 8), padding=(4, 4), scale=1.05
        )

        # Convert results to standard format
        detections = []
        for box, confidence in zip(boxes, weights):
            if confidence > self.confidence_threshold:
                x, y, w, h = box
                detections.append(
                    {
                        "bbox": np.array([x, y, x + w, y + h]),
                        "confidence": float(confidence),
                        "class": "person",
                    }
                )

        return detections

    def _detect_frcnn(
        self, frame: np.ndarray
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect players using Faster R-CNN.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        tensor = F.to_tensor(rgb_frame)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        # Get predictions
        with torch.no_grad():
            predictions = self.models["frcnn"]([tensor])[0]

        # Convert results to standard format
        detections = []
        for box, score, label in zip(
            predictions["boxes"], predictions["scores"], predictions["labels"]
        ):
            if (
                score > self.confidence_threshold and label == 1
            ):  # 1 is person class in COCO
                if torch.cuda.is_available():
                    box = box.cpu()
                detections.append(
                    {"bbox": box.numpy(), "confidence": float(score), "class": "person"}
                )

        return detections

    def _detect_yolo(
        self, frame: np.ndarray
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Detect players using YOLOv5.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List of detection results
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get predictions
        results = self.models["yolo"](rgb_frame)

        # Convert results to standard format
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            if (
                conf > self.confidence_threshold and cls == 0
            ):  # 0 is person class in COCO
                if torch.cuda.is_available():
                    box = [b.cpu() for b in box]
                detections.append(
                    {
                        "bbox": np.array(box),
                        "confidence": float(conf),
                        "class": "person",
                    }
                )

        return detections

    def get_detection_info(self) -> Dict:
        """Get information about available detection methods."""
        return {
            "tracking_mode": self.tracking_mode,
            "available_methods": list(self.models.keys()),
            "gpu_available": torch.cuda.is_available(),
            "confidence_threshold": self.confidence_threshold,
        }
