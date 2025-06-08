"""
Frame preprocessing module for Spygate.

Handles frame normalization, resizing, and ROI selection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer


@dataclass
class PreprocessingConfig:
    """Configuration for frame preprocessing."""

    target_size: Tuple[int, int] = (1280, 720)  # Default to 720p
    normalize_colors: bool = True
    extract_roi: bool = True
    use_gpu: bool = False
    roi_regions: Optional[List[Tuple[int, int, int, int]]] = None  # [(x, y, w, h), ...]


class FramePreprocessor:
    """Handles frame preprocessing operations."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the frame preprocessor."""
        self.config = config or PreprocessingConfig()
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize GPU context if available and needed
        self.use_gpu = (
            self.config.use_gpu
            and self.hardware.has_cuda
            and self.hardware.has_opencv_gpu
        )

        if self.use_gpu:
            self.gpu_stream = cv2.cuda.Stream()
            self.gpu_resizer = cv2.cuda.createResize(
                self.config.target_size[::-1],  # OpenCV uses (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

        # Pre-compute normalization parameters
        self.norm_scale = 1.0 / 255.0
        self.norm_mean = np.array([0.485, 0.456, 0.406])
        self.norm_std = np.array([0.229, 0.224, 0.225])

    def preprocess_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess a frame with configured operations.

        Args:
            frame: Input frame as numpy array (HxWxC)

        Returns:
            Dict containing processed frames and ROIs
        """
        results = {"full_frame": frame}

        # Resize frame
        if frame.shape[:2] != self.config.target_size:
            frame = self._resize_frame(frame)
            results["full_frame"] = frame

        # Extract ROIs if configured
        if self.config.extract_roi and self.config.roi_regions:
            rois = self._extract_rois(frame)
            results["rois"] = rois

        # Normalize colors if configured
        if self.config.normalize_colors:
            norm_frame = self._normalize_colors(frame)
            results["normalized"] = norm_frame

            if "rois" in results:
                norm_rois = [self._normalize_colors(roi) for roi in results["rois"]]
                results["normalized_rois"] = norm_rois

        return results

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size."""
        if self.use_gpu and self.optimizer.should_use_gpu("resize"):
            # GPU resize
            gpu_frame = cv2.cuda_GpuMat(frame)
            resized_gpu = self.gpu_resizer.apply(gpu_frame, stream=self.gpu_stream)
            return resized_gpu.download()
        else:
            # CPU resize
            return cv2.resize(
                frame,
                self.config.target_size[::-1],  # OpenCV uses (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

    def _extract_rois(self, frame: np.ndarray) -> List[np.ndarray]:
        """Extract regions of interest from frame."""
        rois = []
        for x, y, w, h in self.config.roi_regions:
            roi = frame[y : y + h, x : x + w].copy()
            rois.append(roi)
        return rois

    def _normalize_colors(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame colors."""
        if self.use_gpu and self.optimizer.should_use_gpu("normalize"):
            # GPU normalization
            gpu_frame = cv2.cuda_GpuMat(frame)
            gpu_frame = gpu_frame.convertTo(None, cv2.CV_32F, self.norm_scale)
            normalized = gpu_frame.download()
        else:
            # CPU normalization
            normalized = frame.astype(np.float32) * self.norm_scale

        # Apply ImageNet normalization
        for i in range(3):  # RGB channels
            normalized[..., i] = (
                normalized[..., i] - self.norm_mean[i]
            ) / self.norm_std[i]

        return normalized

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for preprocessing."""
        return self.optimizer.get_operation_batch_size("transform")

    def preprocess_batch(self, frames: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Preprocess a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            Dict containing lists of processed frames and ROIs
        """
        results = {
            "full_frames": [],
            "normalized": [] if self.config.normalize_colors else None,
            "rois": [] if self.config.extract_roi else None,
            "normalized_rois": (
                []
                if (self.config.normalize_colors and self.config.extract_roi)
                else None
            ),
        }

        # Process frames in parallel if possible
        for frame in frames:
            processed = self.preprocess_frame(frame)
            results["full_frames"].append(processed["full_frame"])

            if self.config.normalize_colors:
                results["normalized"].append(processed["normalized"])

            if self.config.extract_roi:
                results["rois"].extend(processed["rois"])

                if self.config.normalize_colors:
                    results["normalized_rois"].extend(processed["normalized_rois"])

        return {k: v for k, v in results.items() if v is not None}

    def cleanup(self):
        """Clean up GPU resources."""
        if self.use_gpu:
            self.gpu_stream.free()
            cv2.cuda.resetDevice()

    def crop_frame(self, frame, region=((0, 100), (0, 100))):
        """Crop the frame to the specified region."""
        return frame[region[0][0]:region[0][1], region[1][0]:region[1][1]]
