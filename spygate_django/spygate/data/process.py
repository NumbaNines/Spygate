"""
Video processing with hardware-aware optimizations for SpygateAI.
"""

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Hardware-aware video processing with adaptive performance optimization."""

    def __init__(self, delay_seconds: int = 30):
        """Initialize the video processor.

        Args:
            delay_seconds: Delay in seconds for live analysis (default: 30)
        """
        self.delay_seconds = delay_seconds
        self.frame_count = 0
        self.last_frame_time = 0

        # Initialize hardware detection and optimization
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        logger.info(f"Initialized VideoProcessor with {self.hardware.tier.name} tier")
        logger.info(f"Processing parameters: {self.optimizer.get_current_params()}")

    def frame_generator(self, video_path: str) -> Generator[tuple[np.ndarray, dict], None, None]:
        """Generate processed frames from a video file with hardware-aware optimization.

        Args:
            video_path: Path to the video file

        Yields:
            Tuple of (processed frame, metrics dictionary)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            while True:
                # Enforce delay for live analysis
                current_time = time.time()
                if (
                    current_time - self.last_frame_time
                    < 1.0 / self.optimizer.current_params.target_fps
                ):
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Skip frames based on optimizer settings
                if not self.optimizer.should_process_frame(self.frame_count):
                    continue

                # Process frame with hardware-aware optimization
                start_time = time.time()
                processed_frame, metrics = self.optimizer.optimize_frame(frame)

                # Update performance metrics
                processing_time = time.time() - start_time
                self.optimizer.update_performance(processing_time)

                # Update metrics with performance data
                metrics.update(self.optimizer.get_performance_metrics())
                metrics["processing_time"] = processing_time

                self.last_frame_time = current_time
                yield processed_frame, metrics

        finally:
            cap.release()

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """Process a video file with hardware-aware optimization.

        Args:
            video_path: Path to the input video file
            output_path: Optional path to save the processed video
        """
        if output_path:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            out = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        try:
            for frame, metrics in self.frame_generator(video_path):
                if output_path:
                    out.write(frame)

                # Log performance metrics periodically
                if self.frame_count % 30 == 0:
                    logger.info(f"Processing metrics: {metrics}")

        finally:
            if output_path:
                out.release()

    def get_hardware_tier(self) -> str:
        """Get the current hardware tier name."""
        return self.hardware.tier.name

    def get_performance_metrics(self) -> dict[str, float]:
        """Get current performance metrics."""
        return self.optimizer.get_performance_metrics()

    def get_available_features(self) -> dict[str, bool]:
        """Get available features for current hardware tier."""
        return self.hardware.get_tier_features()
