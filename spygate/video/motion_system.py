"""
Motion system module for integrating all motion detection components.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from ..ml.situation_detector import SituationDetector
from ..services.motion_service import MotionService
from ..utils.hardware_monitor import HardwareMonitor
from ..visualization.motion_visualizer import MotionVisualizer
from .motion_detector import MotionDetectionMethod, MotionDetector

logger = logging.getLogger(__name__)


class MotionSystem:
    """
    Integrates motion detection, analysis, visualization, and hardware monitoring.

    This class serves as the main interface for the motion detection subsystem,
    coordinating all components and providing a unified API for video processing.
    """

    def __init__(
        self,
        db_session: Session,
        frame_width: int,
        frame_height: int,
        detection_method: MotionDetectionMethod = MotionDetectionMethod.FRAME_DIFFERENCING,
        hardware_aware: bool = True,
        store_heatmaps: bool = True,
        store_patterns: bool = True,
        heatmap_interval: int = 300,
        enable_situation_detection: bool = True,
        enable_visualization: bool = True,
    ):
        """
        Initialize the motion system.

        Args:
            db_session: SQLAlchemy database session
            frame_width: Width of video frames
            frame_height: Height of video frames
            detection_method: Motion detection method to use
            hardware_aware: Whether to use hardware-aware processing
            store_heatmaps: Whether to store motion heatmaps
            store_patterns: Whether to store motion patterns
            heatmap_interval: Number of frames between heatmap aggregations
            enable_situation_detection: Whether to enable situation detection
            enable_visualization: Whether to enable visualization
        """
        # Initialize hardware monitoring
        self.hardware_monitor = HardwareMonitor()
        self.performance_tier = self.hardware_monitor.get_performance_tier()
        logger.info(f"System performance tier: {self.performance_tier}")

        # Initialize motion detector with hardware-aware settings
        detector_params = self._get_detector_params()
        self.motion_detector = MotionDetector(
            method=detection_method, **detector_params
        )

        # Initialize motion service
        self.motion_service = MotionService(
            db_session=db_session,
            hardware_aware=hardware_aware,
            store_heatmaps=store_heatmaps,
            store_patterns=store_patterns,
            heatmap_interval=heatmap_interval,
        )

        # Initialize situation detector if enabled
        self.situation_detector = None
        if enable_situation_detection:
            self.situation_detector = SituationDetector()

        # Initialize visualizer if enabled
        self.visualizer = None
        if enable_visualization:
            self.visualizer = MotionVisualizer(
                frame_width=frame_width, frame_height=frame_height
            )

        # Initialize state
        self.frame_count = 0
        self.processing_fps = 0
        self.current_video_id = None
        self._last_resource_check = 0
        self._resource_check_interval = 30  # frames

        logger.info("Motion system initialized successfully")

    def _get_detector_params(self) -> Dict[str, Any]:
        """Get motion detector parameters based on hardware tier."""
        params = {"use_gpu": False, "num_threads": 1, "frame_skip": 0}

        if self.performance_tier == "high":
            params.update({"use_gpu": True, "num_threads": 4, "frame_skip": 0})
        elif self.performance_tier == "medium":
            params.update({"use_gpu": True, "num_threads": 2, "frame_skip": 1})
        else:  # low tier
            params.update({"use_gpu": False, "num_threads": 1, "frame_skip": 2})

        return params

    def process_frame(
        self,
        frame: np.ndarray,
        video_id: int,
        frame_number: int,
        fps: float,
        return_visualization: bool = True,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Process a video frame through the motion detection pipeline.

        Args:
            frame: The video frame to process
            video_id: ID of the video being processed
            frame_number: Current frame number
            fps: Frames per second of the video
            return_visualization: Whether to return the visualized frame

        Returns:
            Tuple containing:
            - Visualized frame (if return_visualization is True)
            - Dictionary with detection results
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")

        # Update video ID if changed
        if self.current_video_id != video_id:
            self._handle_video_change(video_id)

        # Check system resources periodically
        if self.frame_count % self._resource_check_interval == 0:
            self._check_resources()

        # Detect motion
        motion_result = self.motion_detector.detect_motion(frame)

        # Detect situations if enabled
        situations = None
        if self.situation_detector:
            situations = self.situation_detector.detect_situations(
                frame, frame_number, fps
            )

        # Process results through motion service
        vis_frame, results = self.motion_service.process_frame(
            frame=frame, video_id=video_id, frame_number=frame_number, fps=fps
        )

        # Add hardware monitoring data
        results["hardware"] = {
            "tier": self.performance_tier,
            "cpu_usage": self.hardware_monitor.get_cpu_utilization(),
            "memory_usage": self.hardware_monitor.get_memory_usage(),
            "gpu_usage": self.hardware_monitor.get_gpu_utilization(),
            "processing_fps": self.processing_fps,
        }

        # Update state
        self.frame_count += 1

        # Return results
        if return_visualization and self.visualizer:
            return (
                self.visualizer.update(
                    frame=vis_frame,
                    motion_result=motion_result,
                    situations=situations["situations"] if situations else None,
                ),
                results,
            )
        else:
            return None, results

    def _handle_video_change(self, video_id: int):
        """Handle state reset when processing a new video."""
        self.current_video_id = video_id
        self.frame_count = 0
        self.motion_detector.reset()
        if self.situation_detector:
            self.situation_detector.initialize()
        if self.visualizer:
            self.visualizer.reset()
        logger.info(f"Started processing new video: {video_id}")

    def _check_resources(self):
        """Check system resources and adjust processing if needed."""
        cpu_usage = self.hardware_monitor.get_cpu_utilization()
        memory_usage = self.hardware_monitor.get_memory_usage()
        gpu_usage = self.hardware_monitor.get_gpu_utilization()

        # Log resource usage
        logger.debug(
            f"Resource usage - CPU: {cpu_usage}%, Memory: {memory_usage}%, "
            f"GPU: {gpu_usage}% (if available)"
        )

        # Adjust processing based on resource usage
        if cpu_usage > 90 or memory_usage > 90:
            logger.warning("High resource usage detected, adjusting processing")
            self._adjust_processing_params()

    def _adjust_processing_params(self):
        """Adjust processing parameters based on resource usage."""
        current_params = self.motion_detector.get_config()

        # Increase frame skip
        if current_params.get("frame_skip", 0) < 3:
            self.motion_detector.frame_skip += 1
            logger.info(f"Increased frame skip to {self.motion_detector.frame_skip}")

        # Reduce thread count if using multiple threads
        if current_params.get("num_threads", 1) > 1:
            self.motion_detector.num_threads -= 1
            logger.info(f"Reduced thread count to {self.motion_detector.num_threads}")

    def get_motion_events(
        self,
        video_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_confidence: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve motion events for a video within a time range.

        Args:
            video_id: ID of the video
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            min_confidence: Minimum confidence for situations

        Returns:
            List of motion events with their situations
        """
        return self.motion_service.get_motion_events(
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            min_confidence=min_confidence,
        )

    def get_motion_patterns(
        self,
        video_id: int,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve motion patterns for a video.

        Args:
            video_id: ID of the video
            pattern_type: Type of patterns to retrieve (optional)
            min_confidence: Minimum confidence for patterns

        Returns:
            List of motion patterns
        """
        return self.motion_service.get_motion_patterns(
            video_id=video_id, pattern_type=pattern_type, min_confidence=min_confidence
        )

    def get_motion_heatmap(
        self,
        video_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve motion heatmap for a video time range.

        Args:
            video_id: ID of the video
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)

        Returns:
            Motion heatmap as numpy array, or None if not available
        """
        return self.motion_service.get_motion_heatmap(
            video_id=video_id, start_time=start_time, end_time=end_time
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information and status."""
        return {
            "hardware_tier": self.performance_tier,
            "hardware_info": self.hardware_monitor.get_system_info(),
            "current_video": self.current_video_id,
            "frame_count": self.frame_count,
            "processing_fps": self.processing_fps,
            "detector_config": self.motion_detector.get_config(),
            "resource_usage": {
                "cpu": self.hardware_monitor.get_cpu_utilization(),
                "memory": self.hardware_monitor.get_memory_usage(),
                "gpu": self.hardware_monitor.get_gpu_utilization(),
            },
        }
