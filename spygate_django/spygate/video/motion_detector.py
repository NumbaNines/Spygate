"""
Motion detection module for analyzing gameplay video frames.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetectionMethod(Enum):
    """Enum for different motion detection methods"""

    FRAME_DIFFERENCING = "frame_differencing"
    BACKGROUND_SUBTRACTION = "background_subtraction"
    OPTICAL_FLOW = "optical_flow"


@dataclass
class MotionDetectionResult:
    """Data class to store motion detection results"""

    motion_detected: bool
    motion_mask: np.ndarray  # Binary mask showing motion areas
    bounding_boxes: list[tuple[int, int, int, int]]  # List of (x, y, w, h) for motion regions
    metadata: dict[str, Any]  # Additional detection metadata


class MotionDetector:
    """Base class for motion detection in video frames"""

    def __init__(
        self,
        method: MotionDetectionMethod = MotionDetectionMethod.FRAME_DIFFERENCING,
        threshold: int = 30,
        min_area: int = 500,
        blur_size: int = 21,
        history: int = 500,
        var_threshold: float = 16.0,
        learning_rate: float = 0.01,
        max_corners: int = 100,
        quality_level: float = 0.3,
        min_distance: int = 7,
        block_size: int = 7,
        use_gpu: bool = True,
        num_threads: int = 4,
    ):
        """Initialize the motion detector with specified parameters.

        Args:
            method: Motion detection method to use
            threshold: Threshold for motion detection
            min_area: Minimum contour area to consider as motion
            blur_size: Gaussian blur kernel size
            history: History length for background subtractor
            var_threshold: Variance threshold for background subtractor
            learning_rate: Learning rate for background subtractor
            max_corners: Maximum number of corners to track for optical flow
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
            block_size: Block size for corner detection
            use_gpu: Whether to use GPU acceleration if available
            num_threads: Number of threads for parallel processing
        """
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.history = history
        self.var_threshold = var_threshold
        self.learning_rate = learning_rate
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # Initialize state variables
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None

        # GPU acceleration setup
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            try:
                self.gpu_stream = cv2.cuda.Stream()
                self.gpu_detector = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=self.history,
                    varThreshold=self.var_threshold,
                    detectShadows=False,
                )
                logger.info("GPU acceleration enabled for motion detection")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
                self.use_gpu = False

        # Multi-threading setup
        self.num_threads = num_threads
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        self.thread_local = threading.local()

        # Initialize background subtractor based on GPU availability
        if not self.use_gpu:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False,
            )

        # Create kernels
        self.dilate_kernel = np.ones((3, 3), np.uint8)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the search window
            maxLevel=2,  # Number of pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        logger.info(f"Initialized MotionDetector with method: {method.value}")

    def _get_thread_local_subtractor(self) -> cv2.BackgroundSubtractor:
        """Get or create a thread-local background subtractor."""
        if not hasattr(self.thread_local, "bg_subtractor"):
            self.thread_local.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False,
            )
        return self.thread_local.bg_subtractor

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with GPU acceleration if available."""
        if self.use_gpu:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur on GPU
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (self.blur_size, self.blur_size), 0)

            # Download result back to CPU
            return gpu_blurred.download()
        else:
            # CPU processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

    def _process_region(self, region: tuple[slice, slice], frame: np.ndarray) -> np.ndarray:
        """Process a region of the frame for parallel processing."""
        y_slice, x_slice = region
        region_frame = frame[y_slice, x_slice]

        if self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
            # Use thread-local background subtractor
            subtractor = self._get_thread_local_subtractor()
            return subtractor.apply(region_frame)

        return self.preprocess_frame(region_frame)

    def detect_frame_difference(self, frame: np.ndarray) -> MotionDetectionResult:
        """Detect motion using frame differencing method.

        Args:
            frame: Current video frame

        Returns:
            MotionDetectionResult containing detection data
        """
        # Preprocess current frame
        current_gray = self.preprocess_frame(frame)

        # Initialize result with no motion if this is the first frame
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return MotionDetectionResult(
                motion_detected=False,
                motion_mask=np.zeros_like(current_gray),
                bounding_boxes=[],
                metadata={"method": self.method.value},
            )

        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(current_gray, self.prev_gray)

        # Apply threshold to get binary motion mask
        _, motion_mask = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Apply dilation to connect nearby motion regions
        motion_mask = cv2.dilate(motion_mask, self.dilate_kernel, iterations=1)

        # Find contours in the motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and get bounding boxes
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        # Calculate motion score (ratio of motion pixels to total pixels)
        motion_score = np.count_nonzero(motion_mask) / motion_mask.size

        # Create annotated frame
        frame_processed = frame.copy()
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update previous frame
        self.prev_gray = current_gray

        return MotionDetectionResult(
            motion_detected=len(bounding_boxes) > 0,
            motion_mask=motion_mask,
            bounding_boxes=bounding_boxes,
            metadata={
                "method": self.method.value,
                "num_regions": len(bounding_boxes),
                "threshold": self.threshold,
            },
        )

    def detect_background_subtraction(self, frame: np.ndarray) -> MotionDetectionResult:
        """Detect motion using background subtraction method.

        Args:
            frame: Current video frame

        Returns:
            MotionDetectionResult containing detection data
        """
        # Apply background subtraction
        motion_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

        # Apply dilation to connect nearby motion regions
        motion_mask = cv2.dilate(motion_mask, self.dilate_kernel, iterations=1)

        # Find contours in the motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and get bounding boxes
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        # Calculate motion score (ratio of motion pixels to total pixels)
        motion_score = np.count_nonzero(motion_mask) / motion_mask.size

        # Create annotated frame
        frame_processed = frame.copy()
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return MotionDetectionResult(
            motion_detected=len(bounding_boxes) > 0,
            motion_mask=motion_mask,
            bounding_boxes=bounding_boxes,
            metadata={
                "method": self.method.value,
                "num_regions": len(bounding_boxes),
                "learning_rate": self.learning_rate,
                "threshold": self.threshold,
            },
        )

    def detect_optical_flow(self, frame: np.ndarray) -> MotionDetectionResult:
        """Detect motion using optical flow method.

        Args:
            frame: Current video frame

        Returns:
            MotionDetectionResult containing detection data
        """
        # Convert current frame to grayscale
        current_gray = self.preprocess_frame(frame)

        # Initialize result with no motion if this is the first frame
        if self.prev_gray is None:
            self.prev_gray = current_gray
            self.prev_points = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
            )
            return MotionDetectionResult(
                motion_detected=False,
                motion_mask=np.zeros_like(current_gray),
                bounding_boxes=[],
                metadata={"method": self.method.value},
            )

        # Calculate optical flow if we have previous points
        motion_mask = np.zeros_like(current_gray)
        bounding_boxes = []
        if self.prev_points is not None and len(self.prev_points) > 0:
            # Calculate optical flow
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, current_gray, self.prev_points, None, **self.lk_params
            )

            # Filter out points where flow wasn't found
            if current_points is not None:
                good_current = current_points[status == 1]
                good_prev = self.prev_points[status == 1]

                # Calculate flow vectors
                flow_vectors = good_current - good_prev
                magnitudes = np.sqrt(flow_vectors[:, 0] ** 2 + flow_vectors[:, 1] ** 2)

                # Filter significant motion
                significant_motion = magnitudes > self.threshold
                motion_points = good_current[significant_motion].astype(np.int32)

                if len(motion_points) > 0:
                    # Create motion mask from significant motion points
                    for pt in motion_points:
                        cv2.circle(motion_mask, (pt[0], pt[1]), 5, 255, -1)

                    # Apply dilation to connect nearby motion regions
                    motion_mask = cv2.dilate(motion_mask, self.dilate_kernel, iterations=2)

                    # Find contours in the motion mask
                    contours, _ = cv2.findContours(
                        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Filter contours by area and get bounding boxes
                    for contour in contours:
                        if cv2.contourArea(contour) >= self.min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            bounding_boxes.append((x, y, w, h))

        # Calculate motion score
        motion_score = np.count_nonzero(motion_mask) / motion_mask.size

        # Create annotated frame
        frame_processed = frame.copy()

        # Draw motion vectors and bounding boxes
        if self.prev_points is not None and len(self.prev_points) > 0:
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(frame_processed, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw motion vectors
            if "good_current" in locals() and "good_prev" in locals():
                for curr, prev in zip(good_current, good_prev):
                    curr_pt = tuple(map(int, curr.ravel()))
                    prev_pt = tuple(map(int, prev.ravel()))
                    cv2.arrowedLine(
                        frame_processed, prev_pt, curr_pt, (0, 0, 255), 2, tipLength=0.5
                    )

        # Update previous frame and points
        self.prev_gray = current_gray
        if len(bounding_boxes) > 0:
            # If motion detected, find new points to track
            self.prev_points = cv2.goodFeaturesToTrack(
                current_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
            )
        else:
            # If no motion, keep tracking the same points
            self.prev_points = current_points

        return MotionDetectionResult(
            motion_detected=len(bounding_boxes) > 0,
            motion_mask=motion_mask,
            bounding_boxes=bounding_boxes,
            metadata={
                "method": self.method.value,
                "num_regions": len(bounding_boxes),
                "num_tracked_points": (
                    len(self.prev_points) if self.prev_points is not None else 0
                ),
                "threshold": self.threshold,
            },
        )

    def detect_motion(self, frame: np.ndarray) -> MotionDetectionResult:
        """Detect motion using the configured detection method with GPU and multi-threading."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided")

        if self.method == MotionDetectionMethod.FRAME_DIFFERENCING:
            return self.detect_frame_difference(frame)
        elif self.method == MotionDetectionMethod.BACKGROUND_SUBTRACTION:
            return self.detect_background_subtraction(frame)
        elif self.method == MotionDetectionMethod.OPTICAL_FLOW:
            return self.detect_optical_flow(frame)
        else:
            raise ValueError(f"Unsupported motion detection method: {self.method}")

    def reset(self):
        """Reset the detector's state for all methods."""
        # Reset frame differencing
        self.prev_frame = None

        # Reset background subtraction
        if self.use_gpu:
            try:
                self.gpu_detector = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=self.history,
                    varThreshold=self.var_threshold,
                    detectShadows=False,
                )
            except Exception as e:
                logger.warning(f"Failed to reset GPU background subtractor: {e}")
                self.use_gpu = False
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=self.history,
                    varThreshold=self.var_threshold,
                    detectShadows=False,
                )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=False,
            )

        # Reset optical flow
        self.prev_gray = None
        self.prev_points = None

        # Reset thread-local storage
        self.thread_local = threading.local()

        logger.debug("Reset MotionDetector state")
