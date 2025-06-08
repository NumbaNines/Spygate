"""
Video tracking pipeline module.

This module integrates object tracking with the video processing pipeline,
providing a unified interface for video input, frame processing, object tracking,
and formation analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numba import cuda, jit

from ..core.game_detector import GameVersion
from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from ..utils.tracking_hardware import TrackingHardwareManager, TrackingMode
from .algorithm_selector import AlgorithmSelector, SceneComplexity
from .formation_analyzer import FormationAnalyzer, FormationConfig
from .frame_extractor import FrameExtractor, PreprocessingConfig
from .frame_preprocessor import FramePreprocessor
from .motion_system import MotionSystem
from .object_tracker import MultiObjectTracker, ObjectTracker, TrackingConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the tracking pipeline."""

    preprocessing_config: Optional[PreprocessingConfig] = None
    tracking_config: Optional[TrackingConfig] = None
    formation_config: Optional[FormationConfig] = None
    cache_dir: Optional[str] = None
    use_motion_detection: bool = True
    batch_size: int = 16  # Increased for better GPU utilization
    enable_visualization: bool = True
    max_frame_buffer: int = 60  # Increased for smoother FPS calculation
    target_fps: float = 30.0
    min_fps: float = 20.0
    quality_step: float = 0.1
    min_quality: float = 0.5
    max_quality: float = 1.0
    max_memory_usage: float = 2048.0  # Maximum memory usage in MB
    thread_pool_size: int = 4  # Number of worker threads
    enable_prefetch: bool = True  # Enable frame prefetching
    prefetch_size: int = 4  # Number of frames to prefetch


class TrackingPipeline:
    """Integrates video processing with object tracking and analysis."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the tracking pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize hardware-aware components
        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.tracking_hw = TrackingHardwareManager()

        # Initialize pipeline components with hardware optimization
        self.frame_extractor = FrameExtractor(
            preprocessing_config=self.config.preprocessing_config,
            cache_dir=self.config.cache_dir,
        )
        self.preprocessor = FramePreprocessor(self.config.preprocessing_config)
        self.algorithm_selector = AlgorithmSelector()
        self.motion_system = (
            MotionSystem() if self.config.use_motion_detection else None
        )
        self.object_tracker = MultiObjectTracker(
            config=self.config.tracking_config,
            hardware_manager=self.tracking_hw,
        )
        self.formation_analyzer = FormationAnalyzer(config=self.config.formation_config)

        # Initialize state
        self.frame_count = 0
        self.current_video_id = None
        self.scene_complexity = SceneComplexity.LOW
        self.tracking_mode = TrackingMode.STANDARD
        self.fps = 30.0
        
        # Performance monitoring
        self.frame_times = deque(maxlen=self.config.max_frame_buffer)
        self.current_quality = self.config.max_quality
        self.last_quality_check = time.time()
        self.quality_check_interval = 1.0  # Check every second
        
        # Frame buffer for batch processing
        self.frame_buffer = []
        self.result_buffer = {}
        
        # Memory management
        self.memory_usage = 0.0
        self.last_memory_check = time.time()
        self.memory_check_interval = 5.0  # Check every 5 seconds
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Frame prefetching
        if self.config.enable_prefetch:
            self.prefetch_queue = deque(maxlen=self.config.prefetch_size)
            self.prefetch_thread = threading.Thread(target=self._prefetch_frames, daemon=True)
            self.prefetch_thread.start()

        # Initialize GPU context if available
        self._init_gpu()

        logger.info(
            f"Initialized TrackingPipeline with {self.hardware.tier.name} tier"
        )
        logger.info(f"Processing parameters: {self.optimizer.get_current_params()}")

    def _init_gpu(self):
        """Initialize GPU context if available."""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(0)
                self.use_gpu = True
                # Preallocate GPU memory for common operations
                self.gpu_memory_pool = cv2.cuda.MemoryPool()
                cv2.cuda.setBufferPoolConfig(
                    cv2.cuda.MemoryPool.CIRCULAR_BUFFER_POOL,
                    self.config.batch_size * 3,  # Buffer size for batch processing
                )
                logger.info("GPU acceleration enabled with memory pool")
            else:
                self.use_gpu = False
                logger.info("No GPU available, using CPU")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU: {e}")
            self.use_gpu = False

    def _adjust_quality(self):
        """Adjust processing quality based on performance metrics."""
        if len(self.frame_times) < self.config.max_frame_buffer // 2:
            return

        current_time = time.time()
        if current_time - self.last_quality_check < self.quality_check_interval:
            return

        self.last_quality_check = current_time
        
        # Calculate FPS using a weighted moving average
        recent_times = list(self.frame_times)[-10:]  # Last 10 frames
        older_times = list(self.frame_times)[:-10]   # Older frames
        
        if recent_times:
            recent_fps = 1.0 / np.mean(recent_times)
            weight = 0.7  # Give more weight to recent frames
        else:
            recent_fps = 0.0
            weight = 0.0
            
        if older_times:
            older_fps = 1.0 / np.mean(older_times)
        else:
            older_fps = 0.0
            
        current_fps = weight * recent_fps + (1 - weight) * older_fps

        # Check memory usage
        if current_time - self.last_memory_check >= self.memory_check_interval:
            self.last_memory_check = current_time
            self.memory_usage = self.hardware.get_memory_usage()
            
            # Reduce quality if memory usage is too high
            if self.memory_usage > self.config.max_memory_usage * 0.9:
                self.current_quality = max(
                    self.current_quality - self.config.quality_step * 2,
                    self.config.min_quality
                )
                logger.warning(f"High memory usage ({self.memory_usage:.0f}MB), reducing quality")
                return

        # Adjust quality based on FPS
        if current_fps < self.config.min_fps and self.current_quality > self.config.min_quality:
            self.current_quality = max(
                self.current_quality - self.config.quality_step,
                self.config.min_quality
            )
            logger.info(f"Reducing quality to {self.current_quality:.2f} to maintain performance")
        elif current_fps > self.config.target_fps * 1.1 and self.current_quality < self.config.max_quality:
            self.current_quality = min(
                self.current_quality + self.config.quality_step,
                self.config.max_quality
            )
            logger.info(f"Increasing quality to {self.current_quality:.2f}")

    def _prefetch_frames(self):
        """Prefetch and preprocess frames in a background thread."""
        while True:
            try:
                if len(self.prefetch_queue) < self.config.prefetch_size and self.frame_buffer:
                    frame, frame_number = self.frame_buffer[0]
                    
                    # Pre-process frame
                    if self.use_gpu:
                        gpu_frame = cv2.cuda.GpuMat(frame)
                        processed = self.preprocessor.process_gpu(gpu_frame)
                    else:
                        processed = self.preprocessor.process(frame)
                        
                    self.prefetch_queue.append((processed, frame_number))
                    
                else:
                    time.sleep(0.001)  # Short sleep to prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Error in frame prefetching: {e}")
                time.sleep(0.1)  # Longer sleep on error

    def process_frame(
        self,
        frame: np.ndarray,
        video_id: int,
        frame_number: int,
        fps: Optional[float] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Process a single frame through the tracking pipeline.

        Args:
            frame: Input frame
            video_id: Video identifier
            frame_number: Frame number in sequence
            fps: Frames per second (optional)

        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()

        # Update video context if needed
        if video_id != self.current_video_id:
            self.current_video_id = video_id
            self.frame_buffer.clear()
            self.result_buffer.clear()
            self.prefetch_queue.clear()

        if fps is not None:
            self.fps = fps

        # Adjust frame size based on current quality
        if self.current_quality < 1.0:
            h, w = frame.shape[:2]
            new_size = (int(w * self.current_quality), int(h * self.current_quality))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        # Add frame to buffer
        self.frame_buffer.append((frame, frame_number))

        # Process in batches if buffer is full
        if len(self.frame_buffer) >= self.config.batch_size:
            batch_results = self._process_batch()
            self.result_buffer.update(batch_results)
            self.frame_buffer.clear()

        # Get results for current frame
        results = self.result_buffer.get(frame_number, {})
        if not results:
            # Process single frame if not in buffer
            results = self._process_single_frame(frame)

        # Update performance metrics
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        self._adjust_quality()

        self.frame_count += 1
        return results

    def _process_batch(self) -> Dict[int, Dict]:
        """Process a batch of frames efficiently."""
        batch_results = {}
        
        # Use prefetched frames if available
        if self.config.enable_prefetch and self.prefetch_queue:
            norm_frames = []
            frame_numbers = []
            frames = []
            
            while self.prefetch_queue and len(frames) < self.config.batch_size:
                processed, number = self.prefetch_queue.popleft()
                norm_frames.append(processed)
                frame_numbers.append(number)
                frames.append(self.frame_buffer[len(frames)][0])
                
        else:
            # Pre-process batch
            frames = [f[0] for f in self.frame_buffer]
            frame_numbers = [f[1] for f in self.frame_buffer]
            
            # Use GPU if available
            if self.use_gpu:
                gpu_frames = []
                for frame in frames:
                    gpu_frame = self.gpu_memory_pool.getBuffer(
                        frame.shape[0],
                        frame.shape[1],
                        frame.dtype
                    )
                    gpu_frame.upload(frame)
                    gpu_frames.append(gpu_frame)
                    
                norm_frames = [
                    self.preprocessor.process_gpu(f) for f in gpu_frames
                ]
            else:
                norm_frames = [self.preprocessor.process(f) for f in frames]

        # Update scene complexity for batch
        self.scene_complexity = self.algorithm_selector.analyze_batch(norm_frames)
        self.tracking_mode = self.tracking_hw.get_tracking_mode(self.scene_complexity)

        # Process frames in parallel
        futures = []
        for frame, norm_frame, frame_number in zip(frames, norm_frames, frame_numbers):
            future = self.thread_pool.submit(
                self._process_frame_parallel,
                frame,
                norm_frame if not self.use_gpu else norm_frame.download(),
                frame_number
            )
            futures.append((frame_number, future))

        # Collect results
        for frame_number, future in futures:
            try:
                results = future.result()
                batch_results[frame_number] = results
            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {e}")
                batch_results[frame_number] = {}

        # Clean up GPU memory
        if self.use_gpu:
            for frame in gpu_frames:
                frame.release()
            for frame in norm_frames:
                frame.release()

        return batch_results

    def _process_frame_parallel(
        self,
        frame: np.ndarray,
        norm_frame: np.ndarray,
        frame_number: int
    ) -> Dict:
        """Process a single frame in parallel."""
        results = {}
        
        # Run motion detection if enabled
        if self.motion_system:
            motion_frame, motion_results = self.motion_system.process_frame(
                norm_frame,
                self.current_video_id,
                frame_number,
                self.fps,
                return_visualization=self.config.enable_visualization,
            )
            results["motion"] = motion_results
            if motion_frame is not None:
                results["motion_vis"] = motion_frame

        # Run object tracking
        tracking_results = self.object_tracker.track_frame(
            norm_frame,
            frame_number,
            self.tracking_mode,
        )
        results["tracking"] = tracking_results

        # Run formation analysis if we have player positions
        if tracking_results.get("player_positions"):
            formation_results = self.formation_analyzer.analyze_formation(
                tracking_results["player_positions"],
                frame_number,
                self.fps,
            )
            results["formation"] = formation_results

        # Generate visualization if enabled
        if self.config.enable_visualization:
            vis_frame = self._visualize_results(frame.copy(), results)
            results["visualization"] = vis_frame

        return results

    def _process_single_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame when batch processing is not possible."""
        results = {}
        
        # Pre-process frame
        if self.use_gpu:
            gpu_frame = self.gpu_memory_pool.getBuffer(
                frame.shape[0],
                frame.shape[1],
                frame.dtype
            )
            gpu_frame.upload(frame)
            norm_frame = self.preprocessor.process_gpu(gpu_frame)
        else:
            norm_frame = self.preprocessor.process(frame)

        # Update scene complexity
        self.scene_complexity = self.algorithm_selector.analyze_frame(
            norm_frame if not self.use_gpu else norm_frame.download()
        )
        self.tracking_mode = self.tracking_hw.get_tracking_mode(self.scene_complexity)

        # Process frame
        results = self._process_frame_parallel(
            frame,
            norm_frame if not self.use_gpu else norm_frame.download(),
            self.frame_count
        )

        # Clean up GPU memory
        if self.use_gpu:
            gpu_frame.release()
            norm_frame.release()

        return results

    def _visualize_results(
        self,
        frame: np.ndarray,
        results: Dict[str, Union[np.ndarray, Dict]],
    ) -> np.ndarray:
        """Generate visualization of tracking and analysis results.

        Args:
            frame: Original frame to draw on
            results: Processing results to visualize

        Returns:
            Frame with visualizations
        """
        # Draw tracking results
        tracking_results = results.get("tracking", {})
        if tracking_results:
            # Draw object bounding boxes and trajectories
            for obj_id, obj_data in tracking_results.get("objects", {}).items():
                bbox = obj_data.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {obj_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Draw trajectory
                trajectory = obj_data.get("trajectory", [])
                if len(trajectory) > 1:
                    points = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(frame, [points], False, (255, 0, 0), 2)

        # Draw formation analysis results
        formation_results = results.get("formation", {})
        if formation_results:
            formation_type = formation_results.get("type")
            if formation_type:
                cv2.putText(
                    frame,
                    f"Formation: {formation_type}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Draw player positions and roles
            positions = formation_results.get("player_positions", [])
            for pos in positions:
                x, y = map(int, pos[:2])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw scene complexity and tracking mode
        cv2.putText(
            frame,
            f"Scene: {results.get('scene_complexity', 'unknown')}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Mode: {results.get('tracking_mode', 'unknown')}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.frame_times:
            return {
                "fps": 0.0,
                "avg_frame_time": 0.0,
                "quality": self.current_quality,
                "memory_usage_mb": self.memory_usage,
                "batch_size": self.config.batch_size,
                "prefetch_queue_size": len(self.prefetch_queue) if self.config.enable_prefetch else 0,
            }
            
        avg_frame_time = np.mean(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return {
            "fps": current_fps,
            "avg_frame_time": avg_frame_time,
            "quality": self.current_quality,
            "memory_usage_mb": self.memory_usage,
            "batch_size": self.config.batch_size,
            "prefetch_queue_size": len(self.prefetch_queue) if self.config.enable_prefetch else 0,
        }

    def reset(self):
        """Reset the pipeline state."""
        self.frame_count = 0
        self.current_video_id = None
        self.frame_buffer.clear()
        self.result_buffer.clear()
        self.frame_times.clear()
        self.current_quality = self.config.max_quality
        self.last_quality_check = time.time()
        self.prefetch_queue.clear()
        
        if self.motion_system:
            self.motion_system.reset()
        self.object_tracker.reset()
        self.formation_analyzer.reset() 