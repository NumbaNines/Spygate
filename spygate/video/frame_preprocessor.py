"""
Frame preprocessing module for video tracking.

This module handles frame preprocessing operations including normalization,
noise reduction, and enhancement for optimal tracking performance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

import cv2
import numpy as np
from numba import jit, cuda

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for frame preprocessing."""

    target_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    batch_size: int = 16  # Increased for better GPU utilization
    max_thread_workers: int = 4
    enable_parallel: bool = True
    enable_gpu: bool = True  # Enable GPU acceleration if available
    enable_prefetch: bool = True  # Enable frame prefetching
    prefetch_size: int = 4  # Number of frames to prefetch
    quality_step: float = 0.1  # Quality adjustment step
    min_quality: float = 0.5  # Minimum processing quality
    max_quality: float = 1.0  # Maximum processing quality
    max_memory_usage: float = 2048.0  # Maximum memory usage in MB
    cleanup_interval: int = 100  # Frames between cleanup
    cache_size: int = 1000  # Size of preprocessing cache
    enable_memory_tracking: bool = True  # Track memory usage
    memory_warning_threshold: float = 0.9  # 90% of max memory
    enable_adaptive_quality: bool = True  # Adjust quality based on resources
    enable_cache_compression: bool = True  # Compress cached frames
    compression_level: int = 1  # Compression level (1-9)
    max_cached_frames: int = 500  # Maximum frames to keep in memory


@jit(nopython=True, parallel=True)
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame values to [0, 1] range using parallel processing."""
    min_val = np.min(frame)
    max_val = np.max(frame)
    if max_val == min_val:
        return np.zeros_like(frame, dtype=np.float32)
    return ((frame - min_val) / (max_val - min_val)).astype(np.float32)


@cuda.jit
def normalize_frame_gpu(frame, output, min_val, max_val):
    """CUDA kernel for frame normalization."""
    x, y = cuda.grid(2)
    if x < frame.shape[0] and y < frame.shape[1]:
        if max_val == min_val:
            output[x, y] = 0
        else:
            output[x, y] = (frame[x, y] - min_val) / (max_val - min_val)


@jit(nopython=True, parallel=True)
def enhance_contrast(frame: np.ndarray) -> np.ndarray:
    """Enhance frame contrast using parallel processing."""
    # Apply CLAHE-like contrast enhancement
    min_val = np.min(frame)
    max_val = np.max(frame)
    
    if max_val == min_val:
        return frame
        
    # Calculate histogram
    hist = np.zeros(256, dtype=np.int32)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            bin_idx = int(255 * (frame[i, j] - min_val) / (max_val - min_val))
            hist[bin_idx] += 1
            
    # Calculate cumulative histogram
    cum_hist = np.zeros(256, dtype=np.float32)
    cum_hist[0] = hist[0]
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i - 1] + hist[i]
        
    # Normalize cumulative histogram
    cum_hist = cum_hist / cum_hist[-1]
    
    # Apply histogram equalization
    enhanced = np.zeros_like(frame)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            bin_idx = int(255 * (frame[i, j] - min_val) / (max_val - min_val))
            enhanced[i, j] = cum_hist[bin_idx]
            
    return enhanced


@cuda.jit
def enhance_contrast_gpu(frame, output, hist, cum_hist):
    """CUDA kernel for contrast enhancement."""
    x, y = cuda.grid(2)
    if x < frame.shape[0] and y < frame.shape[1]:
        # Map value to histogram bin
        bin_idx = int(frame[x, y] * 255)
        # Apply equalization
        output[x, y] = cum_hist[bin_idx]


class FramePreprocessor:
    """Handles frame preprocessing with memory optimization."""

    def __init__(self, config: PreprocessingConfig, hardware: HardwareDetector):
        self.config = config
        self.optimizer = TierOptimizer(hardware)
        self.frame_cache = {}
        self.cache_order = deque(maxlen=self.config.max_cached_frames)
        self.frame_count = 0
        self.last_cleanup = 0
        self.current_quality = config.max_quality
        self.lock = threading.Lock()
        self.prefetch_queue = deque(maxlen=config.prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=config.max_thread_workers)
        self.memory_usage = 0.0
        self.frame_sizes = {}
        
        # Initialize GPU context if available
        if self.config.enable_gpu and cuda.is_available():
            cuda.select_device(0)
            self.stream = cuda.stream()
        else:
            self.stream = None

    def _track_memory(self, frame_id: int, frame: np.ndarray):
        """Track memory usage of frames."""
        if not self.config.enable_memory_tracking:
            return
            
        frame_size = frame.nbytes / (1024 * 1024)  # Convert to MB
        self.frame_sizes[frame_id] = frame_size
        self.memory_usage += frame_size
        
        # Check memory threshold
        if self.memory_usage > self.config.max_memory_usage * self.config.memory_warning_threshold:
            self._reduce_memory_usage()

    def _reduce_memory_usage(self):
        """Reduce memory usage when threshold is reached."""
        if not self.config.enable_memory_tracking:
            return
            
        logger.info(f"Memory usage ({self.memory_usage:.2f}MB) exceeded warning threshold")
        
        # Remove oldest frames from cache
        while (self.memory_usage > self.config.max_memory_usage * 0.8 and 
               len(self.cache_order) > 0):
            oldest_id = self.cache_order.popleft()
            if oldest_id in self.frame_cache:
                self.memory_usage -= self.frame_sizes.pop(oldest_id, 0)
                del self.frame_cache[oldest_id]
                
        # Force garbage collection
        gc.collect()
        
        # Adjust quality if needed
        if self.config.enable_adaptive_quality:
            self.current_quality = max(
                self.config.min_quality,
                self.current_quality - self.config.quality_step
            )
            logger.info(f"Reduced processing quality to {self.current_quality:.2f}")

    def _compress_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compress frame data for caching."""
        if not self.config.enable_cache_compression:
            return frame
            
        # Use cv2 compression for RGB/BGR frames
        if len(frame.shape) == 3:
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 
                          int(90 * self.current_quality)]
            result, encoded = cv2.imencode('.jpg', frame, encode_param)
            if result:
                return encoded
                
        return frame

    def _decompress_frame(self, data: np.ndarray) -> np.ndarray:
        """Decompress cached frame data."""
        if not self.config.enable_cache_compression:
            return data
            
        # Check if data is compressed
        if isinstance(data, np.ndarray) and data.dtype == np.uint8:
            try:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            except:
                pass
                
        return data

    @jit(nopython=True)
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame values."""
        return (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply denoising based on current quality."""
        if not self.config.denoise:
            return frame
            
        h = int(10 * self.current_quality)  # Adjust filter strength
        if self.config.enable_gpu and self.stream:
            return cv2.cuda.fastNlMeansDenoisingColored(
                frame, None, h, h, 7, 21, stream=self.stream
            )
        else:
            return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame contrast based on current quality."""
        if not self.config.enhance_contrast:
            return frame
            
        # Apply CLAHE with adaptive clip limit
        clip_limit = 2.0 + (1.0 * self.current_quality)
        if len(frame.shape) == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            return clahe.apply(frame)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if target size is set."""
        if self.config.target_size is None:
            return frame
            
        if self.config.enable_gpu and self.stream:
            gpu_frame = cv2.cuda_GpuMat(frame)
            resized = cv2.cuda.resize(
                gpu_frame, 
                self.config.target_size,
                interpolation=cv2.INTER_AREA
            )
            return resized.download()
        else:
            return cv2.resize(
                frame,
                self.config.target_size,
                interpolation=cv2.INTER_AREA
            )

    def preprocess_frame(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """Preprocess a single frame with quality-aware processing."""
        # Check cache first
        if frame_id in self.frame_cache:
            return self._decompress_frame(self.frame_cache[frame_id])
            
        # Apply preprocessing steps
        processed = frame.copy()
        
        # Resize if needed
        processed = self._resize_frame(processed)
        
        # Normalize
        if self.config.normalize:
            processed = self._normalize_frame(processed)
            
        # Denoise with quality-aware settings
        processed = self._denoise_frame(processed)
        
        # Enhance contrast
        processed = self._enhance_contrast(processed)
        
        # Cache the result
        with self.lock:
            compressed = self._compress_frame(processed)
            self.frame_cache[frame_id] = compressed
            self.cache_order.append(frame_id)
            self._track_memory(frame_id, compressed)
            
        # Periodic cleanup
        self.frame_count += 1
        if (self.frame_count - self.last_cleanup) >= self.config.cleanup_interval:
            self._cleanup()
            
        return processed

    def preprocess_batch(self, frames: List[np.ndarray], start_id: int) -> List[np.ndarray]:
        """Preprocess a batch of frames in parallel."""
        if not self.config.enable_parallel:
            return [self.preprocess_frame(f, start_id + i) for i, f in enumerate(frames)]
            
        # Process frames in parallel using thread pool
        futures = []
        for i, frame in enumerate(frames):
            future = self.executor.submit(
                self.preprocess_frame, frame, start_id + i
            )
            futures.append(future)
            
        # Collect results
        return [f.result() for f in futures]

    def _cleanup(self):
        """Perform periodic cleanup of resources."""
        with self.lock:
            # Remove old frames from cache
            while len(self.cache_order) > self.config.max_cached_frames:
                oldest_id = self.cache_order.popleft()
                if oldest_id in self.frame_cache:
                    self.memory_usage -= self.frame_sizes.pop(oldest_id, 0)
                    del self.frame_cache[oldest_id]
                    
            # Reset counters
            self.last_cleanup = self.frame_count
            
            # Force garbage collection
            if self.config.enable_memory_tracking:
                gc.collect()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            "memory_usage_mb": self.memory_usage,
            "cache_size": len(self.frame_cache),
            "quality_level": self.current_quality,
            "memory_threshold": self.config.max_memory_usage * self.config.memory_warning_threshold,
            "frames_processed": self.frame_count
        }

    def __del__(self):
        """Cleanup resources on deletion."""
        self.executor.shutdown(wait=True)
        if self.stream:
            self.stream.synchronize()
        self.frame_cache.clear()
        self.cache_order.clear()
        self.frame_sizes.clear()
        gc.collect()
