"""
Performance monitoring and optimization module.

This module provides real-time performance monitoring, metrics collection,
and adaptive optimization based on system resources and performance targets.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque
import threading
import psutil
import numpy as np

from .hardware import HardwareDetector
from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """Performance thresholds and targets."""
    
    target_fps: float = 30.0
    min_fps: float = 20.0
    max_memory_mb: float = 2048.0
    max_gpu_memory_mb: float = 1024.0
    memory_warning_threshold: float = 0.9
    gpu_warning_threshold: float = 0.9
    max_processing_time: float = 0.05  # 50ms per frame
    max_batch_time: float = 0.2  # 200ms per batch
    min_quality: float = 0.5
    max_quality: float = 1.0
    quality_step: float = 0.1
    fps_buffer_size: int = 100
    metrics_interval: float = 1.0  # 1 second
    cleanup_interval: int = 100  # frames


class PerformanceMonitor:
    """Monitors and optimizes system performance."""

    def __init__(
        self,
        thresholds: Optional[PerformanceThresholds] = None,
        hardware: Optional[HardwareDetector] = None,
    ):
        """Initialize performance monitor.
        
        Args:
            thresholds: Performance thresholds
            hardware: Hardware detector instance
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.hardware = hardware or HardwareDetector()
        self.metrics = MetricsCollector()
        
        # Performance tracking
        self.processing_times = deque(maxlen=self.thresholds.fps_buffer_size)
        self.batch_times = deque(maxlen=20)
        self.memory_usage = deque(maxlen=60)  # 1 minute history
        self.gpu_memory = deque(maxlen=60)
        self.quality_history = deque(maxlen=60)
        
        # Current state
        self.current_quality = self.thresholds.max_quality
        self.frame_count = 0
        self.last_cleanup = 0
        self.last_metrics_time = time.time()
        
        # Performance stats
        self.stats = {
            "fps": 0.0,
            "avg_processing_time": 0.0,
            "memory_usage_mb": 0.0,
            "gpu_memory_mb": 0.0,
            "quality_level": self.current_quality,
            "dropped_frames": 0,
            "optimization_events": 0,
        }
        
        # Threading
        self._lock = threading.Lock()
        self._start_monitoring()
        
        logger.info(
            f"Initialized PerformanceMonitor with {self.hardware.tier.name} tier"
        )

    def _start_monitoring(self):
        """Start background monitoring thread."""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._update_system_metrics()
                self._check_thresholds()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Back off on error

    def _update_system_metrics(self):
        """Update system resource metrics."""
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
        # GPU memory if available
        if self.hardware.has_cuda:
            gpu_memory = self.hardware.get_gpu_memory_usage()
            self.gpu_memory.append(gpu_memory)
            
        # Update metrics collector
        self.metrics.record_gauge("memory_usage_mb", memory_mb)
        if self.hardware.has_cuda:
            self.metrics.record_gauge("gpu_memory_mb", gpu_memory)
            
        # Calculate moving averages
        with self._lock:
            self.stats.update({
                "memory_usage_mb": np.mean(self.memory_usage),
                "gpu_memory_mb": np.mean(self.gpu_memory) if self.gpu_memory else 0.0,
            })

    def _check_thresholds(self):
        """Check if any thresholds are exceeded."""
        with self._lock:
            # Check memory usage
            if self.stats["memory_usage_mb"] > self.thresholds.max_memory_mb * self.thresholds.memory_warning_threshold:
                logger.warning(
                    f"High memory usage: {self.stats['memory_usage_mb']:.1f}MB"
                )
                self._optimize_memory()
                
            # Check GPU memory
            if (
                self.hardware.has_cuda and
                self.stats["gpu_memory_mb"] > self.thresholds.max_gpu_memory_mb * self.thresholds.gpu_warning_threshold
            ):
                logger.warning(
                    f"High GPU memory usage: {self.stats['gpu_memory_mb']:.1f}MB"
                )
                self._optimize_gpu_memory()
                
            # Check FPS
            if self.stats["fps"] < self.thresholds.min_fps:
                logger.warning(f"Low FPS: {self.stats['fps']:.1f}")
                self._optimize_performance()

    def _optimize_memory(self):
        """Optimize memory usage."""
        if self.current_quality > self.thresholds.min_quality:
            self.current_quality = max(
                self.thresholds.min_quality,
                self.current_quality - self.thresholds.quality_step
            )
            logger.info(f"Reduced quality to {self.current_quality:.2f}")
            self.quality_history.append(self.current_quality)
            self.stats["optimization_events"] += 1
            self.metrics.record_event("quality_reduction", {"reason": "memory"})

    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if self.current_quality > self.thresholds.min_quality:
            self.current_quality = max(
                self.thresholds.min_quality,
                self.current_quality - self.thresholds.quality_step
            )
            logger.info(f"Reduced quality to {self.current_quality:.2f}")
            self.quality_history.append(self.current_quality)
            self.stats["optimization_events"] += 1
            self.metrics.record_event("quality_reduction", {"reason": "gpu"})

    def _optimize_performance(self):
        """Optimize processing performance."""
        if self.current_quality > self.thresholds.min_quality:
            self.current_quality = max(
                self.thresholds.min_quality,
                self.current_quality - self.thresholds.quality_step
            )
            logger.info(f"Reduced quality to {self.current_quality:.2f}")
            self.quality_history.append(self.current_quality)
            self.stats["optimization_events"] += 1
            self.metrics.record_event("quality_reduction", {"reason": "fps"})

    def start_frame(self) -> float:
        """Start timing a new frame."""
        return time.perf_counter()

    def end_frame(self, start_time: float):
        """End frame timing and update metrics."""
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        with self._lock:
            # Update processing times
            self.processing_times.append(processing_time)
            
            # Check for dropped frames
            if processing_time > 1.0 / self.thresholds.min_fps:
                self.stats["dropped_frames"] += 1
                self.metrics.record_event("dropped_frame")
            
            # Update FPS stats
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                self.stats.update({
                    "avg_processing_time": avg_time,
                    "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
                })
            
            # Record metrics
            current_time = time.time()
            if current_time - self.last_metrics_time >= self.thresholds.metrics_interval:
                self.metrics.record_gauge("fps", self.stats["fps"])
                self.metrics.record_gauge("processing_time", self.stats["avg_processing_time"])
                self.metrics.record_gauge("quality", self.current_quality)
                self.last_metrics_time = current_time
            
            # Increment frame counter
            self.frame_count += 1
            
            # Periodic cleanup
            if (self.frame_count - self.last_cleanup) >= self.thresholds.cleanup_interval:
                self._cleanup()
                self.last_cleanup = self.frame_count

    def start_batch(self) -> float:
        """Start timing a batch operation."""
        return time.perf_counter()

    def end_batch(self, start_time: float):
        """End batch timing and update metrics."""
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        
        with self._lock:
            self.batch_times.append(batch_time)
            if batch_time > self.thresholds.max_batch_time:
                logger.warning(f"Slow batch processing: {batch_time:.3f}s")
                self.metrics.record_event("slow_batch")

    def get_quality_level(self) -> float:
        """Get current quality level."""
        return self.current_quality

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        with self._lock:
            return self.stats.copy()

    def get_metrics_report(self) -> Dict[str, List[float]]:
        """Get detailed metrics report."""
        with self._lock:
            return {
                "processing_times": list(self.processing_times),
                "batch_times": list(self.batch_times),
                "memory_usage": list(self.memory_usage),
                "gpu_memory": list(self.gpu_memory),
                "quality_history": list(self.quality_history),
            }

    def _cleanup(self):
        """Perform periodic cleanup."""
        # Clear old metrics
        self.metrics.cleanup_old_data()
        
        # Try to recover quality if performance is good
        if (
            self.stats["fps"] > self.thresholds.target_fps * 1.1 and
            self.current_quality < self.thresholds.max_quality and
            self.stats["memory_usage_mb"] < self.thresholds.max_memory_mb * 0.8 and
            (not self.hardware.has_cuda or
             self.stats["gpu_memory_mb"] < self.thresholds.max_gpu_memory_mb * 0.8)
        ):
            self.current_quality = min(
                self.thresholds.max_quality,
                self.current_quality + self.thresholds.quality_step
            )
            logger.info(f"Increased quality to {self.current_quality:.2f}")
            self.quality_history.append(self.current_quality)
            self.metrics.record_event("quality_increase")

    def __del__(self):
        """Cleanup on deletion."""
        self._monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0) 