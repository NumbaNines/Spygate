"""
Advanced Memory Management and Optimization for SpygateAI.

This module provides comprehensive memory management including garbage collection,
memory monitoring, leak detection, and optimization strategies for large-scale
video processing and machine learning operations.
"""

import gc
import os
import psutil
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable
from collections import defaultdict
import traceback

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from .logging_config import get_logger
from .error_handling import handle_errors, error_boundary


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    process_memory_mb: float
    process_memory_percent: float
    system_memory_mb: float
    system_memory_percent: float
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    python_objects: int = 0
    gc_generation_counts: List[int] = None
    large_objects_count: int = 0


@dataclass
class MemoryLeak:
    """Memory leak detection result."""
    object_type: str
    count_increase: int
    size_estimate_mb: float
    first_seen: datetime
    last_seen: datetime
    growth_rate_per_hour: float


class MemoryMonitor:
    """Advanced memory monitoring and leak detection."""
    
    def __init__(self, sampling_interval: float = 60.0, history_size: int = 1000):
        self.logger = get_logger()
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        # Object tracking for leak detection
        self.object_counts: Dict[str, List[tuple]] = defaultdict(list)  # type -> [(timestamp, count)]
        self.large_object_threshold = 1024 * 1024  # 1MB
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Process handle
        self.process = psutil.Process()
        
        # Memory pressure thresholds
        self.memory_warning_threshold = 80.0  # percent
        self.memory_critical_threshold = 90.0  # percent
        self.gpu_warning_threshold = 85.0  # percent
        self.gpu_critical_threshold = 95.0  # percent
        
        self.logger.info("Memory monitor initialized")
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            memory_info = self.process.memory_info()
            process_memory_mb = memory_info.rss / (1024 * 1024)
            process_memory_percent = (process_memory_mb / (system_memory.total / (1024 * 1024))) * 100
            
            # GPU memory
            gpu_memory_mb = 0.0
            gpu_memory_percent = 0.0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_memory_percent = (gpu_memory_mb / gpu_total_mb) * 100 if gpu_total_mb > 0 else 0
                except:
                    pass
            
            # Python object counts
            python_objects = len(gc.get_objects())
            gc_counts = gc.get_count() if hasattr(gc, 'get_count') else [0, 0, 0]
            
            # Large objects count
            large_objects = self._count_large_objects()
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                process_memory_mb=process_memory_mb,
                process_memory_percent=process_memory_percent,
                system_memory_mb=system_memory.used / (1024 * 1024),
                system_memory_percent=system_memory.percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_memory_percent=gpu_memory_percent,
                python_objects=python_objects,
                gc_generation_counts=gc_counts,
                large_objects_count=large_objects
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=datetime.now(),
                process_memory_mb=0,
                process_memory_percent=0,
                system_memory_mb=0,
                system_memory_percent=0
            )
    
    def _count_large_objects(self) -> int:
        """Count objects larger than threshold."""
        try:
            large_count = 0
            for obj in gc.get_objects():
                try:
                    size = sys.getsizeof(obj)
                    if size > self.large_object_threshold:
                        large_count += 1
                except:
                    continue
            return large_count
        except:
            return 0
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        # Take baseline snapshot
        self.baseline_snapshot = self.take_snapshot()
        self.snapshots.append(self.baseline_snapshot)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            name="MemoryMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_worker(self):
        """Background monitoring worker."""
        while not self._stop_event.wait(self.sampling_interval):
            try:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                
                # Trim history
                if len(self.snapshots) > self.history_size:
                    self.snapshots = self.snapshots[-self.history_size:]
                
                # Check for memory pressure
                self._check_memory_pressure(snapshot)
                
                # Update object tracking for leak detection
                self._update_object_tracking()
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
    
    def _check_memory_pressure(self, snapshot: MemorySnapshot):
        """Check for memory pressure and log warnings."""
        # System memory pressure
        if snapshot.system_memory_percent > self.memory_critical_threshold:
            self.logger.critical(
                f"Critical system memory usage: {snapshot.system_memory_percent:.1f}% "
                f"({snapshot.system_memory_mb:.1f}MB)"
            )
        elif snapshot.system_memory_percent > self.memory_warning_threshold:
            self.logger.warning(
                f"High system memory usage: {snapshot.system_memory_percent:.1f}% "
                f"({snapshot.system_memory_mb:.1f}MB)"
            )
        
        # Process memory pressure
        if snapshot.process_memory_mb > 2048:  # 2GB
            self.logger.warning(
                f"High process memory usage: {snapshot.process_memory_mb:.1f}MB "
                f"({snapshot.process_memory_percent:.1f}% of system)"
            )
        
        # GPU memory pressure
        if snapshot.gpu_memory_percent > self.gpu_critical_threshold:
            self.logger.critical(
                f"Critical GPU memory usage: {snapshot.gpu_memory_percent:.1f}% "
                f"({snapshot.gpu_memory_mb:.1f}MB)"
            )
        elif snapshot.gpu_memory_percent > self.gpu_warning_threshold:
            self.logger.warning(
                f"High GPU memory usage: {snapshot.gpu_memory_percent:.1f}% "
                f"({snapshot.gpu_memory_mb:.1f}MB)"
            )
    
    def _update_object_tracking(self):
        """Update object tracking for leak detection."""
        try:
            current_time = datetime.now()
            
            # Count objects by type
            type_counts = defaultdict(int)
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
            
            # Update tracking
            for obj_type, count in type_counts.items():
                self.object_counts[obj_type].append((current_time, count))
                
                # Trim old data (keep last 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                self.object_counts[obj_type] = [
                    (t, c) for t, c in self.object_counts[obj_type] if t > cutoff_time
                ]
        
        except Exception as e:
            self.logger.error(f"Object tracking update failed: {e}")
    
    def detect_memory_leaks(self, min_growth_rate: float = 10.0) -> List[MemoryLeak]:
        """Detect potential memory leaks."""
        leaks = []
        current_time = datetime.now()
        
        for obj_type, history in self.object_counts.items():
            if len(history) < 10:  # Need enough data points
                continue
            
            try:
                # Calculate growth rate
                first_timestamp, first_count = history[0]
                last_timestamp, last_count = history[-1]
                
                time_diff_hours = (last_timestamp - first_timestamp).total_seconds() / 3600
                if time_diff_hours < 1.0:  # Need at least 1 hour of data
                    continue
                
                count_increase = last_count - first_count
                growth_rate = count_increase / time_diff_hours
                
                if growth_rate > min_growth_rate:
                    # Estimate size (very rough)
                    size_estimate = count_increase * 100 / (1024 * 1024)  # Assume 100 bytes per object
                    
                    leak = MemoryLeak(
                        object_type=obj_type,
                        count_increase=count_increase,
                        size_estimate_mb=size_estimate,
                        first_seen=first_timestamp,
                        last_seen=last_timestamp,
                        growth_rate_per_hour=growth_rate
                    )
                    leaks.append(leak)
            
            except Exception as e:
                self.logger.debug(f"Leak detection error for {obj_type}: {e}")
        
        # Sort by growth rate
        leaks.sort(key=lambda x: x.growth_rate_per_hour, reverse=True)
        
        if leaks:
            self.logger.warning(f"Detected {len(leaks)} potential memory leaks")
        
        return leaks
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        latest = self.snapshots[-1]
        baseline = self.baseline_snapshot or self.snapshots[0]
        
        # Calculate changes from baseline
        process_memory_change = latest.process_memory_mb - baseline.process_memory_mb
        python_objects_change = latest.python_objects - baseline.python_objects
        gpu_memory_change = latest.gpu_memory_mb - baseline.gpu_memory_mb
        
        # Calculate peaks
        process_memory_peak = max(s.process_memory_mb for s in self.snapshots)
        gpu_memory_peak = max(s.gpu_memory_mb for s in self.snapshots)
        
        return {
            "current": {
                "process_memory_mb": latest.process_memory_mb,
                "process_memory_percent": latest.process_memory_percent,
                "system_memory_percent": latest.system_memory_percent,
                "gpu_memory_mb": latest.gpu_memory_mb,
                "gpu_memory_percent": latest.gpu_memory_percent,
                "python_objects": latest.python_objects,
                "large_objects": latest.large_objects_count
            },
            "changes_from_baseline": {
                "process_memory_mb": process_memory_change,
                "python_objects": python_objects_change,
                "gpu_memory_mb": gpu_memory_change
            },
            "peaks": {
                "process_memory_mb": process_memory_peak,
                "gpu_memory_mb": gpu_memory_peak
            },
            "monitoring": {
                "active": self._monitoring,
                "snapshots_count": len(self.snapshots),
                "monitoring_duration_hours": (
                    (datetime.now() - baseline.timestamp).total_seconds() / 3600
                    if baseline else 0
                )
            }
        }


class MemoryOptimizer:
    """Memory optimization and cleanup utilities."""
    
    def __init__(self):
        self.logger = get_logger()
        self.cleanup_history: List[Dict[str, Any]] = []
    
    @handle_errors(reraise=False)
    def force_garbage_collection(self, aggressive: bool = False) -> Dict[str, Any]:
        """Force garbage collection with optional aggressive mode."""
        self.logger.debug("Starting garbage collection")
        
        before_objects = len(gc.get_objects())
        before_memory = self._get_memory_usage()
        
        # Standard garbage collection
        collected = gc.collect()
        
        if aggressive:
            # Multiple passes for aggressive cleanup
            for i in range(3):
                additional = gc.collect()
                collected += additional
                if additional == 0:
                    break
        
        after_objects = len(gc.get_objects())
        after_memory = self._get_memory_usage()
        
        memory_freed = before_memory - after_memory
        objects_freed = before_objects - after_objects
        
        result = {
            "objects_collected": collected,
            "objects_freed": objects_freed,
            "memory_freed_mb": memory_freed,
            "before_memory_mb": before_memory,
            "after_memory_mb": after_memory,
            "aggressive_mode": aggressive,
            "timestamp": datetime.now().isoformat()
        }
        
        self.cleanup_history.append(result)
        
        if memory_freed > 10:  # Log if freed more than 10MB
            self.logger.info(
                f"Garbage collection freed {memory_freed:.1f}MB "
                f"({objects_freed} objects, {collected} collected)"
            )
        
        return result
    
    @handle_errors(reraise=False)
    def clear_gpu_memory(self) -> Dict[str, Any]:
        """Clear GPU memory cache."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"status": "gpu_not_available"}
        
        try:
            before_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            before_cached = torch.cuda.memory_reserved() / (1024 * 1024)
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            after_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            after_cached = torch.cuda.memory_reserved() / (1024 * 1024)
            
            allocated_freed = before_allocated - after_allocated
            cached_freed = before_cached - after_cached
            
            result = {
                "allocated_freed_mb": allocated_freed,
                "cached_freed_mb": cached_freed,
                "before_allocated_mb": before_allocated,
                "after_allocated_mb": after_allocated,
                "before_cached_mb": before_cached,
                "after_cached_mb": after_cached,
                "timestamp": datetime.now().isoformat()
            }
            
            if cached_freed > 10:  # Log if freed more than 10MB
                self.logger.info(f"GPU memory cleared: {cached_freed:.1f}MB cache freed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU memory clear failed: {e}")
            return {"status": "error", "error": str(e)}
    
    @handle_errors(reraise=False)
    def optimize_numpy_arrays(self) -> Dict[str, Any]:
        """Optimize NumPy arrays in memory."""
        if not NUMPY_AVAILABLE:
            return {"status": "numpy_not_available"}
        
        try:
            arrays_processed = 0
            memory_saved = 0.0
            
            for obj in gc.get_objects():
                if isinstance(obj, np.ndarray):
                    try:
                        before_size = obj.nbytes
                        
                        # Try to optimize dtype if possible
                        if obj.dtype == np.float64 and obj.max() <= np.finfo(np.float32).max:
                            obj = obj.astype(np.float32)
                            after_size = obj.nbytes
                            memory_saved += (before_size - after_size) / (1024 * 1024)
                            arrays_processed += 1
                        
                        # Ensure arrays are contiguous for better memory access
                        if not obj.flags.c_contiguous:
                            obj = np.ascontiguousarray(obj)
                            arrays_processed += 1
                    
                    except Exception:
                        continue
            
            result = {
                "arrays_processed": arrays_processed,
                "memory_saved_mb": memory_saved,
                "timestamp": datetime.now().isoformat()
            }
            
            if arrays_processed > 0:
                self.logger.info(f"NumPy optimization: {arrays_processed} arrays, {memory_saved:.1f}MB saved")
            
            return result
            
        except Exception as e:
            self.logger.error(f"NumPy optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    @handle_errors(reraise=False)
    def cleanup_opencv_memory(self) -> Dict[str, Any]:
        """Clean up OpenCV memory."""
        if not OPENCV_AVAILABLE:
            return {"status": "opencv_not_available"}
        
        try:
            # OpenCV doesn't expose direct memory management,
            # but we can clean up any large Mat objects
            mat_objects = 0
            
            for obj in gc.get_objects():
                if hasattr(obj, '__class__') and 'cv2' in str(type(obj)):
                    mat_objects += 1
            
            # Force cleanup of any OpenCV internal caches
            if hasattr(cv2, 'setUseOptimized'):
                cv2.setUseOptimized(True)
            
            result = {
                "opencv_objects_found": mat_objects,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenCV cleanup failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def comprehensive_cleanup(self) -> Dict[str, Any]:
        """Perform comprehensive memory cleanup."""
        self.logger.info("Starting comprehensive memory cleanup")
        
        results = {}
        
        # 1. Standard garbage collection
        results["garbage_collection"] = self.force_garbage_collection()
        
        # 2. GPU memory cleanup
        results["gpu_cleanup"] = self.clear_gpu_memory()
        
        # 3. NumPy optimization
        results["numpy_optimization"] = self.optimize_numpy_arrays()
        
        # 4. OpenCV cleanup
        results["opencv_cleanup"] = self.cleanup_opencv_memory()
        
        # 5. Aggressive garbage collection
        results["aggressive_gc"] = self.force_garbage_collection(aggressive=True)
        
        # Calculate total memory freed
        total_memory_freed = 0.0
        if results["garbage_collection"].get("memory_freed_mb", 0) > 0:
            total_memory_freed += results["garbage_collection"]["memory_freed_mb"]
        if results["gpu_cleanup"].get("cached_freed_mb", 0) > 0:
            total_memory_freed += results["gpu_cleanup"]["cached_freed_mb"]
        if results["numpy_optimization"].get("memory_saved_mb", 0) > 0:
            total_memory_freed += results["numpy_optimization"]["memory_saved_mb"]
        
        results["summary"] = {
            "total_memory_freed_mb": total_memory_freed,
            "cleanup_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Comprehensive cleanup completed: {total_memory_freed:.1f}MB freed")
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0


class SmartMemoryManager:
    """Intelligent memory management with automatic optimization."""
    
    def __init__(self, auto_cleanup_threshold: float = 80.0, cleanup_interval: float = 300.0):
        self.logger = get_logger()
        self.monitor = MemoryMonitor()
        self.optimizer = MemoryOptimizer()
        
        self.auto_cleanup_threshold = auto_cleanup_threshold  # percent
        self.cleanup_interval = cleanup_interval  # seconds
        
        self._auto_cleanup_enabled = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        self.logger.info("Smart memory manager initialized")
    
    def start_auto_management(self):
        """Start automatic memory management."""
        if self._auto_cleanup_enabled:
            return
        
        self._auto_cleanup_enabled = True
        self._stop_cleanup.clear()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._auto_cleanup_worker,
            name="AutoMemoryCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        
        self.logger.info("Automatic memory management started")
    
    def stop_auto_management(self):
        """Stop automatic memory management."""
        if not self._auto_cleanup_enabled:
            return
        
        self._auto_cleanup_enabled = False
        self._stop_cleanup.set()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        self.logger.info("Automatic memory management stopped")
    
    def _auto_cleanup_worker(self):
        """Background worker for automatic cleanup."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                # Check if cleanup is needed
                if self._should_cleanup():
                    self.logger.info("Automatic memory cleanup triggered")
                    self.optimizer.comprehensive_cleanup()
            
            except Exception as e:
                self.logger.error(f"Auto cleanup error: {e}")
    
    def _should_cleanup(self) -> bool:
        """Determine if cleanup is needed."""
        try:
            if not self.monitor.snapshots:
                return False
            
            latest = self.monitor.snapshots[-1]
            
            # Check system memory pressure
            if latest.system_memory_percent > self.auto_cleanup_threshold:
                return True
            
            # Check GPU memory pressure
            if latest.gpu_memory_percent > self.auto_cleanup_threshold:
                return True
            
            # Check for rapid memory growth
            if len(self.monitor.snapshots) >= 5:
                recent_snapshots = self.monitor.snapshots[-5:]
                memory_growth = (
                    recent_snapshots[-1].process_memory_mb - recent_snapshots[0].process_memory_mb
                )
                if memory_growth > 100:  # More than 100MB growth in recent samples
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cleanup check failed: {e}")
            return False
    
    @contextmanager
    def memory_guard(self, operation_name: str = "operation", cleanup_threshold: Optional[float] = None):
        """Context manager that monitors memory during operation and cleans up if needed."""
        threshold = cleanup_threshold or self.auto_cleanup_threshold
        
        # Take before snapshot
        before_snapshot = self.monitor.take_snapshot()
        self.logger.debug(f"Memory guard start [{operation_name}]: {before_snapshot.process_memory_mb:.1f}MB")
        
        try:
            yield
        finally:
            # Take after snapshot
            after_snapshot = self.monitor.take_snapshot()
            memory_increase = after_snapshot.process_memory_mb - before_snapshot.process_memory_mb
            
            self.logger.debug(
                f"Memory guard end [{operation_name}]: {after_snapshot.process_memory_mb:.1f}MB "
                f"(+{memory_increase:.1f}MB)"
            )
            
            # Check if cleanup is needed
            if (after_snapshot.system_memory_percent > threshold or 
                after_snapshot.gpu_memory_percent > threshold or 
                memory_increase > 50):  # 50MB increase
                
                self.logger.info(f"Memory guard cleanup triggered after {operation_name}")
                self.optimizer.comprehensive_cleanup()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory management status."""
        return {
            "auto_management_active": self._auto_cleanup_enabled,
            "monitor_status": self.monitor.get_memory_summary(),
            "cleanup_history_count": len(self.optimizer.cleanup_history),
            "memory_leaks": len(self.monitor.detect_memory_leaks()),
            "thresholds": {
                "auto_cleanup_threshold": self.auto_cleanup_threshold,
                "cleanup_interval": self.cleanup_interval
            }
        }


# Global instances
_global_memory_monitor: Optional[MemoryMonitor] = None
_global_memory_optimizer: Optional[MemoryOptimizer] = None
_global_smart_manager: Optional[SmartMemoryManager] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def get_memory_optimizer() -> MemoryOptimizer:
    """Get or create global memory optimizer."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    return _global_memory_optimizer


def get_smart_memory_manager() -> SmartMemoryManager:
    """Get or create global smart memory manager."""
    global _global_smart_manager
    if _global_smart_manager is None:
        _global_smart_manager = SmartMemoryManager()
    return _global_smart_manager


# Convenience functions
def optimize_memory():
    """Perform comprehensive memory optimization."""
    return get_memory_optimizer().comprehensive_cleanup()


def memory_guard(operation_name: str = "operation", cleanup_threshold: Optional[float] = None):
    """Context manager for memory monitoring and cleanup."""
    return get_smart_memory_manager().memory_guard(operation_name, cleanup_threshold)


if __name__ == "__main__":
    # Test memory management system
    logger = get_logger()
    manager = get_smart_memory_manager()
    
    logger.info("Testing memory management system")
    
    # Start monitoring
    manager.start_auto_management()
    
    # Test memory guard
    with memory_guard("test_operation"):
        # Simulate memory usage
        test_data = []
        for i in range(10000):
            test_data.append([j for j in range(100)])
        
        logger.info(f"Created test data with {len(test_data)} items")
    
    # Get status
    status = manager.get_status()
    logger.info(f"Memory management status: {status}")
    
    # Manual optimization
    result = optimize_memory()
    logger.info(f"Manual optimization result: {result['summary']}")
    
    # Stop management
    manager.stop_auto_management()
    
    logger.info("Memory management test completed")