"""
Advanced GPU Memory Management System for SpygateAI.

This module provides comprehensive GPU memory optimization including:
- Dynamic memory pool management
- Adaptive batch sizing based on available memory
- Memory fragmentation prevention
- Hardware-tier specific optimization strategies
- Proactive memory monitoring and cleanup
"""

import gc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .hardware import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies based on hardware tier."""

    ULTRA_LOW = "conservative"  # Aggressive memory saving
    LOW = "balanced"  # Balance performance and memory
    MEDIUM = "adaptive"  # Dynamic adjustment
    HIGH = "performance"  # Favor performance
    ULTRA = "unlimited"  # Maximum performance


@dataclass
class MemoryPoolConfig:
    """Configuration for GPU memory pool management."""

    # Pool configuration
    initial_pool_size: float = 0.5  # Initial pool size as fraction of total GPU memory
    max_pool_size: float = 0.8  # Maximum pool size as fraction of total GPU memory
    min_pool_size: float = 0.2  # Minimum pool size as fraction of total GPU memory

    # Buffer management
    buffer_growth_factor: float = 1.5  # Factor for growing buffer sizes
    max_buffer_count: int = 100  # Maximum number of cached buffers
    buffer_timeout: float = 300.0  # Timeout for unused buffers (seconds)

    # Memory monitoring
    cleanup_threshold: float = 0.85  # Trigger cleanup at this memory usage
    warning_threshold: float = 0.75  # Issue warning at this memory usage
    monitor_interval: float = 10.0  # Memory monitoring interval (seconds)

    # Fragmentation management
    defrag_threshold: float = 0.3  # Fragmentation ratio to trigger defragmentation
    defrag_interval: float = 600.0  # Defragmentation interval (seconds)


@dataclass
class MemoryBuffer:
    """Represents a reusable GPU memory buffer."""

    tensor: "torch.Tensor"
    size: tuple[int, ...]
    dtype: "torch.dtype"
    last_used: float
    usage_count: int = 0
    is_available: bool = True


class DynamicBatchSizer:
    """Dynamically adjusts batch sizes based on available GPU memory."""

    def __init__(self, hardware: HardwareDetector, config: MemoryPoolConfig):
        self.hardware = hardware
        self.config = config

        # Base batch sizes for different tiers
        self.base_batch_sizes = {
            HardwareTier.ULTRA_LOW: 1,
            HardwareTier.LOW: 2,
            HardwareTier.MEDIUM: 4,
            HardwareTier.HIGH: 8,
            HardwareTier.ULTRA: 16,
        }

        # Current optimal batch size
        self.current_batch_size = self.base_batch_sizes[hardware.tier]
        self.max_successful_batch = self.current_batch_size
        self.last_oom_batch = None

        # Performance tracking
        self.batch_performance = deque(maxlen=100)
        self.adjustment_history = deque(maxlen=50)

    def get_optimal_batch_size(self, model_memory_usage: float = 0.0) -> int:
        """Calculate optimal batch size based on current memory state."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 1

        try:
            # Get current memory state
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            free_memory = total_memory - allocated_memory

            # Estimate memory needed per batch item
            if model_memory_usage > 0:
                estimated_per_item = model_memory_usage
            else:
                # Use heuristic based on tier
                base_memory_per_item = {
                    HardwareTier.ULTRA_LOW: 50 * 1024 * 1024,  # 50MB
                    HardwareTier.LOW: 100 * 1024 * 1024,  # 100MB
                    HardwareTier.MEDIUM: 200 * 1024 * 1024,  # 200MB
                    HardwareTier.HIGH: 400 * 1024 * 1024,  # 400MB
                    HardwareTier.ULTRA: 800 * 1024 * 1024,  # 800MB
                }
                estimated_per_item = base_memory_per_item[self.hardware.tier]

            # Calculate safe batch size with buffer
            safety_factor = 0.7  # Use only 70% of available memory
            safe_memory = free_memory * safety_factor
            theoretical_batch_size = max(1, int(safe_memory // estimated_per_item))

            # Apply constraints
            max_batch = self.base_batch_sizes[self.hardware.tier] * 4
            min_batch = 1

            optimal_batch = min(theoretical_batch_size, max_batch)
            optimal_batch = max(optimal_batch, min_batch)

            # Consider OOM history
            if self.last_oom_batch and optimal_batch >= self.last_oom_batch:
                optimal_batch = max(1, self.last_oom_batch - 1)

            # Update current batch size gradually
            if optimal_batch != self.current_batch_size:
                change = optimal_batch - self.current_batch_size
                # Gradual adjustment to avoid sudden changes
                adjustment = max(-2, min(2, change))
                self.current_batch_size = max(1, self.current_batch_size + adjustment)

            return self.current_batch_size

        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {e}")
            return self.base_batch_sizes[self.hardware.tier]

    def record_batch_performance(self, batch_size: int, processing_time: float, success: bool):
        """Record batch processing performance for optimization."""
        self.batch_performance.append(
            {
                "batch_size": batch_size,
                "processing_time": processing_time,
                "success": success,
                "timestamp": time.time(),
            }
        )

        if success:
            self.max_successful_batch = max(self.max_successful_batch, batch_size)
        else:
            self.last_oom_batch = batch_size
            # Reduce current batch size immediately after OOM
            self.current_batch_size = max(1, batch_size // 2)


class GPUMemoryPool:
    """Advanced GPU memory pool with buffer reuse and fragmentation management."""

    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.buffers: dict[str, list[MemoryBuffer]] = {}
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            "allocations": 0,
            "deallocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_memory_saved": 0,
        }

        # Memory monitoring
        self.last_cleanup = time.time()
        self.last_defrag = time.time()

    def _get_buffer_key(self, size: tuple[int, ...], dtype: "torch.dtype") -> str:
        """Generate unique key for buffer lookup."""
        return f"{size}_{dtype}"

    def get_buffer(self, size: tuple[int, ...], dtype=None, device: str = "cuda:0"):
        """Get a reusable buffer from the pool or create new one."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        if dtype is None:
            dtype = torch.float32

        key = self._get_buffer_key(size, dtype)
        current_time = time.time()

        with self.lock:
            # Look for available buffer
            if key in self.buffers:
                for buffer in self.buffers[key]:
                    if buffer.is_available and buffer.dtype == dtype:
                        buffer.is_available = False
                        buffer.last_used = current_time
                        buffer.usage_count += 1
                        self.stats["cache_hits"] += 1

                        # Zero out the buffer for safety
                        buffer.tensor.zero_()
                        return buffer.tensor

            # Create new buffer
            try:
                tensor = torch.zeros(size, dtype=dtype, device=device)
                buffer = MemoryBuffer(
                    tensor=tensor,
                    size=size,
                    dtype=dtype,
                    last_used=current_time,
                    is_available=False,
                )

                if key not in self.buffers:
                    self.buffers[key] = []
                self.buffers[key].append(buffer)

                self.stats["allocations"] += 1
                self.stats["cache_misses"] += 1

                return tensor

            except torch.cuda.OutOfMemoryError:
                # Try cleanup and retry
                self._emergency_cleanup()
                tensor = torch.zeros(size, dtype=dtype, device=device)
                buffer = MemoryBuffer(
                    tensor=tensor,
                    size=size,
                    dtype=dtype,
                    last_used=current_time,
                    is_available=False,
                )

                if key not in self.buffers:
                    self.buffers[key] = []
                self.buffers[key].append(buffer)

                return tensor

    def return_buffer(self, tensor):
        """Return a buffer to the pool for reuse."""
        with self.lock:
            # Find the buffer and mark as available
            for buffer_list in self.buffers.values():
                for buffer in buffer_list:
                    if buffer.tensor.data_ptr() == tensor.data_ptr():
                        buffer.is_available = True
                        return

    def cleanup_expired_buffers(self):
        """Remove expired buffers from the pool."""
        current_time = time.time()

        with self.lock:
            total_freed = 0

            for key in list(self.buffers.keys()):
                buffer_list = self.buffers[key]
                expired_buffers = []

                for i, buffer in enumerate(buffer_list):
                    if (
                        buffer.is_available
                        and current_time - buffer.last_used > self.config.buffer_timeout
                    ):
                        expired_buffers.append(i)
                        total_freed += buffer.tensor.numel() * buffer.tensor.element_size()

                # Remove expired buffers (in reverse order to maintain indices)
                for i in reversed(expired_buffers):
                    del buffer_list[i]
                    self.stats["deallocations"] += 1

                # Remove empty buffer lists
                if not buffer_list:
                    del self.buffers[key]

            if total_freed > 0:
                self.stats["total_memory_saved"] += total_freed
                logger.debug(f"Freed {total_freed / 1024 / 1024:.2f} MB from expired buffers")

    def _emergency_cleanup(self):
        """Perform emergency cleanup to free memory."""
        logger.warning("Performing emergency GPU memory cleanup")

        with self.lock:
            # Free all available buffers
            total_freed = 0
            for key in list(self.buffers.keys()):
                buffer_list = self.buffers[key]
                available_buffers = []

                for i, buffer in enumerate(buffer_list):
                    if buffer.is_available:
                        available_buffers.append(i)
                        total_freed += buffer.tensor.numel() * buffer.tensor.element_size()

                # Remove available buffers
                for i in reversed(available_buffers):
                    del buffer_list[i]

                if not buffer_list:
                    del self.buffers[key]

        # Force garbage collection
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Emergency cleanup freed {total_freed / 1024 / 1024:.2f} MB")

    def get_statistics(self) -> dict:
        """Get memory pool statistics."""
        with self.lock:
            total_buffers = sum(len(buffer_list) for buffer_list in self.buffers.values())
            available_buffers = sum(
                sum(1 for buffer in buffer_list if buffer.is_available)
                for buffer_list in self.buffers.values()
            )

            return {
                **self.stats,
                "total_buffers": total_buffers,
                "available_buffers": available_buffers,
                "buffer_types": len(self.buffers),
                "hit_rate": (
                    self.stats["cache_hits"]
                    / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
                )
                * 100,
            }


class AdvancedGPUMemoryManager:
    """Comprehensive GPU memory management system."""

    def __init__(self, hardware: HardwareDetector, config: Optional[MemoryPoolConfig] = None):
        self.hardware = hardware
        self.config = config or MemoryPoolConfig()

        # Initialize components
        self.memory_pool = (
            GPUMemoryPool(self.config) if TORCH_AVAILABLE and torch.cuda.is_available() else None
        )
        self.batch_sizer = DynamicBatchSizer(hardware, self.config)

        # Memory monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Strategy selection based on hardware tier
        self.strategy = self._select_strategy()

        # Start monitoring if GPU is available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._start_monitoring()

        logger.info(f"Advanced GPU Memory Manager initialized with {self.strategy.value} strategy")

    def _select_strategy(self) -> MemoryStrategy:
        """Select memory management strategy based on hardware tier."""
        strategy_map = {
            HardwareTier.ULTRA_LOW: MemoryStrategy.ULTRA_LOW,
            HardwareTier.LOW: MemoryStrategy.LOW,
            HardwareTier.MEDIUM: MemoryStrategy.MEDIUM,
            HardwareTier.HIGH: MemoryStrategy.HIGH,
            HardwareTier.ULTRA: MemoryStrategy.ULTRA,
        }
        return strategy_map[self.hardware.tier]

    def _start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_thread is None:
            self.monitoring_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
            self.monitoring_thread.start()

    def _memory_monitor_loop(self):
        """Background loop for memory monitoring and cleanup."""
        while not self.stop_monitoring.wait(self.config.monitor_interval):
            try:
                self._check_memory_status()
                self._perform_maintenance()
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

    def _check_memory_status(self):
        """Check current memory status and trigger actions if needed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            usage_ratio = allocated_memory / total_memory

            if usage_ratio > self.config.cleanup_threshold:
                logger.warning(f"High GPU memory usage: {usage_ratio:.2%}")
                self._trigger_cleanup()
            elif usage_ratio > self.config.warning_threshold:
                logger.info(f"GPU memory usage: {usage_ratio:.2%}")

            # Check for fragmentation
            if self.memory_pool:
                fragmentation = self._calculate_fragmentation()
                if fragmentation > self.config.defrag_threshold:
                    current_time = time.time()
                    if current_time - self.memory_pool.last_defrag > self.config.defrag_interval:
                        self._defragment_memory()
                        self.memory_pool.last_defrag = current_time

        except Exception as e:
            logger.error(f"Error checking memory status: {e}")

    def _perform_maintenance(self):
        """Perform regular maintenance tasks."""
        if self.memory_pool:
            self.memory_pool.cleanup_expired_buffers()

    def _trigger_cleanup(self):
        """Trigger memory cleanup based on current strategy."""
        if self.strategy in [MemoryStrategy.ULTRA_LOW, MemoryStrategy.LOW]:
            # Aggressive cleanup
            if self.memory_pool:
                self.memory_pool._emergency_cleanup()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            gc.collect()
        else:
            # Gentle cleanup
            if self.memory_pool:
                self.memory_pool.cleanup_expired_buffers()
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()

    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio."""
        try:
            if not TORCH_AVAILABLE:
                return 0.0

            memory_stats = torch.cuda.memory_stats()
            allocated = memory_stats.get("allocated_bytes.all.current", 0)
            reserved = memory_stats.get("reserved_bytes.all.current", 0)

            if reserved == 0:
                return 0.0

            fragmentation = (reserved - allocated) / reserved
            return fragmentation
        except Exception:
            return 0.0

    def _defragment_memory(self):
        """Perform memory defragmentation."""
        logger.info("Performing GPU memory defragmentation")

        if self.memory_pool:
            # Clear all available buffers
            self.memory_pool._emergency_cleanup()

        # Force defragmentation
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        logger.info("Memory defragmentation completed")

    def get_buffer(self, size: tuple[int, ...], dtype=None):
        """Get a memory buffer with automatic pool management."""
        if self.memory_pool:
            return self.memory_pool.get_buffer(size, dtype)
        else:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available")
            if dtype is None:
                dtype = torch.float32
            return torch.zeros(size, dtype=dtype, device="cpu")

    def return_buffer(self, tensor):
        """Return a buffer to the memory pool."""
        if self.memory_pool:
            self.memory_pool.return_buffer(tensor)

    def get_optimal_batch_size(self, model_memory_usage: float = 0.0) -> int:
        """Get optimal batch size for current memory conditions."""
        return self.batch_sizer.get_optimal_batch_size(model_memory_usage)

    def record_batch_performance(self, batch_size: int, processing_time: float, success: bool):
        """Record batch processing performance."""
        self.batch_sizer.record_batch_performance(batch_size, processing_time, success)

    def get_memory_stats(self) -> dict:
        """Get comprehensive memory statistics."""
        stats = {
            "strategy": self.strategy.value,
            "hardware_tier": self.hardware.tier.name,
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()

            stats.update(
                {
                    "total_memory_gb": total_memory / 1024 / 1024 / 1024,
                    "allocated_memory_gb": allocated_memory / 1024 / 1024 / 1024,
                    "usage_percentage": (allocated_memory / total_memory) * 100,
                    "fragmentation": self._calculate_fragmentation(),
                    "current_batch_size": self.batch_sizer.current_batch_size,
                }
            )

        if self.memory_pool:
            stats.update(self.memory_pool.get_statistics())

        return stats

    def shutdown(self):
        """Shutdown the memory manager and cleanup resources."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        if self.memory_pool:
            self.memory_pool._emergency_cleanup()

        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
        logger.info("GPU Memory Manager shutdown completed")


# Global memory manager instance
_global_memory_manager: Optional[AdvancedGPUMemoryManager] = None


def get_memory_manager() -> AdvancedGPUMemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager

    if _global_memory_manager is None:
        hardware = HardwareDetector()
        _global_memory_manager = AdvancedGPUMemoryManager(hardware)

    return _global_memory_manager


def initialize_memory_manager(
    hardware: Optional[HardwareDetector] = None, config: Optional[MemoryPoolConfig] = None
) -> AdvancedGPUMemoryManager:
    """Initialize the global memory manager with custom configuration."""
    global _global_memory_manager

    if hardware is None:
        hardware = HardwareDetector()

    _global_memory_manager = AdvancedGPUMemoryManager(hardware, config)
    return _global_memory_manager


def shutdown_memory_manager():
    """Shutdown the global memory manager."""
    global _global_memory_manager

    if _global_memory_manager:
        _global_memory_manager.shutdown()
        _global_memory_manager = None


@dataclass
class MemoryStats:
    """GPU memory statistics."""
    total: int = 0
    used: int = 0
    free: int = 0
    utilization: float = 0.0


class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self, hardware=None):
        """Initialize the GPU memory manager."""
        self.hardware = hardware
        self.lock = threading.Lock()
        self.stats_history = deque(maxlen=100)
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current GPU memory statistics."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return MemoryStats()
            
        try:
            device = torch.cuda.current_device()
            stats = MemoryStats(
                total=torch.cuda.get_device_properties(device).total_memory,
                used=torch.cuda.memory_allocated(device),
                free=torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
                utilization=torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
            )
            self.stats_history.append(stats)
            return stats
        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")
            return MemoryStats()
            
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        try:
            with self.lock:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("GPU memory optimized")
        except Exception as e:
            logger.warning(f"Failed to optimize GPU memory: {e}")
            
    def get_optimal_batch_size(self, target_memory_usage: float = 0.8) -> int:
        """Calculate optimal batch size based on available memory."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 1
            
        try:
            stats = self.get_memory_stats()
            available_memory = stats.total * (1 - stats.utilization)
            
            # Start with hardware tier's default batch size
            if self.hardware:
                base_batch_size = self.hardware.config.get("batch_size", 4)
            else:
                base_batch_size = 4
                
            # Adjust based on available memory
            memory_factor = available_memory / (stats.total * target_memory_usage)
            optimal_batch_size = max(1, int(base_batch_size * memory_factor))
            
            return optimal_batch_size
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return 1
