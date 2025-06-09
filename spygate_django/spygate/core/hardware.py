"""
Hardware detection and monitoring.

This module provides functionality to detect and monitor hardware capabilities.
"""

import logging
import platform
from enum import Enum
from typing import Dict, Optional

import psutil

try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """Hardware tier levels with more granular detection."""

    ULTRA_LOW = "ultra_low"  # Below minimum specs
    LOW = "low"  # Minimum specs
    MEDIUM = "medium"  # Standard specs
    HIGH = "high"  # Premium specs
    ULTRA = "ultra"  # Professional specs


class HardwareDetector:
    """Detects and monitors hardware capabilities with enhanced GPU memory management."""

    def __init__(self):
        """Initialize hardware detector with improved memory management."""
        self.system = platform.system().lower()
        self.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_cuda else 0

        # Get detailed system info
        self.cpu_count = psutil.cpu_count(logical=False) or 1
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = 0.8  # 80% memory usage threshold

        # GPU memory information
        self.gpu_memory_total = 0
        self.gpu_memory_reserved = 0
        self.gpu_name = "N/A"

        if self.has_cuda:
            try:
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                self.gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Found {self.gpu_count} CUDA device(s)")
                logger.info(f"  Device 0: {self.gpu_name}")
                logger.info(f"  GPU Memory: {self.gpu_memory_total / 1024**3:.2f} GB")
            except Exception as e:
                logger.warning(f"Error getting GPU properties: {e}")

        self.tier = self._determine_tier()
        logger.info(f"Hardware tier: {self.tier.name}")

        # Set up enhanced memory monitoring
        self._setup_memory_monitoring()

    def _determine_tier(self) -> HardwareTier:
        """Determine hardware tier with enhanced low-end detection."""
        # Get memory in GB
        memory_gb = self.total_memory / (1024**3)

        if self.has_cuda:
            # GPU system classification
            gpu_memory_gb = self.gpu_memory_total / (1024**3)

            if gpu_memory_gb >= 8:
                return HardwareTier.ULTRA
            elif gpu_memory_gb >= 4:
                return HardwareTier.HIGH
            elif gpu_memory_gb >= 2:
                return HardwareTier.MEDIUM

        # CPU-only or low-end GPU classification
        if memory_gb >= 16 and self.cpu_count >= 6:
            return HardwareTier.HIGH
        elif memory_gb >= 8 and self.cpu_count >= 4:
            return HardwareTier.MEDIUM
        elif memory_gb >= 6 and self.cpu_count >= 2:
            return HardwareTier.LOW
        else:
            return HardwareTier.ULTRA_LOW

    def _setup_memory_monitoring(self):
        """Set up proactive memory monitoring for all systems."""
        # Enhanced thresholds based on tier
        if self.tier == HardwareTier.ULTRA_LOW:
            self.memory_warning_threshold = 0.6  # 60% for ultra-low systems
            self.memory_critical_threshold = 0.75  # 75% critical
        elif self.tier == HardwareTier.LOW:
            self.memory_warning_threshold = 0.7  # 70% for low systems
            self.memory_critical_threshold = 0.85  # 85% critical
        else:
            self.memory_warning_threshold = 0.75  # 75% for higher tiers
            self.memory_critical_threshold = 0.90  # 90% critical

        # GPU memory thresholds
        self.gpu_warning_threshold = 0.8  # 80% GPU memory warning
        self.gpu_critical_threshold = 0.9  # 90% GPU memory critical

        # Start monitoring for all tiers now (not just low-end)
        self._start_memory_monitor()

    def _start_memory_monitor(self):
        """Start background memory monitoring for all systems."""

        def check_memory():
            # System memory check
            memory = psutil.virtual_memory()
            if memory.percent >= self.memory_critical_threshold * 100:
                logger.warning("Critical system memory usage detected. Triggering cleanup.")
                self.trigger_memory_cleanup()
            elif memory.percent >= self.memory_warning_threshold * 100:
                logger.info(f"High system memory usage: {memory.percent:.1f}%")

            # GPU memory check
            if self.has_cuda:
                try:
                    gpu_allocated = torch.cuda.memory_allocated(0)
                    gpu_usage_ratio = gpu_allocated / self.gpu_memory_total

                    if gpu_usage_ratio >= self.gpu_critical_threshold:
                        logger.warning(f"Critical GPU memory usage: {gpu_usage_ratio:.1%}")
                        self.trigger_gpu_memory_cleanup()
                    elif gpu_usage_ratio >= self.gpu_warning_threshold:
                        logger.info(f"High GPU memory usage: {gpu_usage_ratio:.1%}")
                except Exception as e:
                    logger.debug(f"Error checking GPU memory: {e}")

        # Run memory check every 30 seconds for low-end, 60 seconds for others
        import threading

        interval = 30.0 if self.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW] else 60.0

        def start_timer():
            check_memory()
            timer = threading.Timer(interval, start_timer)
            timer.daemon = True
            timer.start()
            return timer

        self._memory_monitor = start_timer()

    def trigger_memory_cleanup(self):
        """Enhanced memory cleanup with tier-specific strategies."""
        logger.info("Triggering system memory cleanup")

        # GPU cleanup first if available
        if self.has_cuda:
            self.trigger_gpu_memory_cleanup()

        # System garbage collection
        import gc

        gc.collect()

        # Additional cleanup for low-end systems
        if self.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            # More aggressive cleanup
            gc.collect()  # Run GC twice for low-end systems

    def trigger_gpu_memory_cleanup(self):
        """Enhanced GPU memory cleanup with fragmentation handling."""
        if not self.has_cuda:
            return

        logger.info("Triggering GPU memory cleanup")

        try:
            # Get pre-cleanup stats
            allocated_before = torch.cuda.memory_allocated(0)
            reserved_before = torch.cuda.memory_reserved(0)

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Get post-cleanup stats
            allocated_after = torch.cuda.memory_allocated(0)
            reserved_after = torch.cuda.memory_reserved(0)

            freed_allocated = allocated_before - allocated_after
            freed_reserved = reserved_before - reserved_after

            if freed_allocated > 0 or freed_reserved > 0:
                logger.info(
                    f"GPU cleanup freed: "
                    f"{freed_allocated / 1024**2:.1f} MB allocated, "
                    f"{freed_reserved / 1024**2:.1f} MB reserved"
                )

        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")

    def get_system_memory(self) -> dict[str, float]:
        """Get enhanced system memory usage in MB."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / 1024 / 1024,
            "available": memory.available / 1024 / 1024,
            "used": memory.used / 1024 / 1024,
            "percent": memory.percent,
            "warning_threshold": self.memory_warning_threshold * 100,
            "critical_threshold": self.memory_critical_threshold * 100,
        }

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent()

    def get_gpu_memory_usage(self) -> dict[str, float]:
        """Get comprehensive GPU memory usage in MB."""
        if not self.has_cuda:
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "total": 0.0,
                "percent_allocated": 0.0,
                "percent_reserved": 0.0,
                "fragmentation": 0.0,
            }

        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = self.gpu_memory_total

            # Calculate fragmentation
            fragmentation = 0.0
            if reserved > 0:
                fragmentation = (reserved - allocated) / reserved * 100

            return {
                "allocated": allocated / 1024 / 1024,
                "reserved": reserved / 1024 / 1024,
                "total": total / 1024 / 1024,
                "percent_allocated": (allocated / total) * 100,
                "percent_reserved": (reserved / total) * 100,
                "fragmentation": fragmentation,
                "warning_threshold": self.gpu_warning_threshold * 100,
                "critical_threshold": self.gpu_critical_threshold * 100,
            }
        except Exception as e:
            logger.warning(f"Error getting GPU memory usage: {e}")
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "total": 0.0,
                "percent_allocated": 0.0,
                "percent_reserved": 0.0,
                "fragmentation": 0.0,
            }

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if not self.has_cuda:
            return 0.0

        try:
            # This is a placeholder - actual GPU utilization would require
            # platform-specific tools like nvidia-smi or nvidia-ml-py
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting GPU utilization: {e}")
            return 0.0

    def get_recommended_settings(self) -> dict[str, any]:
        """Get enhanced recommended settings based on hardware tier."""
        settings = {
            "use_gpu": self.has_cuda,
            "parallel_processing": True,
            "max_batch_size": 1,
            "enable_caching": True,
            "max_workers": 2,
            "enable_advanced_features": False,
            "memory_efficient_mode": (
                True if self.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW] else False
            ),
            "gpu_memory_fraction": 0.8,  # Use up to 80% of GPU memory
            "enable_mixed_precision": False,
            "gradient_checkpointing": False,
        }

        if self.tier == HardwareTier.ULTRA:
            settings.update(
                {
                    "max_batch_size": 32,
                    "max_workers": self.cpu_count,
                    "enable_advanced_features": True,
                    "memory_efficient_mode": False,
                    "gpu_memory_fraction": 0.9,
                    "enable_mixed_precision": True,
                    "gradient_checkpointing": False,
                }
            )
        elif self.tier == HardwareTier.HIGH:
            settings.update(
                {
                    "max_batch_size": 16,
                    "max_workers": max(1, self.cpu_count - 1),
                    "enable_advanced_features": True,
                    "memory_efficient_mode": False,
                    "gpu_memory_fraction": 0.85,
                    "enable_mixed_precision": True,
                    "gradient_checkpointing": False,
                }
            )
        elif self.tier == HardwareTier.MEDIUM:
            settings.update(
                {
                    "max_batch_size": 8,
                    "max_workers": max(1, self.cpu_count - 2),
                    "enable_advanced_features": False,
                    "memory_efficient_mode": False,
                    "gpu_memory_fraction": 0.8,
                    "enable_mixed_precision": True,
                    "gradient_checkpointing": True,
                }
            )
        elif self.tier == HardwareTier.LOW:
            settings.update(
                {
                    "max_batch_size": 4,
                    "max_workers": 2,
                    "enable_advanced_features": False,
                    "memory_efficient_mode": True,
                    "gpu_memory_fraction": 0.7,
                    "enable_mixed_precision": False,
                    "gradient_checkpointing": True,
                }
            )
        else:  # ULTRA_LOW
            settings.update(
                {
                    "max_batch_size": 2,
                    "max_workers": 1,
                    "enable_advanced_features": False,
                    "memory_efficient_mode": True,
                    "parallel_processing": False,  # Disable parallel processing for very low-end systems
                    "gpu_memory_fraction": 0.6,
                    "enable_mixed_precision": False,
                    "gradient_checkpointing": True,
                }
            )

        return settings

    def get_comprehensive_stats(self) -> dict[str, any]:
        """Get comprehensive hardware and memory statistics."""
        stats = {
            "hardware_tier": self.tier.name,
            "cpu_count": self.cpu_count,
            "system_memory": self.get_system_memory(),
            "cpu_usage": self.get_cpu_usage(),
            "gpu_available": self.has_cuda,
            "gpu_count": self.gpu_count,
            "gpu_name": self.gpu_name,
        }

        if self.has_cuda:
            stats["gpu_memory"] = self.get_gpu_memory_usage()
            stats["gpu_utilization"] = self.get_gpu_utilization()

        return stats
