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
    """Detects and monitors hardware capabilities with enhanced low-end support."""

    def __init__(self):
        """Initialize hardware detector with improved memory management."""
        self.system = platform.system().lower()
        self.has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_cuda else 0

        # Get detailed system info
        self.cpu_count = psutil.cpu_count(logical=False) or 1
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = 0.8  # 80% memory usage threshold

        # Cache hardware info
        if self.has_cuda:
            logger.info(f"Found {self.gpu_count} CUDA device(s)")
            for i in range(self.gpu_count):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        self.tier = self._determine_tier()
        logger.info(f"Hardware tier: {self.tier.name}")

        # Set up memory monitoring
        self._setup_memory_monitoring()

    def _determine_tier(self) -> HardwareTier:
        """Determine hardware tier with enhanced low-end detection."""
        # Get memory in GB
        memory_gb = self.total_memory / (1024**3)

        if self.has_cuda:
            # GPU system classification
            total_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = total_memory / (1024**3)

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
        """Set up proactive memory monitoring for low-end systems."""
        self.memory_warning_threshold = 0.7  # 70% memory usage warning
        self.memory_critical_threshold = 0.85  # 85% memory usage critical

        # Set up periodic memory checks
        if self.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            self._start_memory_monitor()

    def _start_memory_monitor(self):
        """Start background memory monitoring for low-end systems."""

        def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent >= self.memory_critical_threshold * 100:
                logger.warning("Critical memory usage detected. Triggering cleanup.")
                self.trigger_memory_cleanup()
            elif memory.percent >= self.memory_warning_threshold * 100:
                logger.info("High memory usage detected. Consider cleanup.")

        # Run memory check every 30 seconds
        import threading

        self._memory_monitor = threading.Timer(30.0, check_memory)
        self._memory_monitor.daemon = True
        self._memory_monitor.start()

    def trigger_memory_cleanup(self):
        """Trigger memory cleanup for low-end systems."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

    def get_system_memory(self) -> dict[str, float]:
        """Get system memory usage in MB."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / 1024 / 1024,
            "available": memory.available / 1024 / 1024,
            "used": memory.used / 1024 / 1024,
            "percent": memory.percent,
        }

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent()

    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        if not self.has_cuda:
            return 0.0

        try:
            memory_allocated = torch.cuda.memory_allocated(0)
            return memory_allocated / 1024 / 1024
        except Exception as e:
            logger.warning(f"Error getting GPU memory usage: {e}")
            return 0.0

    def get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if not self.has_cuda:
            return 0.0

        try:
            # This is a placeholder - actual GPU utilization would require
            # platform-specific tools like nvidia-smi
            return 0.0
        except Exception as e:
            logger.warning(f"Error getting GPU utilization: {e}")
            return 0.0

    def get_recommended_settings(self) -> dict[str, any]:
        """Get recommended settings based on hardware tier."""
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
        }

        if self.tier == HardwareTier.ULTRA:
            settings.update(
                {
                    "max_batch_size": 32,
                    "max_workers": self.cpu_count,
                    "enable_advanced_features": True,
                    "memory_efficient_mode": False,
                }
            )
        elif self.tier == HardwareTier.HIGH:
            settings.update(
                {
                    "max_batch_size": 16,
                    "max_workers": max(1, self.cpu_count - 1),
                    "enable_advanced_features": True,
                    "memory_efficient_mode": False,
                }
            )
        elif self.tier == HardwareTier.MEDIUM:
            settings.update(
                {
                    "max_batch_size": 8,
                    "max_workers": max(1, self.cpu_count - 2),
                    "enable_advanced_features": False,
                    "memory_efficient_mode": False,
                }
            )
        elif self.tier == HardwareTier.LOW:
            settings.update(
                {
                    "max_batch_size": 4,
                    "max_workers": 2,
                    "enable_advanced_features": False,
                    "memory_efficient_mode": True,
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
                }
            )

        return settings
