import logging
import os
import platform
from typing import Optional

import cv2
import psutil

logger = logging.getLogger(__name__)


class HardwareMonitor:
    """Monitors system hardware resources and capabilities."""

    def __init__(self):
        """Initialize hardware monitor."""
        self._init_gpu_support()
        self._init_cpu_info()
        self._cache_performance_tier()

    def _init_gpu_support(self) -> None:
        """Initialize GPU support detection."""
        self.has_cuda = False
        self.cuda_device_count = 0

        try:
            self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if self.has_cuda:
                self.cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
                logger.info(f"Found {self.cuda_device_count} CUDA-capable GPU(s)")
            else:
                logger.info("No CUDA-capable GPUs found")

        except Exception as e:
            logger.warning(f"Failed to check CUDA support: {str(e)}")

    def _init_cpu_info(self) -> None:
        """Initialize CPU information."""
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_freq = psutil.cpu_freq()
        self.total_memory = psutil.virtual_memory().total

        logger.info(
            f"CPU cores: {self.cpu_count}, "
            f"Max frequency: {self.cpu_freq.max:.1f}MHz, "
            f"RAM: {self.total_memory / (1024**3):.1f}GB"
        )

    def _cache_performance_tier(self) -> None:
        """Calculate and cache the system performance tier."""
        # Calculate base score from CPU
        cpu_score = (
            self.cpu_count
            * (self.cpu_freq.max / 2000.0)  # Normalize to 2GHz base
            * (self.total_memory / (8 * 1024**3))  # Normalize to 8GB RAM
        )

        # Add GPU score if available
        gpu_score = self.cuda_device_count * 2.0 if self.has_cuda else 0

        # Calculate total score
        total_score = cpu_score + gpu_score

        # Determine tier
        if total_score >= 8.0:
            self._performance_tier = "high"
        elif total_score >= 4.0:
            self._performance_tier = "medium"
        else:
            self._performance_tier = "low"

        logger.info(
            f"System performance tier: {self._performance_tier} "
            f"(Score: {total_score:.1f})"
        )

    def has_gpu_support(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if CUDA-capable GPU is available
        """
        return self.has_cuda

    def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization percentage (0-100) or None if not available
        """
        if not self.has_cuda:
            return None

        try:
            # Note: This is a simplified implementation
            # For production, use nvidia-smi or similar tools
            return psutil.gpu_percent()
        except:
            return None

    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage.

        Returns:
            CPU utilization percentage (0-100)
        """
        return psutil.cpu_percent()

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage.

        Returns:
            Memory usage percentage (0-100)
        """
        return psutil.virtual_memory().percent

    def get_performance_tier(self) -> str:
        """Get the system performance tier.

        Returns:
            Performance tier ('low', 'medium', or 'high')
        """
        return self._performance_tier

    def get_system_info(self) -> dict:
        """Get comprehensive system information.

        Returns:
            Dictionary containing system information
        """
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "cpu_count": self.cpu_count,
            "cpu_freq_max": self.cpu_freq.max,
            "total_memory_gb": self.total_memory / (1024**3),
            "has_cuda": self.has_cuda,
            "cuda_devices": self.cuda_device_count,
            "performance_tier": self._performance_tier,
        }
