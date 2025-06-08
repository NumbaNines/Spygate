"""
Hardware detection module for determining system capabilities.
"""

import logging
import os
import platform
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    name: str
    vram_total: int  # in MB
    vram_free: int  # in MB
    cuda_capability: Optional[float] = None
    is_integrated: bool = False


@dataclass
class SystemInfo:
    """Information about the system hardware."""

    cpu_count: int
    cpu_threads: int
    ram_total: int  # in MB
    ram_free: int  # in MB
    gpus: List[GPUInfo]
    platform: str
    cuda_available: bool
    opencv_gpu: bool


class HardwareDetector:
    """
    Detects and monitors system hardware capabilities.
    Provides information about CPU, RAM, and GPU resources.
    """

    def __init__(self):
        """Initialize the hardware detector."""
        self._system_info: Optional[SystemInfo] = None
        self._performance_tier: Optional[str] = None
        self.refresh()

    def refresh(self):
        """Refresh hardware information."""
        try:
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)

            # Get RAM info
            ram = psutil.virtual_memory()
            ram_total = ram.total // (1024 * 1024)  # Convert to MB
            ram_free = ram.available // (1024 * 1024)

            # Get platform info
            platform_name = platform.system()

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()

            # Check OpenCV GPU support
            opencv_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

            # Get GPU info
            gpus = self._get_gpu_info()

            # Create system info
            self._system_info = SystemInfo(
                cpu_count=cpu_count,
                cpu_threads=cpu_threads,
                ram_total=ram_total,
                ram_free=ram_free,
                gpus=gpus,
                platform=platform_name,
                cuda_available=cuda_available,
                opencv_gpu=opencv_gpu,
            )

            # Determine performance tier
            self._performance_tier = self._calculate_performance_tier()

        except Exception as e:
            logger.error(f"Hardware detection error: {e}", exc_info=True)
            # Set fallback values
            self._system_info = SystemInfo(
                cpu_count=1,
                cpu_threads=2,
                ram_total=4096,  # Assume 4GB
                ram_free=1024,  # Assume 1GB free
                gpus=[],
                platform=platform.system(),
                cuda_available=False,
                opencv_gpu=False,
            )
            self._performance_tier = "minimum"

    def _get_gpu_info(self) -> List[GPUInfo]:
        """Get information about available GPUs."""
        gpus = []

        # Try CUDA GPUs first
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_total = props.total_memory // (1024 * 1024)  # Convert to MB
                # Get free VRAM - this is approximate
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                vram_free = vram_total - (
                    torch.cuda.memory_allocated() // (1024 * 1024)
                )

                gpus.append(
                    GPUInfo(
                        name=props.name,
                        vram_total=vram_total,
                        vram_free=vram_free,
                        cuda_capability=props.major + props.minor / 10,
                        is_integrated="integrated" in props.name.lower(),
                    )
                )

        return gpus

    def _calculate_performance_tier(self) -> str:
        """
        Calculate the system's performance tier based on hardware capabilities.
        Returns: "minimum", "standard", "premium", or "professional"
        """
        if not self._system_info:
            return "minimum"

        # Get the best GPU (most VRAM)
        best_gpu = None
        if self._system_info.gpus:
            best_gpu = max(self._system_info.gpus, key=lambda g: g.vram_total)

        # Professional Tier Requirements
        if (
            best_gpu
            and best_gpu.vram_total >= 12288  # 12GB VRAM
            and self._system_info.ram_total >= 32768  # 32GB RAM
            and self._system_info.cpu_threads >= 16
        ):  # 16 threads
            return "professional"

        # Premium Tier Requirements
        if (
            best_gpu
            and best_gpu.vram_total >= 8192  # 8GB VRAM
            and self._system_info.ram_total >= 16384  # 16GB RAM
            and self._system_info.cpu_threads >= 8
        ):  # 8 threads
            return "premium"

        # Standard Tier Requirements
        if (
            best_gpu
            and best_gpu.vram_total >= 4096  # 4GB VRAM
            and self._system_info.ram_total >= 8192  # 8GB RAM
            and self._system_info.cpu_threads >= 4
        ):  # 4 threads
            return "standard"

        # Minimum Tier (anything below standard)
        return "minimum"

    @property
    def system_info(self) -> SystemInfo:
        """Get current system information."""
        if not self._system_info:
            self.refresh()
        return self._system_info

    @property
    def performance_tier(self) -> str:
        """Get the system's performance tier."""
        if not self._performance_tier:
            self.refresh()
        return self._performance_tier

    @property
    def has_cuda(self) -> bool:
        """Check if CUDA is available."""
        return self.system_info.cuda_available

    @property
    def has_opencv_gpu(self) -> bool:
        """Check if OpenCV GPU acceleration is available."""
        return self.system_info.opencv_gpu

    def get_optimal_thread_count(self) -> int:
        """Get the optimal number of threads for parallel processing."""
        if not self._system_info:
            return 2

        # Use 75% of available threads by default
        return max(2, int(self._system_info.cpu_threads * 0.75))

    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size based on available memory."""
        if not self._system_info:
            return 4

        # Calculate based on available RAM and performance tier
        if self.performance_tier == "professional":
            return 32
        elif self.performance_tier == "premium":
            return 16
        elif self.performance_tier == "standard":
            return 8
        else:
            return 4

    def get_vram_info(self) -> Dict[str, int]:
        """Get information about VRAM usage."""
        if not self.has_cuda:
            return {"total": 0, "free": 0, "used": 0}

        best_gpu = max(self._system_info.gpus, key=lambda g: g.vram_total)
        used = best_gpu.vram_total - best_gpu.vram_free
        return {"total": best_gpu.vram_total, "free": best_gpu.vram_free, "used": used}

    def get_ram_info(self) -> Dict[str, int]:
        """Get information about RAM usage."""
        if not self._system_info:
            return {"total": 0, "free": 0, "used": 0}

        used = self._system_info.ram_total - self._system_info.ram_free
        return {
            "total": self._system_info.ram_total,
            "free": self._system_info.ram_free,
            "used": used,
        }
