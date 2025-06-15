"""
Hardware detection and configuration for SpygateAI.
"""

import logging
import platform
from enum import Enum, IntEnum
from typing import Dict, Optional

import psutil
import torch

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class HardwareTier(IntEnum):
    """Hardware tier levels for performance optimization."""

    ULTRA_LOW = 0  # 4-6GB RAM, 2-core CPU, Integrated GPU
    LOW = 1  # 8GB RAM, 4-core CPU, Integrated GPU
    MEDIUM = 2  # 12GB+ RAM, GTX 1650+, 4-6 core CPU
    HIGH = 3  # 16GB+ RAM, RTX 3060+, 6+ core CPU
    ULTRA = 4  # 32GB+ RAM, RTX 4080+, 8+ core CPU


# Hardware tier configurations
TIER_CONFIGS = {
    HardwareTier.ULTRA_LOW: {
        "model_size": "n",
        "img_size": 320,
        "batch_size": 1,
        "device": "cpu",
        "conf": 0.4,
        "target_fps": 0.2,
    },
    HardwareTier.LOW: {
        "model_size": "n",
        "img_size": 416,
        "batch_size": 2,
        "device": "auto",
        "conf": 0.3,
        "target_fps": 0.5,
    },
    HardwareTier.MEDIUM: {
        "model_size": "s",
        "img_size": 640,
        "batch_size": 4,
        "half": True,
        "quantize": True,
        "target_fps": 1.0,
    },
    HardwareTier.HIGH: {
        "model_size": "m",
        "img_size": 832,
        "batch_size": 8,
        "compile": True,
        "target_fps": 2.0,
    },
    HardwareTier.ULTRA: {
        "model_size": "l",
        "img_size": 1280,
        "batch_size": 16,
        "optimize": True,
        "target_fps": 2.5,
    },
}


class HardwareDetector:
    """Detects and classifies hardware capabilities."""

    def __init__(self):
        """Initialize hardware detector."""
        self.cpu_count = psutil.cpu_count(logical=False)
        self.total_ram = psutil.virtual_memory().total / (1024**3)  # GB
        self.has_cuda = torch.cuda.is_available()
        self.gpu_name = None
        self.gpu_memory = None
        self.tier = None
        self.config = None

        if self.has_cuda:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        # Initialize tier and config
        self.tier = self.detect_tier()
        self.config = TIER_CONFIGS[self.tier]

    def detect_tier(self) -> HardwareTier:
        """Detect hardware tier based on system capabilities.

        Returns:
            HardwareTier: Detected hardware tier
        """
        # Start with ULTRA_LOW and upgrade based on capabilities
        tier = HardwareTier.ULTRA_LOW

        # CPU Check
        if self.cpu_count >= 8:
            tier = max(tier, HardwareTier.ULTRA)
        elif self.cpu_count >= 6:
            tier = max(tier, HardwareTier.HIGH)
        elif self.cpu_count >= 4:
            tier = max(tier, HardwareTier.MEDIUM)

        # RAM Check
        if self.total_ram >= 32:
            tier = max(tier, HardwareTier.ULTRA)
        elif self.total_ram >= 16:
            tier = max(tier, HardwareTier.HIGH)
        elif self.total_ram >= 12:
            tier = max(tier, HardwareTier.MEDIUM)
        elif self.total_ram >= 8:
            tier = max(tier, HardwareTier.LOW)

        # GPU Check
        if self.has_cuda:
            if "RTX 40" in self.gpu_name:
                tier = max(tier, HardwareTier.ULTRA)
            elif "RTX 30" in self.gpu_name:
                tier = max(tier, HardwareTier.HIGH)
            elif "GTX 16" in self.gpu_name or "RTX 20" in self.gpu_name:
                tier = max(tier, HardwareTier.MEDIUM)

        logger.info(f"Hardware tier detected: {tier.name}")
        return tier

    def get_capabilities(self) -> dict:
        """Get detailed hardware capabilities.

        Returns:
            dict: Hardware capabilities
        """
        return {
            "cpu_cores": self.cpu_count,
            "total_ram_gb": self.total_ram,
            "has_cuda": self.has_cuda,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory,
            "hardware_tier": self.tier.name,
        }

    def get_device(self) -> str:
        """Get the appropriate device (cuda/cpu) for the current hardware."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def supports_feature(self, feature: str) -> bool:
        """Check if current hardware tier supports a specific feature."""
        return feature in self.config and self.config[feature]

    def get_optimal_batch_size(self) -> int:
        """Get the optimal batch size for the current hardware."""
        return self.config.get("batch_size", 1)
