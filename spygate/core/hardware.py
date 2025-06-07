"""
Hardware detection and classification for SpygateAI.
Handles detection of CPU, RAM, and GPU capabilities for tier-based feature adaptation.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import cv2
import psutil
import torch

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """Hardware performance tiers for feature adaptation."""

    MINIMUM = auto()  # 8GB RAM, 4-core CPU, no GPU (0.3-0.5 FPS)
    STANDARD = auto()  # 12GB+ RAM, 4-6 core CPU, GTX 1650+ (1.0+ FPS)
    PREMIUM = auto()  # 16GB+ RAM, 6+ core CPU, RTX 3060+ (1.5-2.0+ FPS)
    PROFESSIONAL = auto()  # 32GB+ RAM, 8+ core CPU, RTX 4080+ (2.0++ FPS)


@dataclass
class HardwareSpecs:
    """Container for detected hardware specifications."""

    cpu_cores: int
    ram_gb: float
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    cuda_available: bool = False
    cudnn_available: bool = False


class HardwareDetector:
    """Detects and classifies system hardware capabilities."""

    # Tier classification thresholds
    TIER_THRESHOLDS = {
        HardwareTier.MINIMUM: {"ram_gb": 8, "cpu_cores": 4, "gpu_required": False},
        HardwareTier.STANDARD: {
            "ram_gb": 12,
            "cpu_cores": 4,
            "gpu_required": True,
            "min_vram_gb": 4,
            "min_gpu": "GTX 1650",
        },
        HardwareTier.PREMIUM: {
            "ram_gb": 16,
            "cpu_cores": 6,
            "gpu_required": True,
            "min_vram_gb": 8,
            "min_gpu": "RTX 3060",
        },
        HardwareTier.PROFESSIONAL: {
            "ram_gb": 32,
            "cpu_cores": 8,
            "gpu_required": True,
            "min_vram_gb": 12,
            "min_gpu": "RTX 4080",
        },
    }

    # GPU performance rankings (higher is better)
    GPU_RANKINGS = {
        "GTX 1650": 1,
        "GTX 1660": 2,
        "RTX 2060": 3,
        "RTX 3060": 4,
        "RTX 3070": 5,
        "RTX 3080": 6,
        "RTX 4060": 7,
        "RTX 4070": 8,
        "RTX 4080": 9,
        "RTX 4090": 10,
    }

    def __init__(self):
        """Initialize the hardware detector."""
        self.specs = self._detect_hardware()
        self.tier = self._classify_tier()
        logger.info(f"Hardware detected: {self.specs}")
        logger.info(f"Classified as {self.tier.name} tier")

    def _detect_hardware(self) -> HardwareSpecs:
        """Detect system hardware specifications."""
        try:
            # CPU detection
            cpu_cores = psutil.cpu_count(logical=False)

            # RAM detection
            ram_gb = psutil.virtual_memory().total / (1024**3)

            # GPU detection using PyTorch
            cuda_available = torch.cuda.is_available()
            gpu_name = None
            gpu_vram_gb = None

            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )

            # Check for cuDNN
            cudnn_available = cuda_available and torch.backends.cudnn.is_available()

            return HardwareSpecs(
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                gpu_name=gpu_name,
                gpu_vram_gb=gpu_vram_gb,
                cuda_available=cuda_available,
                cudnn_available=cudnn_available,
            )

        except Exception as e:
            logger.error(f"Error detecting hardware: {e}")
            # Return minimum specs as fallback
            return HardwareSpecs(cpu_cores=2, ram_gb=4)

    def _get_gpu_rank(self, gpu_name: str) -> int:
        """Get the performance rank of a GPU."""
        if not gpu_name:
            return 0

        # Find the best matching GPU model
        for model, rank in self.GPU_RANKINGS.items():
            if model.lower() in gpu_name.lower():
                return rank
        return 0

    def _classify_tier(self) -> HardwareTier:
        """Classify hardware into a performance tier."""
        if not self.specs:
            return HardwareTier.MINIMUM

        # Check tiers from highest to lowest
        for tier in reversed(HardwareTier):
            requirements = self.TIER_THRESHOLDS[tier]

            # Check basic requirements
            if (
                self.specs.ram_gb >= requirements["ram_gb"]
                and self.specs.cpu_cores >= requirements["cpu_cores"]
            ):

                # Check GPU requirements if needed
                if requirements["gpu_required"]:
                    if not (self.specs.cuda_available and self.specs.gpu_name):
                        continue

                    if self.specs.gpu_vram_gb < requirements[
                        "min_vram_gb"
                    ] or self._get_gpu_rank(self.specs.gpu_name) < self._get_gpu_rank(
                        requirements["min_gpu"]
                    ):
                        continue

                return tier

        return HardwareTier.MINIMUM

    def get_tier_capabilities(self) -> Dict[str, float]:
        """Get performance capabilities for the current tier."""
        capabilities = {
            HardwareTier.MINIMUM: {
                "target_fps": 0.3,
                "max_fps": 0.5,
                "resolution_scale": 0.75,
                "frame_skip": 3,
            },
            HardwareTier.STANDARD: {
                "target_fps": 1.0,
                "max_fps": 1.2,
                "resolution_scale": 0.9,
                "frame_skip": 2,
            },
            HardwareTier.PREMIUM: {
                "target_fps": 1.5,
                "max_fps": 2.0,
                "resolution_scale": 1.0,
                "frame_skip": 1,
            },
            HardwareTier.PROFESSIONAL: {
                "target_fps": 2.0,
                "max_fps": 3.0,
                "resolution_scale": 1.0,
                "frame_skip": 1,
            },
        }
        return capabilities[self.tier]

    def get_tier_features(self) -> Dict[str, bool]:
        """Get available features for the current tier."""
        features = {
            "enhanced_cv": True,  # Available on all tiers
            "yolo_detection": self.tier != HardwareTier.MINIMUM,
            "real_time_analysis": self.tier
            in [HardwareTier.PREMIUM, HardwareTier.PROFESSIONAL],
            "advanced_formations": self.tier
            in [HardwareTier.PREMIUM, HardwareTier.PROFESSIONAL],
            "experimental": self.tier == HardwareTier.PROFESSIONAL,
        }
        return features
