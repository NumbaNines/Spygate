"""
Core functionality for SpygateAI.
"""

__version__ = "0.0.1"

from ..models.tracking import TrackingData
from .gpu_memory_manager import GPUMemoryManager
from .hardware import HardwareDetector, HardwareTier
from .tracking_pipeline import TrackingPipeline

__all__ = [
    "HardwareDetector",
    "HardwareTier",
    "GPUMemoryManager",
    "TrackingPipeline",
    "TrackingData",
]
