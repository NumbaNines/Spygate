"""
Core functionality for SpygateAI.
"""

__version__ = "0.0.1"

from .hardware import HardwareDetector, HardwareTier
from .gpu_memory_manager import GPUMemoryManager
from .tracking_pipeline import TrackingPipeline
from ..models.tracking import TrackingData

__all__ = [
    "HardwareDetector",
    "HardwareTier",
    "GPUMemoryManager",
    "TrackingPipeline",
    "TrackingData"
]
