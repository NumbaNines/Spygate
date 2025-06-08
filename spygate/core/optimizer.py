"""
Performance optimization module based on hardware capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .hardware import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Settings profile for performance optimization."""
    
    batch_size: int
    num_workers: int
    use_gpu: bool
    enable_caching: bool
    enable_parallel: bool
    enable_advanced_features: bool
    # Added optimization parameters
    max_memory_usage: float  # in MB
    target_fps: float
    min_fps: float
    quality_scale: float  # 0.5 to 1.0
    prefetch_size: int
    cleanup_interval: int
    max_prediction_age: int


class TierOptimizer:
    """Optimizes processing settings based on hardware tier."""
    
    def __init__(self, hardware: HardwareDetector):
        """Initialize optimizer with hardware detector."""
        self.hardware = hardware
        self.profile = self._create_profile()
        
        logger.info(
            f"Created optimization profile for {hardware.tier.name} tier"
        )

    def _create_profile(self) -> OptimizationProfile:
        """Create optimization profile based on hardware tier."""
        settings = self.hardware.get_recommended_settings()
        
        # Base profile with hardware-specific settings
        return OptimizationProfile(
            batch_size=settings["max_batch_size"],
            num_workers=settings["max_workers"],
            use_gpu=settings["use_gpu"],
            enable_caching=settings["enable_caching"],
            enable_parallel=settings["parallel_processing"],
            enable_advanced_features=settings["enable_advanced_features"],
            max_memory_usage=self._get_memory_limit(),
            target_fps=self._get_target_fps(),
            min_fps=20.0,  # Minimum acceptable FPS
            quality_scale=1.0,  # Start with maximum quality
            prefetch_size=4,  # Default prefetch buffer size
            cleanup_interval=100,  # Frames between memory cleanup
            max_prediction_age=30  # Maximum frames to keep predictions
        )

    def _get_memory_limit(self) -> float:
        """Get memory limit based on hardware tier."""
        limits = {
            HardwareTier.ULTRA: 8192.0,  # 8GB
            HardwareTier.HIGH: 4096.0,   # 4GB
            HardwareTier.MEDIUM: 2048.0, # 2GB
            HardwareTier.LOW: 1024.0     # 1GB
        }
        return limits.get(self.hardware.tier, 2048.0)

    def _get_target_fps(self) -> float:
        """Get target FPS based on hardware tier."""
        targets = {
            HardwareTier.ULTRA: 60.0,
            HardwareTier.HIGH: 45.0,
            HardwareTier.MEDIUM: 30.0,
            HardwareTier.LOW: 24.0
        }
        return targets.get(self.hardware.tier, 30.0)

    def get_batch_size(self, task_type: str) -> int:
        """Get optimal batch size for specific task type."""
        base_size = self.profile.batch_size
        
        # Adjust based on task type
        multipliers = {
            "video_processing": 1.0,
            "formation_analysis": 0.5,  # More complex, smaller batches
            "player_detection": 0.75,
            "motion_tracking": 0.75,
            "visualization": 1.0,
            "heat_map": 0.5,
            "ball_tracking": 0.75
        }
        
        multiplier = multipliers.get(task_type, 1.0)
        return max(1, int(base_size * multiplier))

    def get_worker_count(self, task_type: str) -> int:
        """Get optimal number of workers for specific task type."""
        base_workers = self.profile.num_workers
        
        # Adjust based on task type
        multipliers = {
            "video_processing": 1.0,
            "formation_analysis": 0.75,  # CPU intensive
            "player_detection": 1.0,
            "motion_tracking": 0.75,
            "visualization": 0.5,  # GPU bound
            "heat_map": 0.75,
            "ball_tracking": 0.75
        }
        
        multiplier = multipliers.get(task_type, 1.0)
        return max(1, int(base_workers * multiplier))

    def should_use_gpu(self, task_type: str) -> bool:
        """Determine if GPU should be used for specific task type."""
        if not self.profile.use_gpu:
            return False
            
        # Some tasks might not benefit from GPU
        gpu_beneficial = {
            "video_processing": True,
            "formation_analysis": True,
            "player_detection": True,
            "motion_tracking": True,
            "visualization": True,
            "heat_map": True,
            "ball_tracking": True,
            "data_processing": False,
        }
        
        return gpu_beneficial.get(task_type, True)

    def get_cache_config(self, task_type: str) -> Dict[str, any]:
        """Get caching configuration for specific task type."""
        if not self.profile.enable_caching:
            return {"enabled": False}
            
        # Define cache settings based on hardware tier
        if self.hardware.tier == HardwareTier.ULTRA:
            return {
                "enabled": True,
                "max_size": "8GB",
                "ttl": 3600,  # 1 hour
                "compression": False,
                "prefetch": True,
                "prefetch_size": 8
            }
        elif self.hardware.tier == HardwareTier.HIGH:
            return {
                "enabled": True,
                "max_size": "4GB",
                "ttl": 1800,  # 30 minutes
                "compression": False,
                "prefetch": True,
                "prefetch_size": 6
            }
        elif self.hardware.tier == HardwareTier.MEDIUM:
            return {
                "enabled": True,
                "max_size": "2GB",
                "ttl": 900,  # 15 minutes
                "compression": True,
                "prefetch": True,
                "prefetch_size": 4
            }
        else:  # LOW
            return {
                "enabled": True,
                "max_size": "1GB",
                "ttl": 300,  # 5 minutes
                "compression": True,
                "prefetch": True,
                "prefetch_size": 2
            }

    def get_task_settings(
        self,
        task_type: str,
        override: Optional[Dict] = None,
    ) -> Dict[str, any]:
        """Get optimized settings for a specific task."""
        settings = {
            "batch_size": self.get_batch_size(task_type),
            "num_workers": self.get_worker_count(task_type),
            "use_gpu": self.should_use_gpu(task_type),
            "enable_parallel": self.profile.enable_parallel,
            "enable_advanced": self.profile.enable_advanced_features,
            "caching": self.get_cache_config(task_type),
            "max_memory": self.profile.max_memory_usage,
            "target_fps": self.profile.target_fps,
            "min_fps": self.profile.min_fps,
            "quality_scale": self.profile.quality_scale,
            "prefetch_size": self.profile.prefetch_size,
            "cleanup_interval": self.profile.cleanup_interval,
            "max_prediction_age": self.profile.max_prediction_age
        }
        
        # Apply any overrides
        if override:
            settings.update(override)
            
        return settings

    def adjust_quality(self, current_fps: float, memory_usage: float) -> float:
        """Dynamically adjust quality based on performance metrics."""
        quality = self.profile.quality_scale
        
        # Adjust for FPS
        if current_fps < self.profile.min_fps:
            quality = max(0.5, quality - 0.1)  # Reduce quality
        elif current_fps > self.profile.target_fps * 1.2:
            quality = min(1.0, quality + 0.1)  # Increase quality
            
        # Adjust for memory
        if memory_usage > self.profile.max_memory_usage * 0.9:
            quality = max(0.5, quality - 0.1)  # Reduce quality
            
        return quality

    def get_performance_thresholds(self) -> Dict[str, float]:
        """Get performance thresholds for monitoring."""
        return {
            "min_fps": self.profile.min_fps,
            "target_fps": self.profile.target_fps,
            "max_memory_mb": self.profile.max_memory_usage,
            "memory_warning": self.profile.max_memory_usage * 0.9,
            "min_quality": 0.5,
            "max_quality": 1.0,
            "batch_warning": self.profile.batch_size * 0.8
        }

    def update_profile(self, hardware: Optional[HardwareDetector] = None):
        """Update optimization profile with new hardware info."""
        if hardware:
            self.hardware = hardware
        self.profile = self._create_profile()
        
        logger.info(
            f"Updated optimization profile for {self.hardware.tier.name} tier"
        )
