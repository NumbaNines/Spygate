"""
System configuration management.

This module provides configuration management for the system.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class Config:
    """Manages system configuration."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Optional path to config file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        
        # Load default config
        self._load_defaults()
        
        # Load from file if provided
        if self.config_path and self.config_path.exists():
            self._load_from_file()
    
    def _load_defaults(self):
        """Load default configuration."""
        self._config = {
            "performance": {
                "target_fps": 30.0,
                "min_fps": 20.0,
                "max_memory_mb": 2048.0,
                "max_gpu_memory_mb": 1024.0,
                "memory_warning_threshold": 0.9,
                "gpu_warning_threshold": 0.9,
                "max_processing_time": 0.05,
                "max_batch_time": 0.2,
                "min_quality": 0.5,
                "max_quality": 1.0,
                "quality_step": 0.1,
                "fps_buffer_size": 100,
                "metrics_interval": 1.0,
                "cleanup_interval": 100
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "max_prediction_age": 5,
                "max_tracks": 100,
                "track_buffer_size": 30
            },
            "preprocessing": {
                "target_size": (640, 640),
                "normalize": True,
                "batch_size": 32,
                "num_workers": 4
            },
            "visualization": {
                "draw_tracks": True,
                "draw_labels": True,
                "draw_confidence": True,
                "draw_fps": True,
                "draw_memory": True,
                "label_font_scale": 0.5,
                "label_thickness": 1,
                "box_thickness": 2
            },
            "logging": {
                "level": "INFO",
                "file": "spygate.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _load_from_file(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, "r") as f:
                file_config = json.load(f)
                
            # Update config recursively
            self._update_dict(self._config, file_config)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
    
    def _update_dict(self, d: Dict, u: Dict) -> Dict:
        """Update dictionary recursively."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to file.
        
        Args:
            path: Optional path to save to, defaults to config_path
        """
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No save path specified")
            
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(self._config, f, indent=4)
            logger.info(f"Saved configuration to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration section."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"No such config section: {name}")
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration.
        
        Args:
            updates: Dictionary of updates
        """
        self._update_dict(self._config, updates) 