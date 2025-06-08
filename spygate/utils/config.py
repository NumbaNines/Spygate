"""
Configuration management for Spygate application.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "storage": {
        "videos_dir": "data/videos",
        "max_temp_age_hours": 24,
        "supported_formats": [".mp4", ".mov", ".avi"],
        "max_file_size_gb": 2.0,
    },
    "database": {"path": "data/spygate.db"},
    "logging": {"level": "INFO", "file": "logs/spygate.log"},
}

_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Get the application configuration. Loads from file if not already loaded.

    Returns:
        Dict[str, Any]: Application configuration
    """
    global _config

    if _config is not None:
        return _config

    config_path = os.environ.get("SPYGATE_CONFIG", "config/config.json")

    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                _config = _merge_configs(DEFAULT_CONFIG, loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            _config = DEFAULT_CONFIG.copy()
            logger.warning(f"No config file found at {config_path}, using defaults")

            # Create default config file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logger.info(f"Created default config file at {config_path}")

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        _config = DEFAULT_CONFIG.copy()

    return _config


def _merge_configs(default: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.

    Args:
        default: Default configuration dictionary
        override: Override configuration dictionary

    Returns:
        Dict: Merged configuration
    """
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def update_config(updates: Dict[str, Any], save: bool = True) -> None:
    """
    Update the application configuration.

    Args:
        updates: Dictionary of configuration updates
        save: Whether to save the updates to the config file
    """
    config = get_config()
    new_config = _merge_configs(config, updates)

    global _config
    _config = new_config

    if save:
        config_path = os.environ.get("SPYGATE_CONFIG", "config/config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        try:
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=4)
            logger.info(f"Saved configuration updates to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")


def reset_config() -> None:
    """Reset the configuration to default values."""
    global _config
    _config = DEFAULT_CONFIG.copy()

    config_path = os.environ.get("SPYGATE_CONFIG", "config/config.json")

    try:
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logger.info(f"Reset configuration to defaults and saved to {config_path}")
    except Exception as e:
        logger.error(f"Error resetting config at {config_path}: {e}")
