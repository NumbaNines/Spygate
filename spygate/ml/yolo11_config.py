"""YOLO11 model configuration for HUD element detection."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from ..core.hardware import HardwareTier


class ModelSize(Enum):
    """Available model size configurations."""
    NANO = "nano"      # For ultra-low-end systems
    SMALL = "small"    # For low-end systems
    MEDIUM = "medium"  # For mid-range systems
    LARGE = "large"    # For high-end systems


@dataclass
class BackboneConfig:
    """Configuration for model backbone."""
    depth_multiple: float
    width_multiple: float
    input_channels: int = 3
    input_size: Tuple[int, int] = (640, 640)


@dataclass
class HeadConfig:
    """Configuration for model detection head."""
    num_classes: int = 10  # Number of HUD element classes
    anchors: int = 3      # Number of anchors per output
    num_layers: int = 3   # Number of detection layers


@dataclass
class YOLO11Config:
    """Complete YOLO11 model configuration."""
    model_size: ModelSize
    backbone: BackboneConfig
    head: HeadConfig
    hardware_tier: HardwareTier
    batch_size: int
    workers: int
    use_amp: bool = False
    cache_images: bool = False
    multi_scale: bool = False

    @classmethod
    def from_hardware(cls, hardware_tier: HardwareTier) -> 'YOLO11Config':
        """Create configuration based on hardware tier."""
        configs = {
            HardwareTier.ULTRA_LOW: {
                "model_size": ModelSize.NANO,
                "backbone": BackboneConfig(
                    depth_multiple=0.33,
                    width_multiple=0.25,
                    input_size=(416, 416)
                ),
                "batch_size": 1,
                "workers": 1,
                "use_amp": False,
                "cache_images": False,
                "multi_scale": False
            },
            HardwareTier.LOW: {
                "model_size": ModelSize.SMALL,
                "backbone": BackboneConfig(
                    depth_multiple=0.33,
                    width_multiple=0.50,
                    input_size=(512, 512)
                ),
                "batch_size": 2,
                "workers": 2,
                "use_amp": True,
                "cache_images": False,
                "multi_scale": False
            },
            HardwareTier.MEDIUM: {
                "model_size": ModelSize.MEDIUM,
                "backbone": BackboneConfig(
                    depth_multiple=0.67,
                    width_multiple=0.75,
                    input_size=(640, 640)
                ),
                "batch_size": 4,
                "workers": 4,
                "use_amp": True,
                "cache_images": True,
                "multi_scale": True
            },
            HardwareTier.HIGH: {
                "model_size": ModelSize.LARGE,
                "backbone": BackboneConfig(
                    depth_multiple=1.0,
                    width_multiple=1.0,
                    input_size=(640, 640)
                ),
                "batch_size": 8,
                "workers": 8,
                "use_amp": True,
                "cache_images": True,
                "multi_scale": True
            }
        }
        
        config = configs[hardware_tier]
        return cls(
            model_size=config["model_size"],
            backbone=config["backbone"],
            head=HeadConfig(),  # Default head config
            hardware_tier=hardware_tier,
            batch_size=config["batch_size"],
            workers=config["workers"],
            use_amp=config["use_amp"],
            cache_images=config["cache_images"],
            multi_scale=config["multi_scale"]
        )

    def get_model_hyperparameters(self) -> Dict:
        """Get model hyperparameters based on configuration."""
        base_lr = {
            ModelSize.NANO: 0.001,
            ModelSize.SMALL: 0.001,
            ModelSize.MEDIUM: 0.01,
            ModelSize.LARGE: 0.01
        }[self.model_size]

        return {
            "lr0": base_lr,
            "lrf": 0.01,  # Final learning rate (lr0 * lrf)
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 0.05,
            "cls": 0.5,
            "cls_pw": 1.0,
            "obj": 1.0,
            "obj_pw": 1.0,
            "iou_t": 0.20,
            "anchor_t": 4.0,
            "fl_gamma": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,  # No rotation for HUD elements
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0
        }

    def get_training_settings(self) -> Dict:
        """Get training settings based on configuration."""
        return {
            "epochs": {
                ModelSize.NANO: 50,
                ModelSize.SMALL: 100,
                ModelSize.MEDIUM: 150,
                ModelSize.LARGE: 200
            }[self.model_size],
            "batch_size": self.batch_size,
            "num_workers": self.workers,
            "use_amp": self.use_amp,
            "cache_images": self.cache_images,
            "multi_scale": self.multi_scale,
            "rect": False,  # Rectangular training
            "image_weights": False,
            "single_cls": False,
            "adam": False,  # Use AdamW
            "sync_bn": False,  # Use normal BatchNorm
            "workers": self.workers,
            "project": "spygate/models/yolo11",
            "name": f"hud_detector_{self.model_size.value}",
            "exist_ok": False,
            "quad": False,
            "linear_lr": False,
            "label_smoothing": 0.0,
            "patience": 100,
            "freeze": [0],  # Freeze first layer
            "save_period": -1,
            "local_rank": -1
        } 