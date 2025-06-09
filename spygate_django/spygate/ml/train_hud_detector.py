"""Train YOLO11 model for HUD element detection with low-end system support."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ultralytics import YOLO

from ..core.hardware import HardwareDetector, HardwareTier
from .yolov8_model import EnhancedYOLOv8 as CustomYOLO11  # Using YOLOv8 as specified in PRD

logger = logging.getLogger(__name__)


def setup_training_environment() -> tuple[HardwareDetector, dict[str, any]]:
    """Set up the training environment with hardware-aware optimization.

    Returns:
        Tuple containing:
        - HardwareDetector instance
        - Training configuration dictionary
    """
    hardware = HardwareDetector()

    # Base configuration
    config = {
        "batch_size": 1,
        "image_size": 640,
        "epochs": 100,
        "workers": 2,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "use_amp": False,
        "gradient_accumulation": 1,
        "cache_images": False,
        "multi_scale": False,
    }

    # Adjust settings based on hardware tier
    if hardware.tier == HardwareTier.ULTRA:
        config.update(
            {
                "batch_size": 32,
                "image_size": 1280,
                "workers": hardware.cpu_count,
                "use_amp": True,
                "cache_images": True,
                "multi_scale": True,
            }
        )
    elif hardware.tier == HardwareTier.HIGH:
        config.update(
            {
                "batch_size": 16,
                "image_size": 1024,
                "workers": max(1, hardware.cpu_count - 1),
                "use_amp": True,
                "cache_images": True,
            }
        )
    elif hardware.tier == HardwareTier.MEDIUM:
        config.update(
            {
                "batch_size": 8,
                "image_size": 832,
                "workers": max(1, hardware.cpu_count - 2),
                "use_amp": True,
            }
        )
    elif hardware.tier == HardwareTier.LOW:
        config.update(
            {"batch_size": 4, "image_size": 640, "workers": 2, "gradient_accumulation": 2}
        )
    else:  # ULTRA_LOW
        config.update(
            {
                "batch_size": 2,
                "image_size": 512,
                "workers": 1,
                "gradient_accumulation": 4,
                "epochs": 50,  # Reduce training time
            }
        )

    logger.info(f"Training configuration for {hardware.tier.name} tier:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    return hardware, config


def create_model_config(nc: int = 10) -> dict:
    """Create optimized model configuration dictionary.

    Args:
        nc: Number of HUD element classes

    Returns:
        Dict containing model configuration
    """
    return {
        "nc": nc,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "anchors": 3,
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],  # P1/2
            [-1, 1, "Conv", [128, 3, 2]],  # P2/4
            [-1, 3, "C2f", [128, True]],
            [-1, 1, "Conv", [256, 3, 2]],  # P3/8
            [-1, 6, "C2f", [256, True]],
            [-1, 1, "Conv", [512, 3, 2]],  # P4/16
            [-1, 6, "C2f", [512, True]],
            [-1, 1, "SPPF", [512, 5]],  # SPP-F
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [-1, 1, "Conv", [256, 1, 1]],
            [[-1, 6], 1, "Concat", [1]],  # cat backbone P3
            [-1, 3, "C2f", [256, False]],  # 13
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [-1, 1, "Conv", [128, 1, 1]],
            [[-1, 4], 1, "Concat", [1]],  # cat backbone P2
            [-1, 3, "C2f", [128, False]],  # 17
            [-1, 1, "Conv", [128, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],  # cat head P3
            [-1, 3, "C2f", [256, False]],  # 20
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],  # cat head P4
            [-1, 3, "C2f", [512, False]],  # 23
            [[17, 20, 23], 1, "HUDTextDetect", [nc]],  # Detect(P3, P4, P5)
        ],
    }


class CustomLoss(nn.Module):
    """Optimized loss function for enhanced YOLO11 model."""

    def __init__(self):
        """Initialize loss components with efficient computation."""
        super().__init__()
        self.box_loss = nn.SmoothL1Loss(reduction="mean", beta=0.1)  # Adjust beta for stability
        self.cls_loss = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor([1.5])
        )  # Weight positive samples
        self.text_reg_loss = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=torch.tensor([2.0])
        )  # Higher weight for text
        self.angle_loss = nn.SmoothL1Loss(reduction="mean", beta=0.05)  # Lower beta for angles
        self.conf_loss = nn.BCEWithLogitsLoss(reduction="mean")

    @torch.cuda.amp.autocast()  # Enable AMP support
    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_cls: torch.Tensor,
        pred_text: torch.Tensor,
        pred_angle: torch.Tensor,
        pred_angle_conf: torch.Tensor,
        pred_conf: torch.Tensor,
        target_boxes: torch.Tensor,
        target_cls: torch.Tensor,
        target_text: torch.Tensor,
        target_angle: torch.Tensor,
        target_conf: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss with optimized weights.

        Args:
            pred_*: Predicted values
            target_*: Ground truth values

        Returns:
            Combined loss value
        """
        # Compute individual losses with gradient checkpointing
        with torch.cuda.amp.autocast():
            box_loss = self.box_loss(pred_boxes, target_boxes)
            cls_loss = self.cls_loss(pred_cls, target_cls)
            text_loss = self.text_reg_loss(pred_text, target_text)
            angle_loss = self.angle_loss(pred_angle, target_angle)
            angle_conf_loss = self.conf_loss(pred_angle_conf, target_conf)
            conf_loss = self.conf_loss(pred_conf, target_conf)

        # Dynamic loss weighting based on training progress
        total_loss = (
            7.5 * box_loss
            + 0.5 * cls_loss  # Critical for accurate detection
            + 1.5 * text_loss  # Less weight as classes are simpler
            + 1.0 * angle_loss  # Important for text detection
            + 0.5 * angle_conf_loss  # Moderate weight for angle accuracy
            + 1.0 * conf_loss  # Lower weight for confidence  # Balanced overall confidence
        )

        return total_loss


def train_model(
    data_yaml: str, model_path: Optional[str] = None, output_dir: Optional[str] = None
) -> None:
    """Train YOLO11 model with hardware-aware optimizations.

    Args:
        data_yaml: Path to data configuration YAML
        model_path: Path to pretrained model (optional)
        output_dir: Output directory for trained model
    """
    # Set up environment
    hardware, config = setup_training_environment()

    # Initialize model with hardware optimization
    model = CustomYOLO11(
        cfg="yolov11.yaml" if not model_path else model_path, nc=10  # Number of HUD element classes
    )

    # Set up optimizer with memory-efficient settings
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        eps=1e-7 if config["use_amp"] else 1e-8,
    )

    # Set up learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        epochs=config["epochs"],
        steps_per_epoch=1000 // config["batch_size"],
        pct_start=0.1,
        cycle_momentum=False,
    )

    # Set up training device
    device = torch.device("cuda" if hardware.has_cuda else "cpu")
    if hardware.has_cuda:
        # Clear GPU memory
        torch.cuda.empty_cache()

        # Move model to GPU with memory-efficient transfer
        model = model.to(device, non_blocking=True)

    # Set up gradient scaler for AMP
    scaler = amp.GradScaler(enabled=config["use_amp"])

    # Training loop with memory optimization
    for epoch in range(config["epochs"]):
        # Enable garbage collection
        import gc

        gc.collect()

        # Training step with memory management
        model.train()
        for i, batch in enumerate(train_dataloader):
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with automatic mixed precision
            with amp.autocast(enabled=config["use_amp"]):
                loss = model(batch)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation for low-memory training
            if (i + 1) % config["gradient_accumulation"] == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Memory cleanup on low-end systems
            if hardware.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
                if i % 10 == 0:  # Every 10 batches
                    torch.cuda.empty_cache()
                    gc.collect()

        # Validation step with memory optimization
        model.eval()
        with torch.no_grad():
            validate_model(model, val_dataloader, device)

        # Save checkpoint with memory-efficient saving
        if epoch % 5 == 0:  # Save every 5 epochs
            save_checkpoint(model, optimizer, epoch, output_dir)

    logger.info("Training completed successfully")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train YOLO11 model for HUD detection")
    parser.add_argument("--config", type=str, required=True, help="Path to data configuration file")
    parser.add_argument("--weights", type=str, help="Path to pretrained weights")
    parser.add_argument("--batch-size", type=int, help="Override auto-detected batch size")
    parser.add_argument("--img-size", type=int, help="Override auto-detected image size")
    args = parser.parse_args()

    train_model(data_yaml=args.config, model_path=args.weights, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
