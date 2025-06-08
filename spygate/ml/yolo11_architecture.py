"""YOLO11 model architecture definition for HUD element detection."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from ultralytics.nn.modules import (
    C2f,
    Conv,
    DFL,
    Detect,
    SPPF,
)

from .yolo11_config import YOLO11Config, ModelSize
from ..core.hardware import HardwareTier


def create_backbone_layers(config: YOLO11Config) -> List[List]:
    """Create backbone layers based on model configuration.
    
    Args:
        config: YOLO11 model configuration
        
    Returns:
        List of layer configurations
    """
    # Base channels for different model sizes
    base_channels = {
        ModelSize.NANO: 16,
        ModelSize.SMALL: 32,
        ModelSize.MEDIUM: 48,
        ModelSize.LARGE: 64
    }[config.model_size]
    
    # Apply width multiplier
    ch = int(base_channels * config.backbone.width_multiple)
    
    # Define backbone structure
    backbone = [
        # P1/2
        [-1, 1, "Conv", [ch, 3, 2]],
        
        # P2/4
        [-1, 1, "Conv", [ch * 2, 3, 2]],
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [ch * 2, True]],
        
        # P3/8
        [-1, 1, "Conv", [ch * 4, 3, 2]],
        [-1, int(6 * config.backbone.depth_multiple), "C2f", [ch * 4, True]],
        
        # P4/16
        [-1, 1, "Conv", [ch * 8, 3, 2]],
        [-1, int(6 * config.backbone.depth_multiple), "C2f", [ch * 8, True]],
        
        # P5/32
        [-1, 1, "Conv", [ch * 16, 3, 2]],
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [ch * 16, True]],
        
        # SPP-F for enhanced feature extraction
        [-1, 1, "SPPF", [ch * 16, 5]]
    ]
    
    return backbone


def create_head_layers(config: YOLO11Config, backbone_channels: List[int]) -> List[List]:
    """Create detection head layers based on model configuration.
    
    Args:
        config: YOLO11 model configuration
        backbone_channels: List of channel sizes from backbone layers
        
    Returns:
        List of layer configurations
    """
    # Extract channels from backbone
    c3, c4, c5 = backbone_channels[-3:]  # P3, P4, P5 channels
    
    head = [
        # Upsample P5 and concatenate with P4
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [-1, 1, "Conv", [c4, 1, 1]],
        [[-1, 6], 1, "Concat", [1]],  # cat backbone P4
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [c4, False]],
        
        # Upsample and concatenate with P3
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [-1, 1, "Conv", [c3, 1, 1]],
        [[-1, 4], 1, "Concat", [1]],  # cat backbone P3
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [c3, False]],
        
        # FPN downsampling path
        [-1, 1, "Conv", [c3, 3, 2]],
        [[-1, 12], 1, "Concat", [1]],  # cat head P4
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [c4, False]],
        
        [-1, 1, "Conv", [c4, 3, 2]],
        [[-1, 6], 1, "Concat", [1]],  # cat head P5
        [-1, int(3 * config.backbone.depth_multiple), "C2f", [c5, False]],
        
        # HUD-specific detection head
        [[15, 18, 21], 1, "HUDTextDetect", [config.head.num_classes]]
    ]
    
    return head


def create_model_config(config: YOLO11Config) -> Dict:
    """Create complete model configuration dictionary.
    
    Args:
        config: YOLO11 model configuration
        
    Returns:
        Dict containing model configuration
    """
    # Create backbone and head configurations
    backbone = create_backbone_layers(config)
    head = create_head_layers(config, [64, 128, 256])  # Example channel sizes
    
    return {
        "nc": config.head.num_classes,
        "depth_multiple": config.backbone.depth_multiple,
        "width_multiple": config.backbone.width_multiple,
        "anchors": config.head.anchors,
        "backbone": backbone,
        "head": head
    }


class HUDTextDetect(Detect):
    """Custom detection head for HUD elements and text regions."""
    
    def __init__(self, nc=10, ch=()):
        """Initialize HUD text detection head.
        
        Args:
            nc: Number of classes
            ch: Input channels from backbone
        """
        super().__init__(nc, ch)
        
        # Additional layers for text detection
        self.text_conv = nn.ModuleList(
            Conv(x, x, 3, g=4) for x in ch  # Grouped convolutions for efficiency
        )
        
        # Text angle prediction
        self.angle_pred = nn.ModuleList(
            nn.Sequential(
                Conv(x, x, 3, g=4),
                nn.Conv2d(x, 2, 1),  # 2 channels for sin/cos
                nn.Tanh()
            ) for x in ch
        )
        
        # Text confidence
        self.text_conf = nn.ModuleList(
            nn.Sequential(
                Conv(x, x // 2, 1),
                nn.Conv2d(x // 2, 1, 1),
                nn.Sigmoid()
            ) for x in ch
        )
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Forward pass with text detection.
        
        Args:
            x: List of feature maps from backbone
            
        Returns:
            Tuple of detection outputs
        """
        # Process features
        text_features = [self.text_conv[i](xi) for i, xi in enumerate(x)]
        angles = [self.angle_pred[i](xi) for i, xi in enumerate(x)]
        text_scores = [self.text_conf[i](xi) for i, xi in enumerate(x)]
        
        # Regular object detection
        det_out = super().forward(x)
        
        if self.training:
            return det_out + (text_features, angles, text_scores)
        
        # Post-process for inference
        boxes, cls_scores = det_out[0], det_out[1]
        text_boxes = torch.cat([f.view(f.shape[0], -1) for f in text_features], 1)
        text_angles = torch.cat([a.view(a.shape[0], -1) for a in angles], 1)
        conf_scores = torch.cat([s.view(s.shape[0], -1) for s in text_scores], 1)
        
        return boxes, cls_scores, text_boxes, text_angles, conf_scores 