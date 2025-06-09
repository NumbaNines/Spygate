"""Customized YOLO11 model architecture for HUD element detection and text region detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import DFL, SPPF, C2f, Conv, Detect
from ultralytics.nn.tasks import DetectionModel

from ..core.hardware import HardwareDetector, HardwareTier

logger = logging.getLogger(__name__)

# Update class definitions to include both UI elements and game state elements
UI_CLASSES = {
    # Game State Elements (from PRD)
    0: "score_bug",  # Main score display area
    1: "down_distance",  # Down and yards to go
    2: "game_clock",  # Game time remaining
    3: "play_clock",  # Play clock countdown
    4: "score_home",  # Home team score
    5: "score_away",  # Away team score
    6: "possession",  # Ball possession indicator
    7: "yard_line",  # Current yard line
    8: "timeout_indicator",  # Timeout indicators
    9: "penalty_indicator",  # Penalty notification area
    # UI Interface Elements (user-specified)
    10: "hud",  # Main HUD interface
    11: "gamertag",  # In-game Xbox/PlayStation gamertag
    12: "user_name",  # Community/competitive scene name
    13: "preplay",  # Pre-play interface elements
    14: "playcall",  # Play selection interface
    15: "no_huddle",  # No huddle option indicators
    16: "audible",  # Audible selection interface
}


class LiteFeaturePyramid(nn.Module):
    """Memory-efficient Feature Pyramid for low-end systems."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        reduced_channels = channels // 2

        # Use grouped convolutions for memory efficiency
        self.reduce = Conv(channels, reduced_channels, k=1, g=4)
        self.spatial = Conv(reduced_channels, reduced_channels, k=3, g=4)
        self.expand = Conv(reduced_channels, channels, k=1, g=4)

    def forward(self, x):
        identity = x
        out = self.reduce(x)
        out = self.spatial(out)
        out = self.expand(out)
        return out + identity


class MemoryEfficientDetect(Detect):
    """Memory-efficient detection head for low-end systems."""

    def __init__(self, nc=10, ch=()):
        super().__init__(nc, ch)
        self.hardware = HardwareDetector()

        # Use hardware-aware feature processing
        if self.hardware.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            self.feature_processor = LiteFeaturePyramid
        else:
            self.feature_processor = FeaturePyramidAttention

        # Initialize feature processors
        self.processors = nn.ModuleList([self.feature_processor(x) for x in ch])

    def forward(self, x):
        # Process features with memory-efficient operations
        for i in range(len(x)):
            x[i] = self.processors[i](x[i])

        return super().forward(x)


class FeaturePyramidAttention(nn.Module):
    """Feature Pyramid Attention module for enhanced multi-scale feature learning."""

    def __init__(self, channels: int):
        """Initialize FPA module with memory-efficient design.

        Args:
            channels: Number of input/output channels
        """
        super().__init__()

        # Use grouped convolutions for efficiency
        groups = 4
        self.conv_group = nn.ModuleList(
            [Conv(channels, channels // 4, 1, groups=groups) for _ in range(4)]
        )

        self.out = Conv(channels, channels, 1, groups=groups)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of FPA module with efficient feature extraction."""
        # Multi-scale feature extraction with pooling
        features = []
        for i, conv in enumerate(self.conv_group):
            if i > 0:
                pooled = F.avg_pool2d(x, 2**i)
                feat = conv(pooled)
                feat = F.interpolate(feat, size=x.shape[2:], mode="bilinear", align_corners=False)
            else:
                feat = conv(x)
            features.append(feat)

        # Efficient concatenation and normalization
        out = torch.cat(features, dim=1)
        out = self.out(out)
        return self.norm(out)


class DeformableConv2d(nn.Module):
    """Memory-efficient deformable convolution with grouped operations."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 1):
        """Initialize optimized deformable convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            groups: Number of groups for grouped convolution
        """
        super().__init__()

        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Use grouped convolutions
        self.offset_conv = Conv(
            in_channels, 2 * kernel_size * kernel_size, kernel_size, groups=groups
        )
        self.modulation_conv = Conv(
            in_channels, kernel_size * kernel_size, kernel_size, groups=groups
        )
        self.regular_conv = Conv(
            in_channels, out_channels, kernel_size, padding=padding, groups=groups
        )

        # Add normalization
        self.norm = nn.BatchNorm2d(out_channels)

    @torch.cuda.amp.autocast()  # Enable AMP support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optimized deformable sampling."""
        offset = self.offset_conv(x)
        modulation = torch.sigmoid(self.modulation_conv(x))

        x = self.regular_conv(x)
        x = x * modulation.unsqueeze(1)
        return self.norm(x)


class ConfidenceEstimator(nn.Module):
    """Optimized confidence estimation module."""

    def __init__(self, in_channels: int):
        """Initialize with efficient channel reduction.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()

        # Use efficient channel reduction
        reduction = 8
        self.conv1 = Conv(in_channels, in_channels // reduction, 1)
        self.conv2 = Conv(in_channels // reduction, in_channels // (reduction * 2), 1)
        self.conv3 = Conv(in_channels // (reduction * 2), 1, 1)
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalized confidence estimation."""
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = self.norm(x)
        return torch.sigmoid(x)


class HUDTextDetect(Detect):
    """Optimized detection head for HUD element and text region detection."""

    def __init__(self, nc=10, ch=()):
        """Initialize with memory-efficient design.

        Args:
            nc (int): Number of HUD element classes
            ch (tuple): Input channels from backbone layers
        """
        super().__init__(nc, ch)

        # Use ModuleDict for better organization
        self.fpa = nn.ModuleDict(
            {f"scale_{i}": FeaturePyramidAttention(x) for i, x in enumerate(ch)}
        )

        # Optimize text region detection
        self.text_reg = nn.ModuleList(
            nn.Sequential(
                DeformableConv2d(x, x, groups=4),
                Conv(x, x, 3, groups=4),
                nn.BatchNorm2d(x),
                Conv(x, 1, 1),
                DFL(1) if self.reg_max > 1 else nn.Identity(),
            )
            for x in ch
        )

        # Efficient text angle prediction
        self.text_angle = nn.ModuleList(
            nn.Sequential(
                DeformableConv2d(x, x, groups=4),
                Conv(x, x, 3, groups=4),
                nn.BatchNorm2d(x),
                Conv(x, 2, 1),
                nn.Tanh(),
            )
            for x in ch
        )

        # Memory-efficient confidence estimation
        self.confidence = nn.ModuleList(ConfidenceEstimator(x) for x in ch)

    @torch.cuda.amp.autocast()  # Enable AMP support
    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Optimized forward pass with AMP support."""
        shape = x[0].shape

        # Process features efficiently
        x = [self.fpa[f"scale_{i}"](xi) for i, xi in enumerate(x)]

        for i in range(self.nl):
            x[i] = torch.cat(
                (
                    self.cv2[i](x[i]),
                    self.cv3[i](x[i]),
                    self.text_reg[i](x[i]),
                    self.text_angle[i](x[i]),
                    self.confidence[i](x[i]),
                ),
                1,
            )

        if self.training:
            return x

        # Optimized inference
        bs = shape[0]
        self.anchors = self.anchors.to(x[0].device)

        x_cat = torch.cat([xi.view(bs, self.no + 4, -1) for xi in x], 2)
        box = x_cat[:, : self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4 : self.reg_max * 4 + self.nc]
        text_reg = x_cat[:, -4:-3]
        text_angle = x_cat[:, -3:-2]
        angle_conf = x_cat[:, -2:-1]
        overall_conf = x_cat[:, -1:]

        dbox = self.decode_bboxes(box)

        return (
            dbox,
            cls.sigmoid(),
            text_reg.sigmoid(),
            text_angle * 90,
            angle_conf.sigmoid(),
            overall_conf.sigmoid(),
        )


class CustomYOLO11(DetectionModel):
    """Customized YOLO11 model for HUD element and text detection with advanced GPU memory management."""

    def __init__(self, cfg="yolov11.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.hardware = HardwareDetector()

        # Initialize memory manager integration
        self._setup_memory_management()

        # Configure model based on hardware capabilities
        self._configure_model()

    def _setup_memory_management(self):
        """Set up advanced GPU memory management integration."""
        try:
            # Import here to avoid circular imports
            from ..core.gpu_memory_manager import get_memory_manager

            self.memory_manager = get_memory_manager()

            # Get optimal batch size for this model
            self.optimal_batch_size = self.memory_manager.get_optimal_batch_size()

            logger.info(
                f"GPU Memory Manager integrated. Optimal batch size: {self.optimal_batch_size}"
            )
        except ImportError:
            logger.warning("GPU Memory Manager not available, using basic memory management")
            self.memory_manager = None
            self.optimal_batch_size = self.hardware.get_recommended_settings()["max_batch_size"]

    def _configure_model(self):
        """Configure model based on hardware capabilities."""
        if self.hardware.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
            # Use memory-efficient settings for low-end systems
            self._optimize_for_low_end()
        elif self.hardware.tier == HardwareTier.MEDIUM:
            # Balanced settings for medium-tier systems
            self._optimize_for_medium()
        else:
            # Full features for high-end systems
            self._optimize_for_high_end()

    def _optimize_for_low_end(self):
        """Apply optimizations for low-end systems."""
        # Replace detection head with memory-efficient version
        self.model[-1] = MemoryEfficientDetect(nc=self.model[-1].nc, ch=self.model[-1].ch)

        # Enable gradient checkpointing to save memory
        self.gradient_checkpointing = True

        # Reduce model complexity
        self.model = self._reduce_model_complexity()

    def _optimize_for_medium(self):
        """Apply optimizations for medium-tier systems."""
        # Use balanced settings
        self.model[-1] = MemoryEfficientDetect(nc=self.model[-1].nc, ch=self.model[-1].ch)

        # Enable selective gradient checkpointing
        self.gradient_checkpointing = True

    def _optimize_for_high_end(self):
        """Apply optimizations for high-end systems."""
        # Use full feature set
        pass

    def _reduce_model_complexity(self):
        """Reduce model complexity for low-end systems."""
        reduced_model = nn.ModuleList()

        for module in self.model:
            if isinstance(module, C2f):
                # Reduce number of channels in bottleneck layers
                in_channels = module.c1
                out_channels = module.c2
                reduced_channels = max(in_channels // 2, 32)

                # Create reduced version
                reduced_module = C2f(
                    in_channels,
                    out_channels,
                    n=max(module.n - 1, 1),  # Reduce depth
                    shortcut=module.shortcut,
                    g=max(module.g * 2, 1),  # Increase groups for efficiency
                    e=0.5,  # Reduce expansion ratio
                )
                reduced_model.append(reduced_module)
            else:
                reduced_model.append(module)

        return reduced_model

    def forward(self, x, return_memory_stats=False):
        """Forward pass with advanced memory management."""
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        memory_stats = {}

        # Record initial memory state
        if self.memory_manager and torch.cuda.is_available():
            memory_stats["initial"] = self.memory_manager.get_memory_stats()

        # Start timing
        if start_time:
            start_time.record()

        try:
            # Memory-efficient forward pass based on hardware tier
            if self.hardware.tier in [HardwareTier.ULTRA_LOW, HardwareTier.LOW]:
                # Use memory-efficient forward pass with AMP
                with torch.cuda.amp.autocast(enabled=True):
                    result = super().forward(x)
            else:
                result = super().forward(x)

            # End timing
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                processing_time = 0.0

            # Record batch performance
            if self.memory_manager:
                batch_size = x.shape[0] if hasattr(x, "shape") else 1
                self.memory_manager.record_batch_performance(
                    batch_size=batch_size, processing_time=processing_time, success=True
                )

            # Record final memory state
            if self.memory_manager and torch.cuda.is_available():
                memory_stats["final"] = self.memory_manager.get_memory_stats()
                memory_stats["processing_time"] = processing_time

            if return_memory_stats:
                return result, memory_stats
            else:
                return result

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM during forward pass: {e}")

            # Record failed batch performance
            if self.memory_manager:
                batch_size = x.shape[0] if hasattr(x, "shape") else 1
                self.memory_manager.record_batch_performance(
                    batch_size=batch_size, processing_time=0.0, success=False
                )

            # Trigger emergency cleanup
            if self.memory_manager:
                logger.info("Triggering emergency GPU memory cleanup due to OOM")
                self.memory_manager._trigger_cleanup()

            # Re-raise the error
            raise
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def get_memory_optimized_batch_size(
        self, input_shape: tuple, safety_factor: float = 0.8
    ) -> int:
        """Calculate memory-optimized batch size for given input shape."""
        if not self.memory_manager:
            return self.optimal_batch_size

        # Estimate memory usage per sample
        if torch.cuda.is_available():
            # Create a dummy input to estimate memory usage
            dummy_input = torch.zeros((1,) + input_shape[1:], device="cuda")

            # Measure memory before and after a forward pass
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()

            try:
                with torch.no_grad():
                    _ = self.forward(dummy_input)
                mem_after = torch.cuda.memory_allocated()
                mem_per_sample = mem_after - mem_before
            except Exception:
                # Fallback to heuristic
                mem_per_sample = 100 * 1024 * 1024  # 100MB default
            finally:
                # Cleanup
                del dummy_input
                torch.cuda.empty_cache()
        else:
            # CPU fallback - use smaller batch sizes
            return min(self.optimal_batch_size, 4)

        # Get optimal batch size from memory manager
        return self.memory_manager.get_optimal_batch_size(mem_per_sample * safety_factor)

    def get_memory_buffer(self, size: tuple, dtype=torch.float32):
        """Get a memory buffer from the memory pool."""
        if self.memory_manager:
            return self.memory_manager.get_buffer(size, dtype)
        else:
            return torch.zeros(
                size, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu"
            )

    def return_memory_buffer(self, tensor):
        """Return a memory buffer to the pool."""
        if self.memory_manager:
            self.memory_manager.return_buffer(tensor)

    def _initialize_weights(self):
        """Initialize model weights with improved schemes."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Use Kaiming initialization for better gradient flow
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (FeaturePyramidAttention, DeformableConv2d)):
                # Special initialization for custom modules
                for subm in m.modules():
                    if isinstance(subm, (nn.Conv2d, nn.Linear)):
                        nn.init.kaiming_normal_(subm.weight, mode="fan_out", nonlinearity="relu")
                        if subm.bias is not None:
                            nn.init.constant_(subm.bias, 0)

    def _print_model_info(self):
        """Print detailed model architecture information."""
        n_params = sum(p.numel() for p in self.parameters())
        n_gradients = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("Model Summary:")
        logger.info(f"Total parameters: {n_params:,}")
        logger.info(f"Trainable parameters: {n_gradients:,}")
        logger.info(f"Hardware tier: {self.hardware.tier.name}")
        logger.info(f"Optimal batch size: {self.optimal_batch_size}")
        logger.info("Architecture: Enhanced YOLO11 with:")
        logger.info("- Feature Pyramid Attention")
        logger.info("- Deformable Convolutions")
        logger.info("- Confidence Estimation")
        logger.info("- Hardware-Aware Optimizations")
        logger.info("- Advanced GPU Memory Management")

    def get_model_memory_stats(self) -> dict:
        """Get comprehensive model memory statistics."""
        stats = {
            "hardware_tier": self.hardware.tier.name,
            "optimal_batch_size": self.optimal_batch_size,
            "model_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

        if self.memory_manager:
            stats.update(self.memory_manager.get_memory_stats())

        return stats
