"""Universal HUD detection system for multi-game compatibility with adaptive processing.

This module implements Task 18 enhancements to the CV pipeline, providing:
1. Universal HUD detection across multiple games
2. Adaptive region sizing for optimized processing
3. Tier-based YOLO model selection
4. Enhanced integration with existing detection system
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ..core.hardware import HardwareDetector, HardwareTier
    from ..core.optimizer import TierOptimizer
    from .hud_detector import HUDDetector
    from .yolov8_model import MODEL_CONFIGS, EnhancedYOLOv8, OptimizationConfig

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    HardwareDetector = None
    HardwareTier = None
    TierOptimizer = None
    HUDDetector = None
    EnhancedYOLOv8 = None
    OptimizationConfig = None
    MODEL_CONFIGS = {}

logger = logging.getLogger(__name__)


@dataclass
class HUDLayout:
    """HUD layout configuration for different games."""

    game_id: str
    hud_regions: dict[
        str, tuple[float, float, float, float]
    ]  # Relative coordinates (x1, y1, x2, y2)
    confidence_threshold: float = 0.6
    detection_interval: int = 30  # frames
    adaptive_regions: bool = True


@dataclass
class AdaptiveRegion:
    """Adaptive region definition with optimization parameters."""

    name: str
    base_region: tuple[float, float, float, float]  # (x1, y1, x2, y2) relative
    priority: float = 1.0
    min_size: float = 0.1  # Minimum relative size
    max_size: float = 0.5  # Maximum relative size
    dynamic_sizing: bool = True


class YOLOModelSelector:
    """Hardware-tier based YOLO model selection with optimization."""

    def __init__(self, hardware: HardwareDetector):
        """Initialize the model selector."""
        self.hardware = hardware
        self.models = {}
        self.current_model = None
        self.model_cache = {}
        self.performance_history = {}

        # Model hierarchy by size (smallest to largest)
        self.model_hierarchy = {
            HardwareTier.ULTRA_LOW: "yolov8n",
            HardwareTier.LOW: "yolov8n",
            HardwareTier.MEDIUM: "yolov8s",
            HardwareTier.HIGH: "yolov8m",
            HardwareTier.ULTRA: "yolov8l",
        }

        self.fallback_hierarchy = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

    def get_optimal_model(self, target_fps: Optional[float] = None) -> EnhancedYOLOv8:
        """Get the optimal model for current hardware and performance requirements."""
        tier = self.hardware.tier
        model_key = self.model_hierarchy.get(tier, "yolov8n")

        # Check if we already have this model cached
        if model_key in self.model_cache:
            cached_model = self.model_cache[model_key]

            # Validate performance if target FPS specified
            if target_fps and self._check_performance(model_key, target_fps):
                return cached_model
            elif not target_fps:
                return cached_model

        # Load new model
        try:
            config = MODEL_CONFIGS.get(tier, MODEL_CONFIGS[HardwareTier.LOW])
            optimization_config = OptimizationConfig(
                enable_dynamic_switching=True,
                enable_adaptive_batch_size=True,
                enable_performance_monitoring=True,
                enable_auto_optimization=True,
            )

            model = EnhancedYOLOv8(
                model_path=f"{model_key}.pt",
                hardware=self.hardware,
                optimization_config=optimization_config,
            )

            self.model_cache[model_key] = model
            self.current_model = model_key

            logger.info(f"Loaded optimal model {model_key} for {tier.name} hardware")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return self._get_fallback_model()

    def _get_fallback_model(self) -> EnhancedYOLOv8:
        """Get fallback model if optimal model fails."""
        for model_key in self.fallback_hierarchy:
            try:
                if model_key in self.model_cache:
                    logger.info(f"Using cached fallback model: {model_key}")
                    return self.model_cache[model_key]

                # Try to load fallback
                model = EnhancedYOLOv8(model_path=f"{model_key}.pt", hardware=self.hardware)
                self.model_cache[model_key] = model
                logger.info(f"Loaded fallback model: {model_key}")
                return model

            except Exception as e:
                logger.warning(f"Fallback model {model_key} also failed: {e}")
                continue

        raise RuntimeError("All YOLO models failed to load")

    def _check_performance(self, model_key: str, target_fps: float) -> bool:
        """Check if model meets performance requirements."""
        if model_key not in self.performance_history:
            return True  # No history, assume it's fine

        history = self.performance_history[model_key]
        avg_fps = np.mean([1.0 / time for time in history[-10:]])  # Last 10 measurements

        return avg_fps >= target_fps * 0.8  # 80% of target FPS

    def record_performance(self, model_key: str, inference_time: float):
        """Record performance metrics for a model."""
        if model_key not in self.performance_history:
            self.performance_history[model_key] = []

        self.performance_history[model_key].append(inference_time)

        # Keep only recent history
        if len(self.performance_history[model_key]) > 100:
            self.performance_history[model_key] = self.performance_history[model_key][-50:]

    def update_tier(self, new_tier: HardwareTier):
        """Update hardware tier and recommend model change if needed."""
        if new_tier != self.hardware.tier:
            self.hardware.tier = new_tier
            logger.info(f"Hardware tier updated to {new_tier.name}")

            # Clear current model reference to force reload
            self.current_model = None


class AdaptiveRegionSizer:
    """Dynamic ROI calculation based on detected HUD elements and performance."""

    def __init__(self, hardware: HardwareDetector):
        """Initialize the adaptive region sizer."""
        self.hardware = hardware
        self.optimizer = TierOptimizer(hardware)
        self.region_history = {}
        self.performance_tracker = {}

        # Hardware-tier specific region configurations
        self.tier_configs = {
            HardwareTier.ULTRA_LOW: {"max_regions": 2, "min_region_size": 0.2},
            HardwareTier.LOW: {"max_regions": 3, "min_region_size": 0.15},
            HardwareTier.MEDIUM: {"max_regions": 5, "min_region_size": 0.1},
            HardwareTier.HIGH: {"max_regions": 8, "min_region_size": 0.08},
            HardwareTier.ULTRA: {"max_regions": 12, "min_region_size": 0.05},
        }

        # Standard football field regions for optimization
        self.base_regions = [
            AdaptiveRegion("hud_bar", (0.0, 0.85, 1.0, 1.0), priority=1.0),
            AdaptiveRegion("score_area", (0.0, 0.85, 0.4, 1.0), priority=0.9),
            AdaptiveRegion("down_distance", (0.4, 0.85, 0.7, 1.0), priority=0.8),
            AdaptiveRegion("clock_area", (0.7, 0.85, 1.0, 1.0), priority=0.8),
            AdaptiveRegion("field_center", (0.2, 0.3, 0.8, 0.7), priority=0.6),
            AdaptiveRegion("left_sideline", (0.0, 0.2, 0.3, 0.8), priority=0.4),
            AdaptiveRegion("right_sideline", (0.7, 0.2, 1.0, 0.8), priority=0.4),
        ]

    def calculate_adaptive_regions(
        self,
        frame_shape: tuple[int, int],
        hud_detections: list[dict],
        performance_target: float = 30.0,
    ) -> list[tuple[int, int, int, int]]:
        """Calculate optimal regions based on HUD detections and performance requirements."""
        height, width = frame_shape[:2]
        config = self.tier_configs[self.hardware.tier]

        # Start with base regions
        regions = []

        # Prioritize regions based on HUD detections
        if hud_detections:
            # Adjust regions based on actual HUD element locations
            for detection in hud_detections:
                bbox = detection.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox

                    # Convert to relative coordinates
                    rel_x1, rel_y1 = x1 / width, y1 / height
                    rel_x2, rel_y2 = x2 / width, y2 / height

                    # Expand region slightly for context
                    padding = 0.05
                    expanded_region = (
                        max(0, rel_x1 - padding),
                        max(0, rel_y1 - padding),
                        min(1, rel_x2 + padding),
                        min(1, rel_y2 + padding),
                    )

                    # Convert back to pixel coordinates
                    abs_region = (
                        int(expanded_region[0] * width),
                        int(expanded_region[1] * height),
                        int(expanded_region[2] * width),
                        int(expanded_region[3] * height),
                    )

                    regions.append(abs_region)
        else:
            # Use predefined regions when no HUD detected
            sorted_regions = sorted(self.base_regions, key=lambda r: r.priority, reverse=True)

            for region in sorted_regions[: config["max_regions"]]:
                x1, y1, x2, y2 = region.base_region

                # Ensure minimum size
                region_width = x2 - x1
                region_height = y2 - y1

                if (
                    region_width < config["min_region_size"]
                    or region_height < config["min_region_size"]
                ):
                    continue

                # Convert to absolute coordinates
                abs_region = (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

                regions.append(abs_region)

        # Limit number of regions based on hardware capability
        regions = regions[: config["max_regions"]]

        # Ensure regions don't overlap significantly
        regions = self._merge_overlapping_regions(regions)

        return regions

    def _merge_overlapping_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping regions to optimize processing."""
        if len(regions) <= 1:
            return regions

        merged = []
        regions = sorted(regions, key=lambda r: r[0])  # Sort by x1

        current = regions[0]

        for next_region in regions[1:]:
            # Check for overlap
            if self._calculate_overlap(current, next_region) > 0.3:  # 30% overlap threshold
                # Merge regions
                current = (
                    min(current[0], next_region[0]),
                    min(current[1], next_region[1]),
                    max(current[2], next_region[2]),
                    max(current[3], next_region[3]),
                )
            else:
                merged.append(current)
                current = next_region

        merged.append(current)
        return merged

    def _calculate_overlap(
        self, region1: tuple[int, int, int, int], region2: tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two regions."""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2

        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0

        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        region1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        region2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = region1_area + region2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


class UniversalHUDDetector:
    """Enhanced HUD detector with universal multi-game support and adaptive processing."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the universal HUD detector."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available for UniversalHUDDetector")

        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)
        self.model_selector = YOLOModelSelector(self.hardware)
        self.region_sizer = AdaptiveRegionSizer(self.hardware)

        # Initialize base HUD detector
        self.base_detector = HUDDetector()
        self.base_detector.initialize()

        # Game-specific HUD layouts
        self.hud_layouts = self._load_hud_layouts(config_path)
        self.current_game_id = "unknown"
        self.current_layout = None

        # Adaptive processing state
        self.processing_history = []
        self.performance_target_fps = 30.0
        self.last_optimization_time = 0
        self.optimization_interval = 5.0  # seconds

        # Enhanced caching
        self.layout_cache = {}
        self.region_cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Multi-threading support
        self.thread_pool = None
        self.enable_parallel = self.hardware.tier in [HardwareTier.HIGH, HardwareTier.ULTRA]

        # Performance monitoring
        self.stats = {
            "detections_processed": 0,
            "average_inference_time": 0.0,
            "cache_hit_rate": 0.0,
            "adaptive_optimizations": 0,
        }

        logger.info(f"Universal HUD detector initialized for {self.hardware.tier.name} hardware")

    def _load_hud_layouts(self, config_path: Optional[str]) -> dict[str, HUDLayout]:
        """Load game-specific HUD layout configurations."""
        # Default layouts for common games
        default_layouts = {
            "madden25": HUDLayout(
                game_id="madden25",
                hud_regions={
                    "score_bug": (0.0, 0.85, 0.4, 1.0),
                    "down_distance": (0.4, 0.85, 0.7, 1.0),
                    "game_clock": (0.7, 0.85, 1.0, 1.0),
                    "play_clock": (0.35, 0.75, 0.65, 0.85),
                },
                confidence_threshold=0.6,
                detection_interval=30,
            ),
            "madden24": HUDLayout(
                game_id="madden24",
                hud_regions={
                    "score_bug": (0.0, 0.83, 0.4, 1.0),
                    "down_distance": (0.38, 0.83, 0.72, 1.0),
                    "game_clock": (0.7, 0.83, 1.0, 1.0),
                },
                confidence_threshold=0.65,
                detection_interval=35,
            ),
            "ncaa25": HUDLayout(
                game_id="ncaa25",
                hud_regions={
                    "score_bug": (0.05, 0.88, 0.45, 1.0),
                    "down_distance": (0.4, 0.88, 0.75, 1.0),
                    "game_clock": (0.75, 0.88, 0.95, 1.0),
                },
                confidence_threshold=0.55,
                detection_interval=25,
            ),
            "generic": HUDLayout(
                game_id="generic",
                hud_regions={
                    "hud_bar": (0.0, 0.8, 1.0, 1.0),
                    "score_area": (0.0, 0.8, 0.5, 1.0),
                    "info_area": (0.5, 0.8, 1.0, 1.0),
                },
                confidence_threshold=0.5,
                detection_interval=40,
            ),
        }

        # Load custom layouts if config file provided
        if config_path and Path(config_path).exists():
            try:
                import json

                with open(config_path) as f:
                    custom_layouts = json.load(f)

                for game_id, layout_data in custom_layouts.items():
                    default_layouts[game_id] = HUDLayout(**layout_data)

                logger.info(f"Loaded custom HUD layouts from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom layouts: {e}")

        return default_layouts

    def detect_game_and_layout(self, frame: np.ndarray) -> str:
        """Detect the game type and select appropriate HUD layout."""
        # Simplified game detection based on frame characteristics
        # In a full implementation, this would use more sophisticated methods

        height, width = frame.shape[:2]
        aspect_ratio = width / height

        # Check cache first
        cache_key = f"{width}x{height}_{int(np.mean(frame))}"
        if cache_key in self.layout_cache:
            cached_result = self.layout_cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                return cached_result["game_id"]

        # Basic heuristics for game detection
        if aspect_ratio > 1.7:  # Widescreen format
            # Analyze HUD positioning patterns
            bottom_region = frame[int(height * 0.8) :, :]
            hud_intensity = np.mean(bottom_region)

            if hud_intensity < 50:  # Dark HUD bar typical of Madden 25
                detected_game = "madden25"
            elif 50 <= hud_intensity < 80:  # Medium intensity for Madden 24
                detected_game = "madden24"
            else:  # Brighter HUD for NCAA
                detected_game = "ncaa25"
        else:
            detected_game = "generic"

        # Cache the result
        self.layout_cache[cache_key] = {"game_id": detected_game, "timestamp": time.time()}

        # Update current game and layout
        if detected_game != self.current_game_id:
            self.current_game_id = detected_game
            self.current_layout = self.hud_layouts.get(detected_game, self.hud_layouts["generic"])
            logger.info(f"Detected game: {detected_game}")

        return detected_game

    def detect_hud_elements_enhanced(self, frame: np.ndarray) -> dict[str, Any]:
        """Enhanced HUD detection with universal multi-game support."""
        start_time = time.time()

        # Detect game type and adapt layout
        game_id = self.detect_game_and_layout(frame)

        # Get optimal model for current hardware and performance requirements
        model = self.model_selector.get_optimal_model(self.performance_target_fps)

        # Calculate adaptive regions
        regions = self.region_sizer.calculate_adaptive_regions(
            frame.shape[:2],
            [],  # Initial call without prior detections
            self.performance_target_fps,
        )

        # Process regions in parallel if enabled
        if self.enable_parallel and len(regions) > 1:
            results = self._process_regions_parallel(frame, regions, model)
        else:
            results = self._process_regions_sequential(frame, regions, model)

        # Consolidate results
        consolidated = self._consolidate_detections(results)

        # Record performance
        processing_time = time.time() - start_time
        self.model_selector.record_performance(self.model_selector.current_model, processing_time)
        self._update_performance_stats(processing_time, len(consolidated.get("detections", [])))

        # Adaptive optimization check
        if time.time() - self.last_optimization_time > self.optimization_interval:
            self._perform_adaptive_optimization()
            self.last_optimization_time = time.time()

        # Add enhanced metadata
        consolidated["metadata"].update(
            {
                "game_id": game_id,
                "layout_used": self.current_layout.game_id if self.current_layout else "none",
                "regions_processed": len(regions),
                "parallel_processing": self.enable_parallel,
                "hardware_tier": self.hardware.tier.name,
                "adaptive_optimizations": self.stats["adaptive_optimizations"],
            }
        )

        return consolidated

    def _process_regions_sequential(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]], model: EnhancedYOLOv8
    ) -> list[dict[str, Any]]:
        """Process regions sequentially."""
        results = []

        for i, (x1, y1, x2, y2) in enumerate(regions):
            # Extract ROI
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            try:
                # Use the base detector for consistent processing
                detection_result = self.base_detector.detect_hud_elements(roi)

                # Adjust coordinates back to full frame
                for detection in detection_result.get("detections", []):
                    bbox = detection.get("bbox", [])
                    if len(bbox) == 4:
                        detection["bbox"] = [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]
                        detection["region_id"] = i

                results.append(detection_result)

            except Exception as e:
                logger.warning(f"Failed to process region {i}: {e}")
                continue

        return results

    def _process_regions_parallel(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]], model: EnhancedYOLOv8
    ) -> list[dict[str, Any]]:
        """Process regions in parallel for high-end hardware."""
        # For now, fall back to sequential processing
        # Parallel processing would require thread-safe model handling
        return self._process_regions_sequential(frame, regions, model)

    def _consolidate_detections(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Consolidate detection results from multiple regions."""
        all_detections = []
        total_processing_time = 0.0

        for result in results:
            if "detections" in result:
                all_detections.extend(result["detections"])

            if "metadata" in result and "processing_time" in result["metadata"]:
                total_processing_time += result["metadata"]["processing_time"]

        # Remove duplicate detections
        unique_detections = self._remove_duplicate_detections(all_detections)

        # Create consolidated result
        return {
            "detections": unique_detections,
            "metadata": {
                "total_detections": len(unique_detections),
                "processing_time": total_processing_time,
                "hardware_tier": self.hardware.tier.name,
                "model_version": "UniversalYOLOv8",
            },
        }

    def _remove_duplicate_detections(
        self, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate detections using IoU threshold."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        unique_detections = []

        for detection in sorted_detections:
            is_duplicate = False
            bbox1 = detection.get("bbox", [])

            for existing in unique_detections:
                bbox2 = existing.get("bbox", [])

                if len(bbox1) == 4 and len(bbox2) == 4:
                    iou = self._calculate_iou(bbox1, bbox2)
                    if iou > 0.5:  # IoU threshold for duplicates
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_detections.append(detection)

        return unique_detections

    def _calculate_iou(self, bbox1: list[int], bbox2: list[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0

        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _update_performance_stats(self, processing_time: float, detection_count: int):
        """Update performance statistics."""
        self.stats["detections_processed"] += detection_count

        # Update rolling average of inference time
        self.processing_history.append(processing_time)
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-50:]

        self.stats["average_inference_time"] = np.mean(self.processing_history)

    def _perform_adaptive_optimization(self):
        """Perform adaptive optimization based on performance history."""
        if len(self.processing_history) < 10:
            return

        avg_time = self.stats["average_inference_time"]
        target_time = 1.0 / self.performance_target_fps

        if avg_time > target_time * 1.2:  # 20% over target
            # Performance is poor, apply optimizations
            self._apply_performance_optimizations()
            self.stats["adaptive_optimizations"] += 1

    def _apply_performance_optimizations(self):
        """Apply performance optimizations when needed."""
        # Reduce region count for lower-tier hardware
        if self.hardware.tier in [HardwareTier.LOW, HardwareTier.ULTRA_LOW]:
            # Modify region sizer configuration
            config = self.region_sizer.tier_configs[self.hardware.tier]
            config["max_regions"] = max(1, config["max_regions"] - 1)

        # Consider switching to smaller model if consistently poor performance
        current_fps = 1.0 / self.stats["average_inference_time"]
        if current_fps < self.performance_target_fps * 0.6:  # Less than 60% of target
            # This would trigger the model selector's automatic switching
            logger.info("Triggering model optimization due to poor performance")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "hardware_info": {
                "tier": self.hardware.tier.name,
                "has_cuda": self.hardware.has_cuda,
                "gpu_memory": (
                    f"{self.hardware.total_memory_gb:.1f}GB"
                    if self.hardware.total_memory_gb
                    else "N/A"
                ),
            },
            "performance_stats": self.stats.copy(),
            "current_config": {
                "game_id": self.current_game_id,
                "layout": self.current_layout.game_id if self.current_layout else "none",
                "parallel_processing": self.enable_parallel,
                "target_fps": self.performance_target_fps,
            },
            "model_info": {
                "current_model": self.model_selector.current_model,
                "cached_models": list(self.model_selector.model_cache.keys()),
            },
        }

    def cleanup(self):
        """Clean up resources."""
        if self.thread_pool:
            self.thread_pool.shutdown()

        if self.base_detector:
            # HUDDetector doesn't have cleanup method, but we can clear caches
            pass

        # Clear caches
        self.layout_cache.clear()
        self.region_cache.clear()
        self.model_selector.model_cache.clear()

        logger.info("Universal HUD detector cleaned up")
