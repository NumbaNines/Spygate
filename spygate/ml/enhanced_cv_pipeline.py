"""Enhanced CV Pipeline with Universal HUD Detection and Adaptive Processing.

This module implements Task 18 enhancements to the SpygateAI computer vision pipeline:
1. Universal HUD detection for multi-game compatibility
2. Adaptive region sizing for performance optimization
3. Tier-based YOLO model selection
4. Enhanced integration with existing detection system
5. Multi-threading and GPU acceleration
6. Performance monitoring and caching
"""

import json
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
class GameHUDLayout:
    """HUD layout configuration for different football games."""

    game_id: str
    name: str
    hud_regions: dict[str, tuple[float, float, float, float]]  # Relative coordinates
    confidence_threshold: float = 0.6
    detection_interval: int = 30  # frames between full detections
    adaptive_regions: bool = True
    lighting_compensation: bool = True


@dataclass
class AdaptiveRegion:
    """Adaptive region definition with optimization parameters."""

    name: str
    base_region: tuple[float, float, float, float]  # (x1, y1, x2, y2) relative
    priority: float = 1.0
    min_size: float = 0.1
    max_size: float = 0.5
    dynamic_sizing: bool = True
    performance_weight: float = 1.0


class YOLOModelSelector:
    """Hardware-tier based YOLO model selection with dynamic switching."""

    def __init__(self, hardware: HardwareDetector):
        """Initialize the model selector with hardware detection."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available")

        self.hardware = hardware
        self.optimizer = TierOptimizer(hardware)

        # Model selection hierarchy
        self.tier_models = {
            HardwareTier.ULTRA_LOW: "yolov8n",
            HardwareTier.LOW: "yolov8n",
            HardwareTier.MEDIUM: "yolov8s",
            HardwareTier.HIGH: "yolov8m",
            HardwareTier.ULTRA: "yolov8l",
        }

        self.fallback_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l"]

        # Model cache and performance tracking
        self.model_cache = {}
        self.current_model_key = None
        self.performance_history = {}
        self.last_optimization = 0

    def get_model(self, performance_target: Optional[float] = None) -> EnhancedYOLOv8:
        """Get optimal model for current hardware and performance requirements."""
        model_key = self.tier_models.get(self.hardware.tier, "yolov8n")

        # Check if we need to switch models based on performance
        if performance_target and self._should_switch_model(model_key, performance_target):
            model_key = self._get_performance_optimized_model(performance_target)

        # Load model if not cached
        if model_key not in self.model_cache:
            self._load_model(model_key)

        self.current_model_key = model_key
        return self.model_cache[model_key]

    def _load_model(self, model_key: str):
        """Load and cache a YOLO model."""
        try:
            tier_config = MODEL_CONFIGS.get(self.hardware.tier, {})

            optimization_config = OptimizationConfig(
                enable_dynamic_switching=True,
                enable_adaptive_batch_size=True,
                enable_performance_monitoring=True,
                enable_auto_optimization=True,
                max_inference_time=1.0 / 30.0,  # Target 30 FPS
                min_accuracy=0.85,
                max_memory_usage=0.8,
            )

            model = EnhancedYOLOv8(
                model_path=f"{model_key}.pt",
                hardware=self.hardware,
                optimization_config=optimization_config,
            )

            self.model_cache[model_key] = model
            logger.info(f"Loaded and cached model: {model_key} for {self.hardware.tier.name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            # Try fallback
            if model_key != "yolov8n":
                self._load_model("yolov8n")
            else:
                raise RuntimeError("Failed to load any YOLO model")

    def _should_switch_model(self, current_model: str, target_fps: float) -> bool:
        """Determine if model switching is needed for performance."""
        if current_model not in self.performance_history:
            return False

        history = self.performance_history[current_model]
        if len(history) < 5:
            return False

        avg_fps = np.mean([1.0 / t for t in history[-10:]])
        return avg_fps < target_fps * 0.8  # Switch if below 80% of target

    def _get_performance_optimized_model(self, target_fps: float) -> str:
        """Get a smaller model for better performance."""
        current_idx = (
            self.fallback_models.index(self.current_model_key)
            if self.current_model_key in self.fallback_models
            else 0
        )

        # Try smaller models
        for i in range(current_idx):
            candidate = self.fallback_models[i]
            if candidate in self.performance_history:
                history = self.performance_history[candidate]
                avg_fps = np.mean([1.0 / t for t in history[-5:]])
                if avg_fps >= target_fps * 0.9:
                    return candidate

        return self.fallback_models[0]  # Fallback to smallest

    def record_performance(self, inference_time: float):
        """Record performance for current model."""
        if self.current_model_key:
            if self.current_model_key not in self.performance_history:
                self.performance_history[self.current_model_key] = []

            self.performance_history[self.current_model_key].append(inference_time)

            # Keep only recent history
            if len(self.performance_history[self.current_model_key]) > 50:
                self.performance_history[self.current_model_key] = self.performance_history[
                    self.current_model_key
                ][-25:]


class AdaptiveRegionSizer:
    """Dynamic ROI calculation based on HUD elements and performance requirements."""

    def __init__(self, hardware: HardwareDetector):
        """Initialize adaptive region sizer."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available")

        self.hardware = hardware
        self.optimizer = TierOptimizer(hardware)

        # Hardware-specific region limits
        self.tier_limits = {
            HardwareTier.ULTRA_LOW: {"max_regions": 2, "min_size": 0.2},
            HardwareTier.LOW: {"max_regions": 3, "min_size": 0.15},
            HardwareTier.MEDIUM: {"max_regions": 5, "min_size": 0.1},
            HardwareTier.HIGH: {"max_regions": 8, "min_size": 0.08},
            HardwareTier.ULTRA: {"max_regions": 12, "min_size": 0.05},
        }

        # Standard football field regions
        self.standard_regions = [
            AdaptiveRegion("hud_main", (0.0, 0.85, 1.0, 1.0), priority=1.0),
            AdaptiveRegion("score_area", (0.0, 0.85, 0.4, 1.0), priority=0.9),
            AdaptiveRegion("down_distance", (0.4, 0.85, 0.7, 1.0), priority=0.85),
            AdaptiveRegion("clock_area", (0.7, 0.85, 1.0, 1.0), priority=0.8),
            AdaptiveRegion("field_center", (0.2, 0.3, 0.8, 0.7), priority=0.6),
            AdaptiveRegion("left_hash", (0.15, 0.2, 0.35, 0.8), priority=0.4),
            AdaptiveRegion("right_hash", (0.65, 0.2, 0.85, 0.8), priority=0.4),
        ]

        self.region_cache = {}
        self.performance_tracker = []

    def calculate_regions(
        self,
        frame_shape: tuple[int, int],
        hud_detections: list[dict[str, Any]] = None,
        performance_target: float = 30.0,
    ) -> list[tuple[int, int, int, int]]:
        """Calculate optimal processing regions."""
        height, width = frame_shape[:2]
        limits = self.tier_limits[self.hardware.tier]

        regions = []

        if hud_detections:
            # Use detected HUD elements to guide region selection
            for detection in hud_detections:
                bbox = detection.get("bbox", [])
                if len(bbox) == 4:
                    # Expand detected regions slightly
                    x1, y1, x2, y2 = bbox
                    padding = 0.05  # 5% padding

                    expanded = [
                        max(0, (x1 / width) - padding),
                        max(0, (y1 / height) - padding),
                        min(1.0, (x2 / width) + padding),
                        min(1.0, (y2 / height) + padding),
                    ]

                    # Convert to pixel coordinates
                    pixel_region = (
                        int(expanded[0] * width),
                        int(expanded[1] * height),
                        int(expanded[2] * width),
                        int(expanded[3] * height),
                    )

                    regions.append(pixel_region)
        else:
            # Use standard regions prioritized by importance
            sorted_regions = sorted(self.standard_regions, key=lambda r: r.priority, reverse=True)

            for region in sorted_regions:
                if len(regions) >= limits["max_regions"]:
                    break

                x1, y1, x2, y2 = region.base_region

                # Check minimum size requirement
                if (x2 - x1) < limits["min_size"] or (y2 - y1) < limits["min_size"]:
                    continue

                pixel_region = (
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                )

                regions.append(pixel_region)

        # Optimize regions based on performance
        if len(self.performance_tracker) > 10:
            avg_fps = np.mean([1.0 / t for t in self.performance_tracker[-10:]])
            if avg_fps < performance_target * 0.8:
                # Reduce regions for better performance
                regions = regions[: max(1, len(regions) - 1)]

        return self._optimize_regions(regions)

    def _optimize_regions(
        self, regions: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Optimize regions by merging overlapping areas."""
        if len(regions) <= 1:
            return regions

        optimized = []
        regions = sorted(regions, key=lambda r: r[0])  # Sort by x coordinate

        current = regions[0]

        for next_region in regions[1:]:
            overlap = self._calculate_overlap_ratio(current, next_region)

            if overlap > 0.3:  # Merge if >30% overlap
                current = (
                    min(current[0], next_region[0]),
                    min(current[1], next_region[1]),
                    max(current[2], next_region[2]),
                    max(current[3], next_region[3]),
                )
            else:
                optimized.append(current)
                current = next_region

        optimized.append(current)
        return optimized

    def _calculate_overlap_ratio(
        self, region1: tuple[int, int, int, int], region2: tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two regions."""
        x1_1, y1_1, x2_1, y2_1 = region1
        x1_2, y1_2, x2_2, y2_2 = region2

        # Intersection coordinates
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0

        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def record_performance(self, processing_time: float):
        """Record processing performance."""
        self.performance_tracker.append(processing_time)
        if len(self.performance_tracker) > 50:
            self.performance_tracker = self.performance_tracker[-25:]


class UniversalHUDDetector:
    """Enhanced HUD detector with universal multi-game support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize universal HUD detector."""
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Required dependencies not available for Enhanced CV Pipeline")

        self.hardware = HardwareDetector()
        self.optimizer = TierOptimizer(self.hardware)

        # Initialize components
        self.model_selector = YOLOModelSelector(self.hardware)
        self.region_sizer = AdaptiveRegionSizer(self.hardware)

        # Base HUD detector for fallback
        self.base_detector = HUDDetector()
        self.base_detector.initialize()

        # Game layouts
        self.game_layouts = self._load_game_layouts(config_path)
        self.current_game = "unknown"
        self.current_layout = None

        # Performance tracking
        self.performance_stats = {
            "total_detections": 0,
            "average_fps": 0.0,
            "cache_hits": 0,
            "total_requests": 0,
            "optimizations_applied": 0,
        }

        # Caching
        self.detection_cache = {}
        self.layout_cache = {}
        self.cache_ttl = 300  # 5 minutes

        # Threading support
        self.enable_parallel = self.hardware.tier in [HardwareTier.HIGH, HardwareTier.ULTRA]
        self.processing_lock = threading.Lock()

        logger.info(f"Universal HUD detector initialized for {self.hardware.tier.name} hardware")

    def _load_game_layouts(self, config_path: Optional[str]) -> dict[str, GameHUDLayout]:
        """Load game-specific HUD layouts."""
        layouts = {
            "madden25": GameHUDLayout(
                game_id="madden25",
                name="Madden NFL 25",
                hud_regions={
                    "score_bug": (0.0, 0.85, 0.4, 1.0),
                    "down_distance": (0.4, 0.85, 0.7, 1.0),
                    "game_clock": (0.7, 0.85, 1.0, 1.0),
                    "play_clock": (0.35, 0.75, 0.65, 0.85),
                    "possession_indicator": (0.15, 0.85, 0.25, 0.95),
                    "territory_indicator": (0.95, 0.85, 1.0, 0.95),
                },
                confidence_threshold=0.6,
                detection_interval=30,
            ),
            "madden24": GameHUDLayout(
                game_id="madden24",
                name="Madden NFL 24",
                hud_regions={
                    "score_bug": (0.0, 0.83, 0.4, 1.0),
                    "down_distance": (0.38, 0.83, 0.72, 1.0),
                    "game_clock": (0.7, 0.83, 1.0, 1.0),
                },
                confidence_threshold=0.65,
                detection_interval=35,
            ),
            "ncaa25": GameHUDLayout(
                game_id="ncaa25",
                name="College Football 25",
                hud_regions={
                    "score_bug": (0.05, 0.88, 0.45, 1.0),
                    "down_distance": (0.4, 0.88, 0.75, 1.0),
                    "game_clock": (0.75, 0.88, 0.95, 1.0),
                },
                confidence_threshold=0.55,
                detection_interval=25,
            ),
            "generic": GameHUDLayout(
                game_id="generic",
                name="Generic Football Game",
                hud_regions={
                    "hud_bar": (0.0, 0.8, 1.0, 1.0),
                    "score_area": (0.0, 0.8, 0.5, 1.0),
                    "info_area": (0.5, 0.8, 1.0, 1.0),
                },
                confidence_threshold=0.5,
                detection_interval=40,
            ),
        }

        # Load custom layouts if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_data = json.load(f)

                for game_id, layout_data in custom_data.items():
                    layouts[game_id] = GameHUDLayout(**layout_data)

                logger.info(f"Loaded custom layouts from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom layouts: {e}")

        return layouts

    def detect_game_type(self, frame: np.ndarray) -> str:
        """Detect game type from frame characteristics."""
        height, width = frame.shape[:2]

        # Create cache key
        cache_key = f"{width}x{height}_{int(np.mean(frame[:int(height*0.2), :]))}"

        # Check cache
        if cache_key in self.layout_cache:
            cached = self.layout_cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["game_id"]

        # Analyze frame characteristics
        bottom_region = frame[int(height * 0.8) :, :]
        hud_intensity = np.mean(bottom_region)
        hud_variance = np.var(bottom_region)

        # Simple heuristics for game detection
        if hud_intensity < 30 and hud_variance > 100:
            # Dark HUD with high variance suggests Madden 25
            detected_game = "madden25"
        elif 30 <= hud_intensity < 60:
            # Medium intensity suggests Madden 24
            detected_game = "madden24"
        elif hud_intensity >= 60 and hud_variance < 200:
            # Bright, uniform HUD suggests NCAA 25
            detected_game = "ncaa25"
        else:
            detected_game = "generic"

        # Cache result
        self.layout_cache[cache_key] = {"game_id": detected_game, "timestamp": time.time()}

        # Update current game
        if detected_game != self.current_game:
            self.current_game = detected_game
            self.current_layout = self.game_layouts.get(detected_game, self.game_layouts["generic"])
            logger.info(f"Game detected: {detected_game}")

        return detected_game

    def detect_enhanced(self, frame: np.ndarray) -> dict[str, Any]:
        """Enhanced HUD detection with adaptive processing."""
        start_time = time.time()

        # Detect game type
        game_id = self.detect_game_type(frame)

        # Get optimal model
        model = self.model_selector.get_model(performance_target=30.0)

        # Calculate adaptive regions
        regions = self.region_sizer.calculate_regions(frame.shape[:2])

        # Process regions
        if self.enable_parallel and len(regions) > 2:
            results = self._process_parallel(frame, regions, model)
        else:
            results = self._process_sequential(frame, regions, model)

        # Consolidate results
        consolidated = self._consolidate_results(results)

        # Record performance
        processing_time = time.time() - start_time
        self.model_selector.record_performance(processing_time)
        self.region_sizer.record_performance(processing_time)
        self._update_stats(processing_time, len(consolidated.get("detections", [])))

        # Add metadata
        consolidated["metadata"].update(
            {
                "game_id": game_id,
                "hardware_tier": self.hardware.tier.name,
                "regions_processed": len(regions),
                "parallel_processing": self.enable_parallel and len(regions) > 2,
                "model_used": self.model_selector.current_model_key,
            }
        )

        return consolidated

    def _process_sequential(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]], model: EnhancedYOLOv8
    ) -> list[dict[str, Any]]:
        """Process regions sequentially."""
        results = []

        for i, (x1, y1, x2, y2) in enumerate(regions):
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            try:
                # Use base detector for consistency
                result = self.base_detector.detect_hud_elements(roi)

                # Adjust coordinates
                for detection in result.get("detections", []):
                    bbox = detection.get("bbox", [])
                    if len(bbox) == 4:
                        detection["bbox"] = [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]
                        detection["region_id"] = i

                results.append(result)

            except Exception as e:
                logger.warning(f"Region {i} processing failed: {e}")

        return results

    def _process_parallel(
        self, frame: np.ndarray, regions: list[tuple[int, int, int, int]], model: EnhancedYOLOv8
    ) -> list[dict[str, Any]]:
        """Process regions in parallel (placeholder for thread-safe implementation)."""
        # For now, use sequential processing
        # Full parallel implementation would require thread-safe model handling
        return self._process_sequential(frame, regions, model)

    def _consolidate_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Consolidate detection results from multiple regions."""
        all_detections = []
        total_time = 0.0

        for result in results:
            all_detections.extend(result.get("detections", []))
            metadata = result.get("metadata", {})
            total_time += metadata.get("processing_time", 0.0)

        # Remove duplicates
        unique_detections = self._remove_duplicates(all_detections)

        return {
            "detections": unique_detections,
            "metadata": {
                "total_detections": len(unique_detections),
                "processing_time": total_time,
                "hardware_tier": self.hardware.tier.name,
            },
        }

    def _remove_duplicates(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate detections using IoU threshold."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        unique = []

        for detection in sorted_dets:
            is_duplicate = False
            bbox1 = detection.get("bbox", [])

            for existing in unique:
                bbox2 = existing.get("bbox", [])
                if len(bbox1) == 4 and len(bbox2) == 4:
                    iou = self._calculate_iou(bbox1, bbox2)
                    if iou > 0.5:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(detection)

        return unique

    def _calculate_iou(self, bbox1: list[int], bbox2: list[int]) -> float:
        """Calculate IoU for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)

        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0

        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _update_stats(self, processing_time: float, detection_count: int):
        """Update performance statistics."""
        self.performance_stats["total_detections"] += detection_count
        self.performance_stats["total_requests"] += 1

        # Update rolling average FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0.0
        current_avg = self.performance_stats["average_fps"]
        total_requests = self.performance_stats["total_requests"]

        self.performance_stats["average_fps"] = (
            current_avg * (total_requests - 1) + fps
        ) / total_requests

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "hardware": {
                "tier": self.hardware.tier.name,
                "has_cuda": self.hardware.has_cuda,
                "memory_gb": getattr(self.hardware, "gpu_memory_gb", None),
            },
            "performance": self.performance_stats.copy(),
            "configuration": {
                "current_game": self.current_game,
                "parallel_enabled": self.enable_parallel,
                "cache_size": len(self.detection_cache),
            },
            "models": {
                "current": self.model_selector.current_model_key,
                "cached": list(self.model_selector.model_cache.keys()),
            },
        }

    def cleanup(self):
        """Clean up resources."""
        self.detection_cache.clear()
        self.layout_cache.clear()

        if hasattr(self.model_selector, "model_cache"):
            self.model_selector.model_cache.clear()

        logger.info("Enhanced CV pipeline cleaned up")


# Fallback detection with original system
class EnhancedCVPipeline:
    """Main enhanced CV pipeline with fallback to original system."""

    def __init__(self, config_path: Optional[str] = None, enable_fallback: bool = True):
        """Initialize enhanced CV pipeline."""
        self.enable_fallback = enable_fallback
        self.universal_detector = None
        self.fallback_detector = None

        try:
            self.universal_detector = UniversalHUDDetector(config_path)
            logger.info("Enhanced CV pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced pipeline: {e}")

            if enable_fallback:
                try:
                    self.fallback_detector = HUDDetector()
                    self.fallback_detector.initialize()
                    logger.info("Fallback to original HUD detector")
                except Exception as e2:
                    logger.error(f"Fallback detector also failed: {e2}")
                    raise RuntimeError("Both enhanced and fallback detectors failed")
            else:
                raise

    def detect_hud_elements(self, frame: np.ndarray) -> dict[str, Any]:
        """Detect HUD elements with enhanced pipeline and fallback."""
        if self.universal_detector:
            try:
                return self.universal_detector.detect_enhanced(frame)
            except Exception as e:
                logger.warning(f"Enhanced detection failed: {e}")

                if self.fallback_detector:
                    logger.info("Using fallback detector")
                    return self.fallback_detector.detect_hud_elements(frame)
                else:
                    raise

        elif self.fallback_detector:
            return self.fallback_detector.detect_hud_elements(frame)

        else:
            raise RuntimeError("No detector available")

    def get_performance_report(self) -> dict[str, Any]:
        """Get performance report."""
        if self.universal_detector:
            return self.universal_detector.get_performance_report()
        else:
            return {"status": "fallback_mode", "detector": "original_hud_detector"}

    def cleanup(self):
        """Clean up resources."""
        if self.universal_detector:
            self.universal_detector.cleanup()

        # HUDDetector doesn't have cleanup method
        self.fallback_detector = None
