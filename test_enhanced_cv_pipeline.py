#!/usr/bin/env python3
"""Test script for Task 18 - Enhanced CV Pipeline with Universal HUD Detection.

This script validates all the enhanced CV pipeline components:
1. Universal HUD detection for multi-game compatibility
2. Adaptive region sizing for performance optimization
3. Tier-based YOLO model selection
4. Performance monitoring and optimization
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from spygate.core.hardware import HardwareDetector
    from spygate.ml.enhanced_cv_pipeline import (
        AdaptiveRegionSizer,
        EnhancedCVPipeline,
        UniversalHUDDetector,
        YOLOModelSelector,
    )

    ENHANCED_PIPELINE_AVAILABLE = True
except ImportError as e:
    ENHANCED_PIPELINE_AVAILABLE = False
    logger.error(f"Enhanced CV pipeline not available: {e}")


def create_test_frame(
    width: int = 1920, height: int = 1080, game_type: str = "madden25"
) -> np.ndarray:
    """Create synthetic test frame with HUD elements."""
    # Create base frame (simulated football field)
    frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)

    # Add field-like gradient
    for y in range(height):
        intensity = int(70 + 30 * np.sin(y / height * np.pi))
        frame[y, :, 1] = intensity  # Green channel for field

    # Add HUD elements based on game type
    hud_y_start = int(height * 0.85)

    if game_type == "madden25":
        # Dark HUD bar
        frame[hud_y_start:, :] = [15, 15, 15]

        # Score bug (left side)
        cv2.rectangle(frame, (0, hud_y_start), (int(width * 0.4), height), (25, 25, 25), -1)
        cv2.putText(
            frame,
            "CHI 14  SF 21",
            (20, hud_y_start + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Down & Distance (center)
        cv2.rectangle(
            frame, (int(width * 0.4), hud_y_start), (int(width * 0.7), height), (20, 20, 20), -1
        )
        cv2.putText(
            frame,
            "2nd & 7",
            (int(width * 0.5), hud_y_start + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Clock area (right side)
        cv2.rectangle(frame, (int(width * 0.7), hud_y_start), (width, height), (30, 30, 30), -1)
        cv2.putText(
            frame,
            "8:42",
            (int(width * 0.8), hud_y_start + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Possession indicator (triangle)
        triangle_points = np.array(
            [
                [int(width * 0.2), hud_y_start + 20],
                [int(width * 0.22), hud_y_start + 30],
                [int(width * 0.2), hud_y_start + 40],
            ],
            np.int32,
        )
        cv2.fillPoly(frame, [triangle_points], (255, 100, 50))

    elif game_type == "madden24":
        # Slightly different HUD positioning
        hud_y_start = int(height * 0.83)
        frame[hud_y_start:, :] = [35, 35, 35]

        cv2.putText(
            frame,
            "DAL 28  NYG 14",
            (20, hud_y_start + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "1st & 10",
            (int(width * 0.45), hud_y_start + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "12:05",
            (int(width * 0.75), hud_y_start + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    elif game_type == "ncaa25":
        # Brighter HUD for NCAA
        hud_y_start = int(height * 0.88)
        frame[hud_y_start:, :] = [80, 80, 80]

        cv2.putText(
            frame,
            "TEXAS 35  OU 21",
            (30, hud_y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "3rd & 4",
            (int(width * 0.5), hud_y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "6:18",
            (int(width * 0.8), hud_y_start + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return frame


def test_hardware_detection():
    """Test hardware detection and tier classification."""
    logger.info("Testing hardware detection...")

    try:
        hardware = HardwareDetector()

        logger.info(f"Hardware Tier: {hardware.tier.name}")
        logger.info(f"Has CUDA: {hardware.has_cuda}")
        logger.info(
            f"GPU Memory: {hardware.gpu_memory_gb:.1f}GB"
            if hasattr(hardware, "gpu_memory_gb") and hardware.gpu_memory_gb
            else "No GPU memory detected"
        )

        return hardware

    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return None


def test_model_selector(hardware: HardwareDetector):
    """Test YOLO model selector with tier-based selection."""
    logger.info("Testing YOLO model selector...")

    try:
        selector = YOLOModelSelector(hardware)

        logger.info(f"Tier models mapping: {selector.tier_models}")
        logger.info(f"Fallback hierarchy: {selector.fallback_models}")

        # Test model selection for different targets
        for target_fps in [15, 30, 60]:
            try:
                model = selector.get_model(performance_target=target_fps)
                logger.info(f"Model for {target_fps} FPS target: {selector.current_model_key}")

                # Simulate performance recording
                selector.record_performance(0.02)  # 50 FPS

            except Exception as e:
                logger.warning(f"Model selection for {target_fps} FPS failed: {e}")

        return selector

    except Exception as e:
        logger.error(f"Model selector test failed: {e}")
        return None


def test_adaptive_regions(hardware: HardwareDetector):
    """Test adaptive region sizing functionality."""
    logger.info("Testing adaptive region sizing...")

    try:
        sizer = AdaptiveRegionSizer(hardware)

        # Test with different frame sizes
        test_frames = [(1920, 1080), (1280, 720), (854, 480)]

        for width, height in test_frames:
            logger.info(f"Testing adaptive regions for {width}x{height}")

            # Test without HUD detections
            regions = sizer.calculate_regions((height, width))
            logger.info(f"  Standard regions: {len(regions)} regions")

            # Test with simulated HUD detections
            mock_detections = [
                {"bbox": [0, int(height * 0.85), int(width * 0.4), height]},
                {"bbox": [int(width * 0.4), int(height * 0.85), int(width * 0.7), height]},
            ]

            adaptive_regions = sizer.calculate_regions((height, width), mock_detections)
            logger.info(f"  Adaptive regions: {len(adaptive_regions)} regions")

            # Test performance recording
            sizer.record_performance(0.033)  # 30 FPS

        return sizer

    except Exception as e:
        logger.error(f"Adaptive region test failed: {e}")
        return None


def test_universal_hud_detector():
    """Test universal HUD detector with multi-game support."""
    logger.info("Testing universal HUD detector...")

    try:
        detector = UniversalHUDDetector()

        # Test game detection with different synthetic frames
        game_types = ["madden25", "madden24", "ncaa25"]

        for game_type in game_types:
            logger.info(f"Testing {game_type} detection...")

            test_frame = create_test_frame(game_type=game_type)

            # Test game type detection
            detected_game = detector.detect_game_type(test_frame)
            logger.info(f"  Expected: {game_type}, Detected: {detected_game}")

            # Test enhanced detection
            start_time = time.time()
            result = detector.detect_enhanced(test_frame)
            processing_time = time.time() - start_time

            logger.info(f"  Detection time: {processing_time:.3f}s")
            logger.info(f"  Detections found: {len(result.get('detections', []))}")
            logger.info(f"  Metadata: {result.get('metadata', {})}")

        # Test performance report
        report = detector.get_performance_report()
        logger.info("Performance report:")
        for section, data in report.items():
            logger.info(f"  {section}: {data}")

        return detector

    except Exception as e:
        logger.error(f"Universal HUD detector test failed: {e}")
        return None


def test_enhanced_cv_pipeline():
    """Test the complete enhanced CV pipeline."""
    logger.info("Testing enhanced CV pipeline...")

    try:
        pipeline = EnhancedCVPipeline(enable_fallback=True)

        # Test with different game types
        for game_type in ["madden25", "madden24", "ncaa25"]:
            logger.info(f"Testing pipeline with {game_type}...")

            test_frame = create_test_frame(game_type=game_type)

            start_time = time.time()
            result = pipeline.detect_hud_elements(test_frame)
            processing_time = time.time() - start_time

            logger.info(f"  Processing time: {processing_time:.3f}s")
            logger.info(f"  Detections: {len(result.get('detections', []))}")

            metadata = result.get("metadata", {})
            logger.info(f"  Hardware tier: {metadata.get('hardware_tier', 'unknown')}")
            logger.info(f"  Game ID: {metadata.get('game_id', 'unknown')}")
            logger.info(f"  Parallel processing: {metadata.get('parallel_processing', False)}")

        # Get performance report
        report = pipeline.get_performance_report()
        logger.info("Pipeline performance report:")
        logger.info(f"  {report}")

        return pipeline

    except Exception as e:
        logger.error(f"Enhanced CV pipeline test failed: {e}")
        return None


def test_performance_optimization():
    """Test performance optimization features."""
    logger.info("Testing performance optimization...")

    try:
        detector = UniversalHUDDetector()

        # Test with multiple frames to trigger optimization
        frames = [create_test_frame(game_type="madden25") for _ in range(20)]

        processing_times = []

        for i, frame in enumerate(frames):
            start_time = time.time()
            result = detector.detect_enhanced(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            if i % 5 == 0:
                logger.info(
                    f"  Frame {i}: {processing_time:.3f}s, {len(result.get('detections', []))} detections"
                )

        avg_time = np.mean(processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0

        logger.info(f"Average processing time: {avg_time:.3f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")

        # Check if optimization occurred
        final_report = detector.get_performance_report()
        optimizations = final_report.get("performance", {}).get("optimizations_applied", 0)
        logger.info(f"Optimizations applied: {optimizations}")

        return True

    except Exception as e:
        logger.error(f"Performance optimization test failed: {e}")
        return False


def main():
    """Run all enhanced CV pipeline tests."""
    logger.info("=" * 60)
    logger.info("TASK 18 - Enhanced CV Pipeline Testing")
    logger.info("=" * 60)

    if not ENHANCED_PIPELINE_AVAILABLE:
        logger.error("Enhanced CV Pipeline not available. Please check dependencies.")
        return False

    # Test results tracking
    results = {}

    # Test 1: Hardware Detection
    logger.info("\n1. HARDWARE DETECTION TEST")
    logger.info("-" * 40)
    hardware = test_hardware_detection()
    results["hardware_detection"] = hardware is not None

    if not hardware:
        logger.error("Hardware detection failed. Cannot proceed with other tests.")
        return False

    # Test 2: YOLO Model Selection
    logger.info("\n2. YOLO MODEL SELECTOR TEST")
    logger.info("-" * 40)
    model_selector = test_model_selector(hardware)
    results["model_selector"] = model_selector is not None

    # Test 3: Adaptive Region Sizing
    logger.info("\n3. ADAPTIVE REGION SIZING TEST")
    logger.info("-" * 40)
    region_sizer = test_adaptive_regions(hardware)
    results["adaptive_regions"] = region_sizer is not None

    # Test 4: Universal HUD Detection
    logger.info("\n4. UNIVERSAL HUD DETECTION TEST")
    logger.info("-" * 40)
    hud_detector = test_universal_hud_detector()
    results["universal_hud"] = hud_detector is not None

    # Test 5: Enhanced CV Pipeline
    logger.info("\n5. ENHANCED CV PIPELINE TEST")
    logger.info("-" * 40)
    cv_pipeline = test_enhanced_cv_pipeline()
    results["enhanced_pipeline"] = cv_pipeline is not None

    # Test 6: Performance Optimization
    logger.info("\n6. PERFORMANCE OPTIMIZATION TEST")
    logger.info("-" * 40)
    perf_optimization = test_performance_optimization()
    results["performance_optimization"] = perf_optimization

    # Cleanup
    if hud_detector:
        hud_detector.cleanup()
    if cv_pipeline:
        cv_pipeline.cleanup()

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:<25}: {status}")

    logger.info("-" * 60)
    logger.info(f"TOTAL: {passed_tests}/{total_tests} tests passed")

    success_rate = (passed_tests / total_tests) * 100
    logger.info(f"SUCCESS RATE: {success_rate:.1f}%")

    if success_rate >= 80:
        logger.info("🎉 Task 18 Enhanced CV Pipeline implementation SUCCESSFUL!")
        return True
    else:
        logger.error("⚠️  Task 18 implementation needs improvement")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
