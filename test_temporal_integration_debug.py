#!/usr/bin/env python3
"""
Temporal Integration Debug: Burst Sampling vs Temporal Manager
============================================================
Identifies and debugs the confidence voting conflict between:
1. Burst sampling consensus (current_time=None)
2. Temporal manager voting (current_time!=None)

This addresses the core integration issue you discovered.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer
from spygate.ml.enhanced_ocr import EnhancedOCR

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_frame():
    """Load a test frame from the video for debugging."""
    video_path = "1 min 30 test clip.mp4"
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return None

    # Jump to middle of video for a good test frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Could not read frame")
        return None

    logger.info(f"Loaded test frame #{target_frame} shape: {frame.shape}")
    return frame


def extract_down_distance_region(frame, analyzer):
    """Extract down_distance_area region from frame."""
    try:
        # Resize to burst sampling resolution
        frame = cv2.resize(frame, (854, 480))

        # Run YOLO detection
        detections = analyzer.model.detect(frame)

        for detection in detections:
            class_name = detection.get("class", "")
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox", [0, 0, 0, 0])

            if class_name == "down_distance_area" and confidence > 0.7:
                x1, y1, x2, y2 = map(int, bbox)
                region = frame[y1:y2, x1:x2]

                logger.info(
                    f"Found down_distance_area: conf={confidence:.3f}, region={region.shape}"
                )
                return region, bbox, confidence

        logger.warning("No down_distance_area found with confidence > 0.7")
        return None, None, None

    except Exception as e:
        logger.error(f"Error extracting region: {e}")
        return None, None, None


def test_burst_vs_temporal_integration():
    """Test the integration issue between burst sampling and temporal manager."""
    logger.info("=" * 80)
    logger.info("TEMPORAL INTEGRATION DEBUG: Burst Sampling vs Temporal Manager")
    logger.info("=" * 80)

    # Load test frame
    frame = load_test_frame()
    if frame is None:
        return

    # Initialize components
    logger.info("\n--- Initializing Components ---")
    analyzer = EnhancedGameAnalyzer()
    enhanced_ocr = EnhancedOCR()

    # Extract test region
    logger.info("\n--- Extracting Down Distance Region ---")
    region, bbox, yolo_conf = extract_down_distance_region(frame, analyzer)

    if region is None:
        logger.error("Could not extract test region")
        return

    # Save test region for visual inspection
    cv2.imwrite("debug_test_region.png", region)
    logger.info("Saved test region to debug_test_region.png")

    # Test 1: BURST SAMPLING MODE (current_time=None)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: BURST SAMPLING MODE (current_time=None)")
    logger.info("=" * 60)

    burst_results = enhanced_ocr.debug_temporal_integration_issue(region, current_time=None)

    # Test 2: TEMPORAL MANAGER MODE (current_time=123.45)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: TEMPORAL MANAGER MODE (current_time=123.45)")
    logger.info("=" * 60)

    temporal_results = enhanced_ocr.debug_temporal_integration_issue(region, current_time=123.45)

    # Test 3: Integration Comparison
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: INTEGRATION COMPARISON")
    logger.info("=" * 60)

    burst_confidence = burst_results["integration_status"]["ocr_confidence"]
    temporal_confidence = temporal_results["integration_status"]["ocr_confidence"]

    logger.info(f"Burst sampling confidence: {burst_confidence:.3f}")
    logger.info(f"Temporal manager confidence: {temporal_confidence:.3f}")
    logger.info(f"Confidence difference: {abs(burst_confidence - temporal_confidence):.3f}")

    # Check for the specific integration issue
    if abs(burst_confidence - temporal_confidence) > 0.1:
        logger.warning("⚠️  INTEGRATION ISSUE DETECTED!")
        logger.warning("Different confidence scores between modes could cause voting conflicts")
    else:
        logger.info("✅ Confidence scores are consistent between modes")

    # Test 4: Actual Analyzer Integration Test
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: ACTUAL ANALYZER INTEGRATION")
    logger.info("=" * 60)

    # Test burst sampling extraction
    logger.info("--- Burst Sampling Extraction ---")
    region_data = {"roi": region, "bbox": bbox, "confidence": yolo_conf}

    try:
        burst_extraction = analyzer._extract_down_distance_from_region(region_data, None)
        logger.info(f"Burst result: {burst_extraction}")
    except Exception as e:
        logger.error(f"Burst extraction failed: {e}")
        burst_extraction = None

    # Test temporal manager extraction
    logger.info("--- Temporal Manager Extraction ---")
    try:
        temporal_extraction = analyzer._extract_down_distance_from_region(region_data, 123.45)
        logger.info(f"Temporal result: {temporal_extraction}")
    except Exception as e:
        logger.error(f"Temporal extraction failed: {e}")
        temporal_extraction = None

    # Test 5: Integration Fix Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: INTEGRATION FIX RECOMMENDATIONS")
    logger.info("=" * 60)

    if burst_extraction is None and temporal_extraction is None:
        logger.error("❌ CRITICAL: Both modes failed - OCR preprocessing issue")
        logger.info("RECOMMENDATION: Enhance OCR preprocessing for small HUD regions")

    elif burst_extraction is None:
        logger.warning("⚠️  Burst sampling failed, temporal manager worked")
        logger.info("RECOMMENDATION: Lower confidence threshold for burst sampling")

    elif temporal_extraction is None:
        logger.warning("⚠️  Temporal manager failed, burst sampling worked")
        logger.info("RECOMMENDATION: Check temporal manager bypass logic")

    elif burst_extraction != temporal_extraction:
        logger.warning("⚠️  Different results between modes")
        logger.info(f"Burst: {burst_extraction}")
        logger.info(f"Temporal: {temporal_extraction}")
        logger.info("RECOMMENDATION: Ensure consistent OCR preprocessing")

    else:
        logger.info("✅ Both modes returned consistent results")
        logger.info("INTEGRATION STATUS: Working correctly")

    # Test 6: Performance Impact Analysis
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: PERFORMANCE IMPACT ANALYSIS")
    logger.info("=" * 60)

    import time

    # Time burst sampling
    burst_times = []
    for i in range(5):
        start = time.time()
        enhanced_ocr.process_region(region, debug_mode=False)
        burst_times.append(time.time() - start)

    avg_burst_time = sum(burst_times) / len(burst_times)
    logger.info(f"Average burst sampling time: {avg_burst_time:.4f}s")

    # Simulate temporal manager overhead
    temporal_times = []
    for i in range(5):
        start = time.time()
        # Simulate temporal manager checks (would be faster in reality)
        enhanced_ocr.process_region(region, debug_mode=False)
        # Add simulated temporal manager overhead
        time.sleep(0.001)  # 1ms overhead simulation
        temporal_times.append(time.time() - start)

    avg_temporal_time = sum(temporal_times) / len(temporal_times)
    logger.info(f"Average temporal manager time: {avg_temporal_time:.4f}s")
    logger.info(f"Performance difference: {avg_temporal_time - avg_burst_time:.4f}s")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION DEBUG SUMMARY")
    logger.info("=" * 80)

    integration_health = "HEALTHY"
    issues_found = []

    if abs(burst_confidence - temporal_confidence) > 0.1:
        integration_health = "ISSUES_DETECTED"
        issues_found.append("Confidence score inconsistency")

    if burst_extraction != temporal_extraction:
        integration_health = "ISSUES_DETECTED"
        issues_found.append("Different extraction results")

    if burst_extraction is None or temporal_extraction is None:
        integration_health = "CRITICAL_ISSUES"
        issues_found.append("Extraction failures")

    logger.info(f"Integration health: {integration_health}")
    if issues_found:
        logger.info("Issues found:")
        for issue in issues_found:
            logger.info(f"  - {issue}")
    else:
        logger.info("✅ No integration issues detected")

    # Save debug results
    debug_summary = {
        "burst_confidence": burst_confidence,
        "temporal_confidence": temporal_confidence,
        "burst_extraction": burst_extraction,
        "temporal_extraction": temporal_extraction,
        "integration_health": integration_health,
        "issues_found": issues_found,
        "avg_burst_time": avg_burst_time,
        "avg_temporal_time": avg_temporal_time,
    }

    import json

    with open("temporal_integration_debug.json", "w") as f:
        json.dump(debug_summary, f, indent=2, default=str)

    logger.info("Debug results saved to temporal_integration_debug.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_burst_vs_temporal_integration()
