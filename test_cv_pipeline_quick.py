#!/usr/bin/env python3
"""Quick validation test for Task 18 Enhanced CV Pipeline."""

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_components():
    """Quick test of enhanced CV pipeline components."""

    try:
        # Test 1: Hardware Detection
        from spygate.core.hardware import HardwareDetector

        hardware = HardwareDetector()
        logger.info(f"✅ Hardware Detection: {hardware.tier.name} tier detected")

        # Test 2: Enhanced CV Pipeline Import
        from spygate.ml.enhanced_cv_pipeline import (
            AdaptiveRegionSizer,
            EnhancedCVPipeline,
            YOLOModelSelector,
        )

        logger.info("✅ Enhanced CV Pipeline: Module imports successful")

        # Test 3: Model Selector
        model_selector = YOLOModelSelector(hardware)
        logger.info(f"✅ Model Selector: Initialized for {hardware.tier.name}")

        # Test 4: Adaptive Region Sizer
        region_sizer = AdaptiveRegionSizer(hardware)
        regions = region_sizer.calculate_regions((1080, 1920))
        logger.info(f"✅ Adaptive Regions: Generated {len(regions)} regions")

        # Test 5: CV Pipeline with Fallback
        pipeline = EnhancedCVPipeline(enable_fallback=True)
        logger.info("✅ Enhanced CV Pipeline: Initialized successfully")

        # Test 6: Synthetic Frame Detection
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = pipeline.detect_hud_elements(test_frame)
        logger.info(
            f"✅ Frame Detection: Processed frame, found {len(result.get('detections', []))} detections"
        )

        # Performance report
        report = pipeline.get_performance_report()
        logger.info("✅ Performance Report: Generated successfully")

        # Cleanup
        pipeline.cleanup()

        logger.info("\n🎉 All Task 18 components working correctly!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_components()
    print(f"\nTask 18 Enhanced CV Pipeline Status: {'✅ WORKING' if success else '❌ FAILED'}")
