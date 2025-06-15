#!/usr/bin/env python3
"""
Quick test script to verify error handling fixes work correctly.
"""
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.spygate.core.hardware import HardwareDetector
from src.spygate.core.optimizer import TierOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hardware_detection():
    """Test hardware detection functionality."""
    try:
        detector = HardwareDetector()
        tier = detector.detect_tier()
        logger.info(f"Hardware tier detected: {tier.name}")
        return True
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        return False


def test_optimizer():
    """Test optimizer functionality."""
    try:
        detector = HardwareDetector()
        optimizer = TierOptimizer(detector)
        logger.info("Optimizer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Optimizer test failed: {e}")
        return False


def test_optimizer_fix():
    """Test that the TierOptimizer hardware tier comparison fix works."""
    print("🔧 Testing TierOptimizer Fix...")

    try:
        from spygate.core.hardware import HardwareDetector
        from spygate.core.optimizer import TierOptimizer

        # Create a hardware detector
        hardware = HardwareDetector()
        optimizer = TierOptimizer(hardware)

        # Test the get_model_config method that was failing
        config = optimizer.get_model_config("yolo")

        print(f"✅ TierOptimizer.get_model_config() works!")
        print(f"   • Half precision enabled: {config['half']}")
        print(f"   • Device: {config['device']}")
        print(f"   • Batch size: {config['batch_size']}")

        return True

    except Exception as e:
        print(f"❌ TierOptimizer test failed: {e}")
        traceback.print_exc()
        return False


def test_ocr_corrupted_data():
    """Test that OCR handles corrupted image data gracefully."""
    print("\n🔧 Testing OCR Corrupted Data Handling...")

    try:
        from enhanced_ocr_system import EnhancedOCRSystem

        ocr = EnhancedOCRSystem(debug=False)

        # Test corrupted data cases
        test_cases = [
            ("NaN values", np.full((50, 100, 3), np.nan)),
            ("Infinite values", np.full((50, 100, 3), np.inf)),
            ("Negative values", np.full((50, 100, 3), -255)),
            ("Out of range values", np.full((50, 100, 3), 500)),
            ("Wrong dtype", np.ones((50, 100, 3), dtype=np.float64) * 127.5),
        ]

        all_passed = True

        for case_name, test_image in test_cases:
            try:
                result = ocr.extract_text_from_region(test_image, [5, 5, 45, 95])
                print(f"✅ {case_name}: Handled gracefully")
                if result.error:
                    print(f"   • Graceful error: {result.error}")
                else:
                    print(f"   • Processing successful")
            except Exception as e:
                print(f"❌ {case_name}: Unhandled exception - {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ OCR corrupted data test failed: {e}")
        traceback.print_exc()
        return False


def test_enum_fix():
    """Test that EngineStatus enum usage is correct."""
    print("\n🔧 Testing EngineStatus Enum Fix...")

    try:
        from enhanced_ocr_system import EngineStatus, EnhancedOCRSystem

        ocr = EnhancedOCRSystem(debug=False)

        # Test setting engine status to FAILED
        original_status = ocr.engine_status["easyocr"]
        ocr.engine_status["easyocr"] = EngineStatus.FAILED

        print(f"✅ EngineStatus.FAILED assignment works!")
        print(f"   • Original: {original_status}")
        print(f"   • New: {ocr.engine_status['easyocr']}")

        # Restore original status
        ocr.engine_status["easyocr"] = original_status

        return True

    except Exception as e:
        print(f"❌ EngineStatus enum test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all quick fix tests."""
    print("🚀 QUICK FIX VERIFICATION TESTS")
    print("=" * 50)

    tests = [
        ("Hardware Detection", test_hardware_detection),
        ("Optimizer", test_optimizer),
        ("TierOptimizer Fix", test_optimizer_fix),
        ("OCR Corrupted Data", test_ocr_corrupted_data),
        ("EngineStatus Enum", test_enum_fix),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        if test_func():
            logger.info(f"✓ {test_name} test passed")
            passed += 1
        else:
            logger.error(f"✗ {test_name} test failed")

    logger.info(f"Test results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
