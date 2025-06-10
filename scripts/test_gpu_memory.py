#!/usr/bin/env python3
"""
GPU Memory Management Test Runner for SpygateAI

This script provides a convenient way to run GPU memory management tests
with various options and detailed reporting.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_gpu_memory_management import GPUMemoryTestSuite


def save_results_to_file(results: dict, output_file: str):
    """Save test results to a JSON file."""
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save results to {output_file}: {e}")


def print_detailed_results(results: dict):
    """Print detailed test results in a formatted way."""
    print("\n" + "=" * 80)
    print("🧠 SPYGATE AI - GPU MEMORY MANAGEMENT TEST RESULTS")
    print("=" * 80)

    # System Information
    print("\n📊 SYSTEM INFORMATION:")
    print(f"   Hardware Tier: {results['system_info']['hardware_tier']}")
    print(f"   CPU Cores: {results['system_info']['cpu_count']}")
    print(f"   Total Memory: {results['system_info']['total_memory_gb']:.1f} GB")
    print(f"   GPU Available: {'✅ Yes' if results['gpu_available'] else '❌ No'}")

    # Overall Summary
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Tests Run: {results['total_tests']}")
    print(f"   Passed: {results['passed_tests']} ✅")
    print(f"   Failed: {results['failed_tests']} ❌")
    print(f"   Success Rate: {results['success_rate']:.1f}%")

    # Individual Test Results
    print(f"\n🔍 DETAILED TEST RESULTS:")
    for i, test_result in enumerate(results["test_results"], 1):
        status_icon = "✅" if test_result["passed"] else "❌"
        print(f"\n{i}. {status_icon} {test_result['test_name'].upper()}")

        if test_result["passed"]:
            # Print test-specific metrics
            if "initialization_time" in test_result:
                print(f"   ⏱️  Initialization Time: {test_result['initialization_time']:.3f}s")
            if "gpu_enabled" in test_result:
                print(f"   🖥️  GPU Enabled: {test_result['gpu_enabled']}")
            if "memory_usage" in test_result:
                mem = test_result["memory_usage"]
                print(f"   💾 Memory Usage: {mem.get('difference_mb', 0):.1f} MB")
            if "analysis_count" in test_result:
                print(f"   🔄 Analyses Performed: {test_result['analysis_count']}")
            if "memory_leak_detected" in test_result:
                leak_status = "⚠️ Detected" if test_result["memory_leak_detected"] else "✅ None"
                print(f"   🔍 Memory Leaks: {leak_status}")
            if "speedup_factor" in test_result:
                if test_result["speedup_factor"] > 0:
                    print(f"   🚀 GPU Speedup: {test_result['speedup_factor']:.2f}x")
            if "concurrent_analyses" in test_result:
                print(f"   ⚡ Concurrent Analyses: {test_result['concurrent_analyses']}")
        else:
            # Print errors
            print(f"   ❌ Errors: {test_result.get('errors', ['Unknown error'])}")

    print("\n" + "=" * 80)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="SpygateAI GPU Memory Management Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_gpu_memory.py                    # Run all tests
  python scripts/test_gpu_memory.py --quick            # Run quick tests only
  python scripts/test_gpu_memory.py --save results.json # Save results to file
  python scripts/test_gpu_memory.py --verbose          # Verbose output
        """,
    )

    parser.add_argument("--save", "-s", help="Save detailed results to JSON file", metavar="FILE")

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick tests only (skip performance and concurrent tests)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )

    parser.add_argument(
        "--no-gpu", action="store_true", help="Skip GPU-specific tests (useful for CI/CD)"
    )

    args = parser.parse_args()

    # Setup logging
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🧠 SpygateAI GPU Memory Management Test Suite")
    print("=" * 50)

    # Create test suite
    test_suite = GPUMemoryTestSuite()

    # Check GPU availability if not skipping GPU tests
    if not args.no_gpu:
        if not test_suite._check_gpu_availability():
            print("⚠️  No GPU detected. Some tests will be skipped.")
            print("   Use --no-gpu flag to skip all GPU-specific tests.\n")

    # Run tests
    start_time = time.time()

    if args.quick:
        print("🏃 Running quick tests only...")
        # Run essential tests only
        essential_tests = [
            test_suite.test_formation_analyzer_gpu_initialization,
            test_suite.test_gpu_error_handling,
            test_suite.test_memory_cleanup,
        ]

        all_results = []
        passed_tests = 0

        for test_method in essential_tests:
            try:
                result = test_method()
                all_results.append(result)
                if result.get("passed", False):
                    passed_tests += 1
                    print(f"✅ {result['test_name']}")
                else:
                    print(f"❌ {result['test_name']}: {result.get('errors', [])}")
            except Exception as e:
                print(f"❌ {test_method.__name__}: {e}")
                all_results.append(
                    {"test_name": test_method.__name__, "passed": False, "errors": [str(e)]}
                )

        # Create summary for quick tests
        results = {
            "total_tests": len(essential_tests),
            "passed_tests": passed_tests,
            "failed_tests": len(essential_tests) - passed_tests,
            "success_rate": (passed_tests / len(essential_tests)) * 100,
            "gpu_available": test_suite._check_gpu_availability(),
            "test_results": all_results,
            "system_info": {
                "hardware_tier": test_suite.hardware_detector.tier.name,
                "cpu_count": os.cpu_count(),
                "total_memory_gb": 0,  # Will be filled by test suite if needed
            },
            "test_mode": "quick",
        }
    else:
        print("🔄 Running comprehensive tests...")
        results = test_suite.run_all_tests()
        results["test_mode"] = "comprehensive"

    execution_time = time.time() - start_time
    results["execution_time"] = execution_time

    # Print results
    print_detailed_results(results)
    print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")

    # Save results if requested
    if args.save:
        save_results_to_file(results, args.save)

    # Exit with appropriate code
    if results["failed_tests"] == 0:
        print("\n🎉 All tests passed! GPU memory management is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {results['failed_tests']} test(s) failed. Check the details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
