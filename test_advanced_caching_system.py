"""
Test Advanced Caching System
Tests Redis-based caching with fallback, performance optimization, and cache statistics.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spygate.ml.cache_manager import (
    CacheConfig,
    get_cache_manager,
    get_game_analyzer_cache,
    reset_cache_instances,
)
from spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer


def test_cache_initialization():
    """Test cache system initialization."""
    print("🚀 Testing Cache Initialization...")

    # Reset any existing cache instances
    reset_cache_instances()

    # Test cache manager creation
    cache_config = CacheConfig(
        redis_host="localhost",
        redis_port=6379,
        frame_cache_ttl=1800,
        ocr_cache_ttl=7200,
        enable_compression=True,
        enable_fallback=True,
    )

    cache_manager = get_cache_manager(cache_config)
    print(f"✅ Cache manager created: {type(cache_manager).__name__}")

    # Test health check
    health = cache_manager.health_check()
    print(f"📊 Cache health: {health}")

    # Test game analyzer cache
    game_cache = get_game_analyzer_cache(cache_config)
    print(f"✅ Game analyzer cache created: {type(game_cache).__name__}")

    return cache_manager, game_cache


def test_basic_cache_operations(cache_manager):
    """Test basic cache operations."""
    print("\n🔧 Testing Basic Cache Operations...")

    # Test set/get operations
    test_key = "test_key_123"
    test_value = {"test": "data", "number": 42, "array": [1, 2, 3]}

    # Set value
    success = cache_manager.set(test_key, test_value, ttl=300)
    print(f"✅ Cache set operation: {success}")

    # Get value
    retrieved_value = cache_manager.get(test_key)
    print(f"✅ Cache get operation: {retrieved_value == test_value}")

    # Test cache stats
    stats = cache_manager.get_stats()
    print(
        f"📊 Cache stats: hits={stats['hits']}, misses={stats['misses']}, hit_rate={stats['hit_rate']:.2%}"
    )

    # Test delete
    deleted = cache_manager.delete(test_key)
    print(f"✅ Cache delete operation: {deleted}")

    # Verify deletion
    retrieved_after_delete = cache_manager.get(test_key)
    print(f"✅ Value deleted successfully: {retrieved_after_delete is None}")


def test_frame_analysis_caching():
    """Test frame analysis caching with real analyzer."""
    print("\n🎯 Testing Frame Analysis Caching...")

    try:
        # Initialize analyzer with caching
        analyzer = EnhancedGameAnalyzer()

        # Check if cache is enabled
        if hasattr(analyzer, "cache_enabled") and analyzer.cache_enabled:
            print("✅ Analyzer cache enabled")

            # Get cache stats
            cache_stats = analyzer.get_cache_stats()
            print(f"📊 Initial cache stats: {cache_stats}")
        else:
            print("⚠️ Analyzer cache not enabled, testing fallback mode")

        # Create test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # First analysis (should miss cache)
        print("🔍 First analysis (cache miss expected)...")
        start_time = time.time()
        result1 = analyzer.analyze_frame(test_frame, current_time=1.0)
        first_analysis_time = time.time() - start_time
        print(f"⏱️ First analysis time: {first_analysis_time:.3f}s")

        # Second analysis (should hit cache if enabled)
        print("🔍 Second analysis (cache hit expected)...")
        start_time = time.time()
        result2 = analyzer.analyze_frame(test_frame, current_time=2.0)
        second_analysis_time = time.time() - start_time
        print(f"⏱️ Second analysis time: {second_analysis_time:.3f}s")

        # Compare results
        results_match = (
            result1.down == result2.down
            and result1.distance == result2.distance
            and result1.possession_team == result2.possession_team
        )
        print(f"✅ Results consistency: {results_match}")

        # Performance improvement
        if second_analysis_time > 0:
            speedup = first_analysis_time / second_analysis_time
            print(f"🚀 Performance speedup: {speedup:.1f}x")

        # Final cache stats
        if hasattr(analyzer, "cache_enabled") and analyzer.cache_enabled:
            final_stats = analyzer.get_cache_stats()
            print(f"📊 Final cache stats: {final_stats}")

        return analyzer

    except Exception as e:
        print(f"❌ Frame analysis caching test failed: {e}")
        return None


def test_ocr_result_caching(analyzer):
    """Test OCR result caching."""
    print("\n🔤 Testing OCR Result Caching...")

    if not analyzer or not hasattr(analyzer, "cache_enabled"):
        print("⚠️ Analyzer not available, skipping OCR cache test")
        return

    try:
        # Create test ROI (region of interest)
        test_roi = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)

        # Add some text-like patterns
        cv2.putText(
            test_roi, "3RD & 7", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Test OCR caching through the analyzer's cache system
        if analyzer.cache_enabled and analyzer.advanced_cache:
            # Test direct OCR cache
            test_result = {"down": 3, "distance": 7, "confidence": 0.85, "method": "test_cache"}

            # Set OCR result in cache
            cache_set = analyzer.advanced_cache.set_ocr_result(
                test_roi, "down_distance", "enhanced_multi", test_result
            )
            print(f"✅ OCR cache set: {cache_set}")

            # Get OCR result from cache
            cached_result = analyzer.advanced_cache.get_ocr_result(
                test_roi, "down_distance", "enhanced_multi"
            )
            print(f"✅ OCR cache get: {cached_result is not None}")

            if cached_result:
                print(f"📊 Cached OCR result: {cached_result}")
        else:
            print("⚠️ Advanced cache not available")

    except Exception as e:
        print(f"❌ OCR caching test failed: {e}")


def test_cache_performance_under_load():
    """Test cache performance under load."""
    print("\n⚡ Testing Cache Performance Under Load...")

    try:
        cache_manager = get_cache_manager()

        # Test multiple operations
        num_operations = 100
        start_time = time.time()

        # Set operations
        for i in range(num_operations):
            key = f"load_test_key_{i}"
            value = {"frame_id": i, "data": np.random.random(10).tolist()}
            cache_manager.set(key, value, ttl=300)

        set_time = time.time() - start_time
        print(
            f"⏱️ {num_operations} SET operations: {set_time:.3f}s ({num_operations/set_time:.1f} ops/sec)"
        )

        # Get operations
        start_time = time.time()
        hits = 0
        for i in range(num_operations):
            key = f"load_test_key_{i}"
            result = cache_manager.get(key)
            if result is not None:
                hits += 1

        get_time = time.time() - start_time
        print(
            f"⏱️ {num_operations} GET operations: {get_time:.3f}s ({num_operations/get_time:.1f} ops/sec)"
        )
        print(f"📊 Cache hit rate: {hits}/{num_operations} ({hits/num_operations:.1%})")

        # Final stats
        final_stats = cache_manager.get_stats()
        print(f"📊 Load test stats: {final_stats}")

    except Exception as e:
        print(f"❌ Load test failed: {e}")


def test_cache_invalidation(analyzer):
    """Test cache invalidation functionality."""
    print("\n🗑️ Testing Cache Invalidation...")

    if not analyzer or not hasattr(analyzer, "cache_enabled"):
        print("⚠️ Analyzer not available, skipping invalidation test")
        return

    try:
        if analyzer.cache_enabled:
            # Test frame cache invalidation
            frame_result = analyzer.invalidate_cache("frame")
            print(f"✅ Frame cache invalidation: {frame_result}")

            # Test OCR cache invalidation
            ocr_result = analyzer.invalidate_cache("ocr")
            print(f"✅ OCR cache invalidation: {ocr_result}")

            # Test full cache invalidation
            full_result = analyzer.invalidate_cache("all")
            print(f"✅ Full cache invalidation: {full_result}")
        else:
            print("⚠️ Cache not enabled, skipping invalidation test")

    except Exception as e:
        print(f"❌ Cache invalidation test failed: {e}")


def test_cache_compression():
    """Test cache compression functionality."""
    print("\n🗜️ Testing Cache Compression...")

    try:
        # Test with compression enabled
        compressed_config = CacheConfig(enable_compression=True)
        compressed_cache = get_cache_manager(compressed_config)

        # Test with compression disabled
        uncompressed_config = CacheConfig(enable_compression=False)
        reset_cache_instances()
        uncompressed_cache = get_cache_manager(uncompressed_config)

        # Large test data
        large_data = {
            "large_array": np.random.random(1000).tolist(),
            "text_data": "This is a large text string that should compress well. " * 100,
            "nested_data": {"level1": {"level2": {"level3": list(range(100))}}},
        }

        # Test compressed storage
        start_time = time.time()
        compressed_cache.set("large_data_compressed", large_data, ttl=300)
        compressed_time = time.time() - start_time

        # Test uncompressed storage
        start_time = time.time()
        uncompressed_cache.set("large_data_uncompressed", large_data, ttl=300)
        uncompressed_time = time.time() - start_time

        print(f"⏱️ Compressed storage time: {compressed_time:.3f}s")
        print(f"⏱️ Uncompressed storage time: {uncompressed_time:.3f}s")

        # Test retrieval
        compressed_result = compressed_cache.get("large_data_compressed")
        uncompressed_result = uncompressed_cache.get("large_data_uncompressed")

        print(f"✅ Compressed data integrity: {compressed_result == large_data}")
        print(f"✅ Uncompressed data integrity: {uncompressed_result == large_data}")

    except Exception as e:
        print(f"❌ Compression test failed: {e}")


def main():
    """Run all cache tests."""
    print("🎯 ADVANCED CACHING SYSTEM TEST SUITE")
    print("=" * 50)

    try:
        # Test 1: Cache initialization
        cache_manager, game_cache = test_cache_initialization()

        # Test 2: Basic operations
        test_basic_cache_operations(cache_manager)

        # Test 3: Frame analysis caching
        analyzer = test_frame_analysis_caching()

        # Test 4: OCR result caching
        test_ocr_result_caching(analyzer)

        # Test 5: Performance under load
        test_cache_performance_under_load()

        # Test 6: Cache invalidation
        test_cache_invalidation(analyzer)

        # Test 7: Compression
        test_cache_compression()

        print("\n🎉 ALL CACHE TESTS COMPLETED!")
        print("=" * 50)

        # Final system summary
        if analyzer and hasattr(analyzer, "cache_enabled"):
            final_stats = analyzer.get_cache_stats()
            print(f"📊 FINAL SYSTEM STATS:")
            print(f"   Cache Enabled: {final_stats.get('cache_enabled', False)}")
            if "performance_stats" in final_stats:
                perf = final_stats["performance_stats"]
                print(f"   Hit Rate: {perf.get('hit_rate', 0):.1%}")
                print(f"   Total Hits: {perf.get('hits', 0)}")
                print(f"   Total Misses: {perf.get('misses', 0)}")
            if "cache_health" in final_stats:
                health = final_stats["cache_health"]
                print(f"   Overall Status: {health.get('overall_status', 'unknown')}")

    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
