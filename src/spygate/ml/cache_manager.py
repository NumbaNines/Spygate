"""
Advanced Caching Layer for Enhanced Game Analyzer
Provides Redis-based caching with intelligent invalidation strategies.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheConfig:
    """Configuration for caching system."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour
        frame_cache_ttl: int = 1800,  # 30 minutes
        ocr_cache_ttl: int = 7200,  # 2 hours
        model_cache_ttl: int = 14400,  # 4 hours
        max_memory_mb: int = 512,
        enable_compression: bool = True,
        enable_fallback: bool = True,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.default_ttl = default_ttl
        self.frame_cache_ttl = frame_cache_ttl
        self.ocr_cache_ttl = ocr_cache_ttl
        self.model_cache_ttl = model_cache_ttl
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.enable_fallback = enable_fallback


class CacheKey:
    """Cache key generation utilities."""

    @staticmethod
    def frame_hash(frame: np.ndarray, region_type: str = None) -> str:
        """Generate hash for frame or frame region."""
        try:
            # Resize frame for consistent hashing
            if frame.size > 0:
                small_frame = cv2.resize(frame, (64, 64))
                frame_bytes = small_frame.tobytes()
            else:
                frame_bytes = b""

            # Create hash
            hasher = hashlib.md5()
            hasher.update(frame_bytes)
            if region_type:
                hasher.update(region_type.encode())

            return hasher.hexdigest()
        except Exception as e:
            logger.debug(f"Error generating frame hash: {e}")
            return f"fallback_{hash(str(frame.shape))}"

    @staticmethod
    def ocr_key(region_hash: str, preprocess_type: str, ocr_engine: str) -> str:
        """Generate OCR cache key."""
        return f"ocr:{region_hash}:{preprocess_type}:{ocr_engine}"

    @staticmethod
    def frame_analysis_key(frame_hash: str, analyzer_version: str = "v1") -> str:
        """Generate frame analysis cache key."""
        return f"frame_analysis:{analyzer_version}:{frame_hash}"

    @staticmethod
    def model_prediction_key(model_name: str, input_hash: str) -> str:
        """Generate model prediction cache key."""
        return f"model:{model_name}:{input_hash}"

    @staticmethod
    def consensus_key(frame_hashes: List[str], consensus_type: str) -> str:
        """Generate consensus cache key."""
        combined_hash = hashlib.md5("".join(sorted(frame_hashes)).encode()).hexdigest()
        return f"consensus:{consensus_type}:{combined_hash}"


class FallbackCache:
    """In-memory fallback cache when Redis is unavailable."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        try:
            # Evict old entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = value
            self.access_times[key] = datetime.now()
            return True
        except Exception as e:
            logger.debug(f"Fallback cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            return True
        return False

    def _evict_lru(self):
        """Evict least recently used items."""
        if not self.access_times:
            return

        # Remove 10% of oldest entries
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        evict_count = max(1, len(sorted_items) // 10)

        for key, _ in sorted_items[:evict_count]:
            self.delete(key)


class AdvancedCacheManager:
    """Advanced caching manager with Redis backend and intelligent strategies."""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self.fallback_cache = FallbackCache()
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0, "fallback_hits": 0}

        self._initialize_redis()

    def _initialize_redis(self) -> bool:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using fallback cache only")
            return False

        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # We handle encoding ourselves
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            self.redis_client.ping()
            logger.info(
                f"✅ Redis cache connected: {self.config.redis_host}:{self.config.redis_port}"
            )
            return True

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using fallback cache")
            self.redis_client = None
            return False

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            if self.config.enable_compression:
                import zlib

                serialized = pickle.dumps(value)
                return zlib.compress(serialized)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.debug(f"Serialization error: {e}")
            return pickle.dumps(str(value))

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            if self.config.enable_compression:
                import zlib

                decompressed = zlib.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.debug(f"Deserialization error: {e}")
            return None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    data = self.redis_client.get(key)
                    if data is not None:
                        self.cache_stats["hits"] += 1
                        return self._deserialize_value(data)
                except Exception as e:
                    logger.debug(f"Redis get error: {e}")
                    self.cache_stats["errors"] += 1

            # Fallback to in-memory cache
            if self.config.enable_fallback:
                result = self.fallback_cache.get(key)
                if result is not None:
                    self.cache_stats["fallback_hits"] += 1
                    return result

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            self.cache_stats["errors"] += 1
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.config.default_ttl
            serialized = self._serialize_value(value)

            # Try Redis first
            if self.redis_client:
                try:
                    self.redis_client.setex(key, ttl, serialized)
                    self.cache_stats["sets"] += 1
                    return True
                except Exception as e:
                    logger.debug(f"Redis set error: {e}")
                    self.cache_stats["errors"] += 1

            # Fallback to in-memory cache
            if self.config.enable_fallback:
                return self.fallback_cache.set(key, value, ttl)

            return False

        except Exception as e:
            logger.debug(f"Cache set error: {e}")
            self.cache_stats["errors"] += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            success = False

            # Delete from Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(key)
                    success = True
                except Exception as e:
                    logger.debug(f"Redis delete error: {e}")

            # Delete from fallback
            if self.config.enable_fallback:
                self.fallback_cache.delete(key)
                success = True

            return success

        except Exception as e:
            logger.debug(f"Cache delete error: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            count = 0

            if self.redis_client:
                try:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        count = self.redis_client.delete(*keys)
                except Exception as e:
                    logger.debug(f"Redis pattern clear error: {e}")

            return count

        except Exception as e:
            logger.debug(f"Cache pattern clear error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache_stats.copy()

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0

        # Add Redis info if available
        if self.redis_client:
            try:
                redis_info = self.redis_client.info("memory")
                stats["redis_memory_used"] = redis_info.get("used_memory", 0)
                stats["redis_memory_peak"] = redis_info.get("used_memory_peak", 0)
                stats["redis_connected"] = True
            except Exception:
                stats["redis_connected"] = False
        else:
            stats["redis_connected"] = False

        # Add fallback cache info
        stats["fallback_cache_size"] = len(self.fallback_cache.cache)

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache system."""
        health = {
            "redis_available": False,
            "redis_responsive": False,
            "fallback_available": True,
            "overall_status": "degraded",
        }

        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                health["redis_available"] = True
                health["redis_responsive"] = True
            except Exception as e:
                logger.debug(f"Redis health check failed: {e}")

        # Determine overall status
        if health["redis_responsive"]:
            health["overall_status"] = "healthy"
        elif health["fallback_available"]:
            health["overall_status"] = "degraded"
        else:
            health["overall_status"] = "unhealthy"

        return health


class GameAnalyzerCache:
    """Specialized cache for game analyzer operations."""

    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache = cache_manager
        self.config = cache_manager.config

    def get_frame_analysis(
        self, frame: np.ndarray, analyzer_version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """Get cached frame analysis result."""
        frame_hash = CacheKey.frame_hash(frame)
        key = CacheKey.frame_analysis_key(frame_hash, analyzer_version)
        return self.cache.get(key)

    def set_frame_analysis(
        self, frame: np.ndarray, result: Dict[str, Any], analyzer_version: str = "v1"
    ) -> bool:
        """Cache frame analysis result."""
        frame_hash = CacheKey.frame_hash(frame)
        key = CacheKey.frame_analysis_key(frame_hash, analyzer_version)
        return self.cache.set(key, result, self.config.frame_cache_ttl)

    def get_ocr_result(
        self, region: np.ndarray, preprocess_type: str, ocr_engine: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached OCR result."""
        region_hash = CacheKey.frame_hash(region, f"ocr_{preprocess_type}")
        key = CacheKey.ocr_key(region_hash, preprocess_type, ocr_engine)
        return self.cache.get(key)

    def set_ocr_result(
        self, region: np.ndarray, preprocess_type: str, ocr_engine: str, result: Dict[str, Any]
    ) -> bool:
        """Cache OCR result."""
        region_hash = CacheKey.frame_hash(region, f"ocr_{preprocess_type}")
        key = CacheKey.ocr_key(region_hash, preprocess_type, ocr_engine)
        return self.cache.set(key, result, self.config.ocr_cache_ttl)

    def get_model_prediction(self, model_name: str, input_data: np.ndarray) -> Optional[Any]:
        """Get cached model prediction."""
        input_hash = CacheKey.frame_hash(input_data, f"model_{model_name}")
        key = CacheKey.model_prediction_key(model_name, input_hash)
        return self.cache.get(key)

    def set_model_prediction(
        self, model_name: str, input_data: np.ndarray, prediction: Any
    ) -> bool:
        """Cache model prediction."""
        input_hash = CacheKey.frame_hash(input_data, f"model_{model_name}")
        key = CacheKey.model_prediction_key(model_name, input_hash)
        return self.cache.set(key, prediction, self.config.model_cache_ttl)

    def get_consensus_result(
        self, frame_hashes: List[str], consensus_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached consensus result."""
        key = CacheKey.consensus_key(frame_hashes, consensus_type)
        return self.cache.get(key)

    def set_consensus_result(
        self, frame_hashes: List[str], consensus_type: str, result: Dict[str, Any]
    ) -> bool:
        """Cache consensus result."""
        key = CacheKey.consensus_key(frame_hashes, consensus_type)
        return self.cache.set(key, result, self.config.frame_cache_ttl)

    def invalidate_frame_cache(self, pattern: str = "frame_analysis:*") -> int:
        """Invalidate frame analysis cache."""
        return self.cache.clear_pattern(pattern)

    def invalidate_ocr_cache(self, pattern: str = "ocr:*") -> int:
        """Invalidate OCR cache."""
        return self.cache.clear_pattern(pattern)

    def invalidate_model_cache(self, model_name: str = None) -> int:
        """Invalidate model prediction cache."""
        pattern = f"model:{model_name}:*" if model_name else "model:*"
        return self.cache.clear_pattern(pattern)


# Global cache instance
_cache_manager = None
_game_analyzer_cache = None


def get_cache_manager(config: CacheConfig = None) -> AdvancedCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = AdvancedCacheManager(config)
    return _cache_manager


def get_game_analyzer_cache(config: CacheConfig = None) -> GameAnalyzerCache:
    """Get global game analyzer cache instance."""
    global _game_analyzer_cache
    if _game_analyzer_cache is None:
        cache_manager = get_cache_manager(config)
        _game_analyzer_cache = GameAnalyzerCache(cache_manager)
    return _game_analyzer_cache


def reset_cache_instances():
    """Reset global cache instances (useful for testing)."""
    global _cache_manager, _game_analyzer_cache
    _cache_manager = None
    _game_analyzer_cache = None
