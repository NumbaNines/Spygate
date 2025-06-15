"""
Resource Management System for SpygateAI.

This module provides comprehensive resource management including GPU memory,
database connections, file handles, and thread safety mechanisms.
"""

import gc
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import psutil
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .logging_config import get_logger


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    database_connections: int = 0


class GPUResourceManager:
    """Manages GPU memory and CUDA resources."""
    
    def __init__(self):
        self.logger = get_logger()
        self.device = None
        self.max_memory_mb = 0
        self.memory_warnings_sent = 0
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.max_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            self.logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)} ({self.max_memory_mb:.0f}MB)")
        else:
            self.logger.warning("CUDA not available, GPU resource management disabled")
    
    def get_memory_usage(self) -> tuple[float, float]:
        """Get current GPU memory usage in MB and percentage."""
        if not self.device:
            return 0.0, 0.0
        
        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
            total_used = allocated + cached
            percentage = (total_used / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0
            
            return total_used, percentage
        except Exception as e:
            self.logger.error(f"Failed to get GPU memory usage: {e}")
            return 0.0, 0.0
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if not self.device:
            return
        
        try:
            before_mb, _ = self.get_memory_usage()
            torch.cuda.empty_cache()
            after_mb, _ = self.get_memory_usage()
            freed_mb = before_mb - after_mb
            
            if freed_mb > 0:
                self.logger.info(f"GPU cache cleared: freed {freed_mb:.1f}MB")
            
            self.last_cleanup = datetime.now()
        except Exception as e:
            self.logger.error(f"Failed to clear GPU cache: {e}")
    
    def check_memory_pressure(self) -> bool:
        """Check if GPU memory pressure is high and cleanup if needed."""
        if not self.device:
            return False
        
        used_mb, percentage = self.get_memory_usage()
        
        if percentage > 90:
            self.memory_warnings_sent += 1
            self.logger.warning(f"High GPU memory usage: {percentage:.1f}% ({used_mb:.1f}MB)")
            
            # Automatic cleanup if memory is very high
            if percentage > 95:
                self.clear_cache()
                return True
        
        # Periodic cleanup
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self.clear_cache()
        
        return percentage > 85
    
    @contextmanager
    def memory_guard(self, operation_name: str = "GPU operation"):
        """Context manager that monitors GPU memory during operation."""
        if not self.device:
            yield
            return
        
        before_mb, before_pct = self.get_memory_usage()
        self.logger.debug(f"Starting {operation_name} - GPU memory: {before_pct:.1f}%")
        
        try:
            yield
        finally:
            after_mb, after_pct = self.get_memory_usage()
            delta_mb = after_mb - before_mb
            
            if delta_mb > 100:  # More than 100MB increase
                self.logger.warning(f"{operation_name} increased GPU memory by {delta_mb:.1f}MB")
            
            self.logger.debug(f"Completed {operation_name} - GPU memory: {after_pct:.1f}% (Δ{delta_mb:+.1f}MB)")
            
            # Cleanup if memory is getting high
            self.check_memory_pressure()


class DatabaseConnectionManager:
    """Manages database connections with pooling and automatic cleanup."""
    
    def __init__(self, max_connections: int = 10):
        self.logger = get_logger()
        self.max_connections = max_connections
        self.connections: Dict[int, sqlite3.Connection] = {}
        self.connection_times: Dict[int, datetime] = {}
        self.lock = threading.RLock()
        self.active_connections = 0
        self.total_connections_created = 0
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker to cleanup old connections."""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._cleanup_old_connections()
            except Exception as e:
                self.logger.error(f"Database cleanup worker error: {e}")
    
    def _cleanup_old_connections(self):
        """Close connections that have been idle for too long."""
        cutoff_time = datetime.now() - timedelta(minutes=10)
        connections_to_close = []
        
        with self.lock:
            for thread_id, last_used in self.connection_times.items():
                if last_used < cutoff_time and thread_id in self.connections:
                    connections_to_close.append(thread_id)
        
        for thread_id in connections_to_close:
            self._close_connection(thread_id)
    
    def _close_connection(self, thread_id: int):
        """Close a specific connection."""
        with self.lock:
            if thread_id in self.connections:
                try:
                    self.connections[thread_id].close()
                    del self.connections[thread_id]
                    del self.connection_times[thread_id]
                    self.active_connections -= 1
                    self.logger.debug(f"Closed database connection for thread {thread_id}")
                except Exception as e:
                    self.logger.error(f"Error closing database connection: {e}")
    
    @contextmanager
    def get_connection(self, db_path: str, timeout: float = 30.0):
        """Get a database connection with automatic management."""
        thread_id = threading.get_ident()
        connection = None
        
        try:
            with self.lock:
                # Reuse existing connection for this thread
                if thread_id in self.connections:
                    connection = self.connections[thread_id]
                    self.connection_times[thread_id] = datetime.now()
                else:
                    # Create new connection if under limit
                    if self.active_connections >= self.max_connections:
                        # Clean up old connections first
                        self._cleanup_old_connections()
                        
                        if self.active_connections >= self.max_connections:
                            raise RuntimeError(f"Maximum database connections ({self.max_connections}) exceeded")
                    
                    # Create new connection
                    connection = sqlite3.connect(
                        db_path, 
                        timeout=timeout,
                        check_same_thread=False
                    )
                    connection.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                    connection.execute("PRAGMA synchronous=NORMAL")  # Balanced safety/performance
                    connection.execute("PRAGMA cache_size=10000")  # 10MB cache
                    connection.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
                    
                    self.connections[thread_id] = connection
                    self.connection_times[thread_id] = datetime.now()
                    self.active_connections += 1
                    self.total_connections_created += 1
                    
                    self.logger.debug(f"Created database connection for thread {thread_id} ({self.active_connections} active)")
            
            yield connection
            
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            # Close problematic connection
            if thread_id in self.connections:
                self._close_connection(thread_id)
            raise
        finally:
            # Update last used time
            with self.lock:
                if thread_id in self.connection_times:
                    self.connection_times[thread_id] = datetime.now()
    
    def close_all(self):
        """Close all database connections."""
        with self.lock:
            thread_ids = list(self.connections.keys())
            for thread_id in thread_ids:
                self._close_connection(thread_id)
            
            self.logger.info(f"Closed all database connections ({len(thread_ids)} total)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        with self.lock:
            return {
                "active_connections": self.active_connections,
                "total_created": self.total_connections_created,
                "max_connections": self.max_connections,
                "connection_threads": list(self.connections.keys())
            }


class MemoryManager:
    """Manages system memory and garbage collection."""
    
    def __init__(self):
        self.logger = get_logger()
        self.last_gc = datetime.now()
        self.gc_interval = timedelta(minutes=2)
        self.memory_threshold_mb = 1024  # 1GB threshold for warnings
        
    def get_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage in MB and percentage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_mb / (system_memory.total / (1024 * 1024))) * 100
            
            return memory_mb, memory_percent
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0, 0.0
    
    def force_garbage_collection(self) -> int:
        """Force garbage collection and return number of objects collected."""
        try:
            before_objects = len(gc.get_objects())
            collected = gc.collect()
            after_objects = len(gc.get_objects())
            
            self.logger.debug(f"Garbage collection: {collected} objects collected, {before_objects - after_objects} objects freed")
            self.last_gc = datetime.now()
            
            return collected
        except Exception as e:
            self.logger.error(f"Garbage collection failed: {e}")
            return 0
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high and cleanup if needed."""
        memory_mb, memory_percent = self.get_memory_usage()
        
        if memory_mb > self.memory_threshold_mb:
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
            
            # Force garbage collection if memory is very high
            if memory_mb > self.memory_threshold_mb * 1.5:
                self.force_garbage_collection()
                return True
        
        # Periodic garbage collection
        if datetime.now() - self.last_gc > self.gc_interval:
            self.force_garbage_collection()
        
        return memory_mb > self.memory_threshold_mb
    
    @contextmanager
    def memory_guard(self, operation_name: str = "Memory operation"):
        """Context manager that monitors memory during operation."""
        before_mb, before_pct = self.get_memory_usage()
        self.logger.debug(f"Starting {operation_name} - Memory: {before_mb:.1f}MB ({before_pct:.1f}%)")
        
        try:
            yield
        finally:
            after_mb, after_pct = self.get_memory_usage()
            delta_mb = after_mb - before_mb
            
            if delta_mb > 100:  # More than 100MB increase
                self.logger.warning(f"{operation_name} increased memory by {delta_mb:.1f}MB")
            
            self.logger.debug(f"Completed {operation_name} - Memory: {after_mb:.1f}MB ({after_pct:.1f}%) (Δ{delta_mb:+.1f}MB)")
            
            # Cleanup if memory is getting high
            self.check_memory_pressure()


class ThreadManager:
    """Manages thread safety and synchronization."""
    
    def __init__(self):
        self.logger = get_logger()
        self.active_threads: Dict[int, threading.Thread] = {}
        self.thread_locks: Dict[str, threading.RLock] = {}
        self.lock = threading.RLock()
        
    def get_lock(self, resource_name: str) -> threading.RLock:
        """Get a named lock for resource synchronization."""
        with self.lock:
            if resource_name not in self.thread_locks:
                self.thread_locks[resource_name] = threading.RLock()
            return self.thread_locks[resource_name]
    
    @contextmanager
    def synchronized(self, resource_name: str):
        """Context manager for synchronized access to named resource."""
        lock = self.get_lock(resource_name)
        acquired = lock.acquire(timeout=30.0)
        
        if not acquired:
            raise TimeoutError(f"Failed to acquire lock for {resource_name} within 30 seconds")
        
        try:
            self.logger.debug(f"Acquired lock for {resource_name}")
            yield
        finally:
            lock.release()
            self.logger.debug(f"Released lock for {resource_name}")
    
    def register_thread(self, thread: threading.Thread):
        """Register a thread for monitoring."""
        with self.lock:
            self.active_threads[thread.ident] = thread
            self.logger.debug(f"Registered thread {thread.name} ({thread.ident})")
    
    def unregister_thread(self, thread_id: int):
        """Unregister a thread."""
        with self.lock:
            if thread_id in self.active_threads:
                thread = self.active_threads.pop(thread_id)
                self.logger.debug(f"Unregistered thread {thread.name} ({thread_id})")
    
    def get_active_threads(self) -> List[threading.Thread]:
        """Get list of active registered threads."""
        with self.lock:
            return [t for t in self.active_threads.values() if t.is_alive()]
    
    def cleanup_dead_threads(self):
        """Remove dead threads from registry."""
        with self.lock:
            dead_threads = [tid for tid, thread in self.active_threads.items() if not thread.is_alive()]
            for tid in dead_threads:
                del self.active_threads[tid]
            
            if dead_threads:
                self.logger.debug(f"Cleaned up {len(dead_threads)} dead threads")


class ResourceManager:
    """Main resource manager coordinating all resource management subsystems."""
    
    def __init__(self):
        self.logger = get_logger()
        self.gpu_manager = GPUResourceManager()
        self.db_manager = DatabaseConnectionManager()
        self.memory_manager = MemoryManager()
        self.thread_manager = ThreadManager()
        
        self.start_time = datetime.now()
        self.usage_history: List[ResourceUsage] = []
        self.max_history_size = 1000
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Resource manager initialized")
    
    def _monitoring_worker(self):
        """Background worker for resource monitoring."""
        while True:
            try:
                time.sleep(60)  # Monitor every minute
                self._collect_usage_stats()
                self._check_resource_health()
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    def _collect_usage_stats(self):
        """Collect current resource usage statistics."""
        try:
            # CPU and system memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # GPU memory
            gpu_mb, gpu_percent = self.gpu_manager.get_memory_usage()
            
            # Thread count
            active_threads = len(self.thread_manager.get_active_threads())
            
            # Open files
            try:
                process = psutil.Process()
                open_files = len(process.open_files())
            except:
                open_files = 0
            
            # Database connections
            db_connections = self.db_manager.get_stats()["active_connections"]
            
            usage = ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                gpu_memory_mb=gpu_mb,
                gpu_memory_percent=gpu_percent,
                active_threads=active_threads,
                open_files=open_files,
                database_connections=db_connections
            )
            
            self.usage_history.append(usage)
            
            # Trim history if too large
            if len(self.usage_history) > self.max_history_size:
                self.usage_history = self.usage_history[-self.max_history_size:]
            
            # Log performance metrics
            self.logger.log_system_stats(cpu_percent, memory_mb, memory.total / (1024 * 1024))
            if gpu_mb > 0:
                self.logger.log_gpu_memory(gpu_mb, self.gpu_manager.max_memory_mb, "monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to collect usage stats: {e}")
    
    def _check_resource_health(self):
        """Check overall resource health and perform cleanup if needed."""
        # Check GPU memory pressure
        if self.gpu_manager.check_memory_pressure():
            self.logger.warning("GPU memory pressure detected, performing cleanup")
        
        # Check system memory pressure
        if self.memory_manager.check_memory_pressure():
            self.logger.warning("System memory pressure detected, performing cleanup")
        
        # Clean up dead threads
        self.thread_manager.cleanup_dead_threads()
    
    @contextmanager
    def resource_guard(self, operation_name: str = "Resource operation"):
        """Context manager that monitors all resources during operation."""
        with self.gpu_manager.memory_guard(operation_name):
            with self.memory_manager.memory_guard(operation_name):
                self.logger.debug(f"Starting resource-guarded operation: {operation_name}")
                try:
                    yield {
                        'gpu': self.gpu_manager,
                        'memory': self.memory_manager,
                        'threads': self.thread_manager,
                        'database': self.db_manager
                    }
                finally:
                    self.logger.debug(f"Completed resource-guarded operation: {operation_name}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        if not self.usage_history:
            return {"error": "No usage data collected yet"}
        
        latest = self.usage_history[-1]
        uptime = datetime.now() - self.start_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "current": {
                "cpu_percent": latest.cpu_percent,
                "memory_mb": latest.memory_mb,
                "memory_percent": latest.memory_percent,
                "gpu_memory_mb": latest.gpu_memory_mb,
                "gpu_memory_percent": latest.gpu_memory_percent,
                "active_threads": latest.active_threads,
                "open_files": latest.open_files,
                "database_connections": latest.database_connections
            },
            "peak": {
                "cpu_percent": max(u.cpu_percent for u in self.usage_history),
                "memory_mb": max(u.memory_mb for u in self.usage_history),
                "gpu_memory_mb": max(u.gpu_memory_mb for u in self.usage_history),
                "active_threads": max(u.active_threads for u in self.usage_history)
            },
            "averages": {
                "cpu_percent": sum(u.cpu_percent for u in self.usage_history) / len(self.usage_history),
                "memory_mb": sum(u.memory_mb for u in self.usage_history) / len(self.usage_history),
                "gpu_memory_mb": sum(u.gpu_memory_mb for u in self.usage_history) / len(self.usage_history)
            },
            "managers": {
                "gpu": "available" if self.gpu_manager.device else "unavailable",
                "database": self.db_manager.get_stats(),
                "thread_locks": len(self.thread_manager.thread_locks)
            }
        }
    
    def save_resource_report(self, filepath: Optional[str] = None):
        """Save comprehensive resource usage report."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"resource_report_{timestamp}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_resource_summary(),
            "usage_history": [
                {
                    "timestamp": u.timestamp.isoformat(),
                    "cpu_percent": u.cpu_percent,
                    "memory_mb": u.memory_mb,
                    "memory_percent": u.memory_percent,
                    "gpu_memory_mb": u.gpu_memory_mb,
                    "gpu_memory_percent": u.gpu_memory_percent,
                    "active_threads": u.active_threads,
                    "open_files": u.open_files,
                    "database_connections": u.database_connections
                }
                for u in self.usage_history[-100:]  # Last 100 entries
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Resource report saved to {filepath}")
    
    def cleanup_all(self):
        """Perform comprehensive cleanup of all resources."""
        self.logger.info("Starting comprehensive resource cleanup")
        
        # GPU cleanup
        if self.gpu_manager.device:
            self.gpu_manager.clear_cache()
        
        # Memory cleanup
        collected = self.memory_manager.force_garbage_collection()
        
        # Database cleanup
        self.db_manager.close_all()
        
        # Thread cleanup
        self.thread_manager.cleanup_dead_threads()
        
        self.logger.info(f"Resource cleanup completed: {collected} objects collected")


# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get or create the global resource manager instance."""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
    return _global_resource_manager


# Convenience functions
def gpu_memory_guard(operation_name: str = "GPU operation"):
    """Convenience function for GPU memory guarding."""
    return get_resource_manager().gpu_manager.memory_guard(operation_name)


def memory_guard(operation_name: str = "Memory operation"):
    """Convenience function for memory guarding."""
    return get_resource_manager().memory_manager.memory_guard(operation_name)


def synchronized(resource_name: str):
    """Convenience function for synchronized access."""
    return get_resource_manager().thread_manager.synchronized(resource_name)


def get_database_connection(db_path: str, timeout: float = 30.0):
    """Convenience function for database connections."""
    return get_resource_manager().db_manager.get_connection(db_path, timeout)


if __name__ == "__main__":
    # Test the resource management system
    manager = get_resource_manager()
    logger = get_logger()
    
    logger.info("Testing resource management system")
    
    # Test GPU memory guard
    with gpu_memory_guard("test_operation"):
        logger.info("Inside GPU memory guard")
    
    # Test memory guard
    with memory_guard("test_memory_operation"):
        # Simulate memory usage
        data = [i for i in range(10000)]
        logger.info(f"Created test data with {len(data)} items")
    
    # Test synchronized access
    with synchronized("test_resource"):
        logger.info("Inside synchronized block")
    
    # Test database connection
    try:
        with get_database_connection(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (1)")
            conn.commit()
            logger.info("Database test completed")
    except Exception as e:
        logger.error(f"Database test failed: {e}")
    
    # Print resource summary
    summary = manager.get_resource_summary()
    logger.info(f"Resource summary: {summary}")
    
    # Save report
    manager.save_resource_report("test_resource_report.json")
    
    # Cleanup
    manager.cleanup_all()