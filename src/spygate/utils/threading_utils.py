"""
Advanced Threading and Concurrency Utilities for SpygateAI.

This module provides thread-safe operations, synchronization primitives,
and managed threading for complex video processing and GUI operations.
"""

import queue
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum
import functools

try:
    from PyQt6.QtCore import QThread, QObject, pyqtSignal, QTimer
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer
        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False

from .logging_config import get_logger
from .error_handling import ThreadingError, handle_errors, error_boundary


class ThreadState(Enum):
    """Thread execution states."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class ThreadInfo:
    """Thread information and metrics."""
    thread_id: int
    name: str
    state: ThreadState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_count: int = 0
    last_heartbeat: Optional[datetime] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreadSafeCounter:
    """Thread-safe counter with increment/decrement operations."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set value and return new value."""
        with self._lock:
            self._value = value
            return self._value
    
    def reset(self) -> int:
        """Reset to zero and return previous value."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class ThreadSafeDict:
    """Thread-safe dictionary wrapper."""
    
    def __init__(self, initial_data: Dict[Any, Any] = None):
        self._data = initial_data or {}
        self._lock = threading.RLock()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value by key."""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: Any, value: Any) -> None:
        """Set value by key."""
        with self._lock:
            self._data[key] = value
    
    def update(self, updates: Dict[Any, Any]) -> None:
        """Update multiple values."""
        with self._lock:
            self._data.update(updates)
    
    def delete(self, key: Any) -> bool:
        """Delete key if exists, return True if deleted."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def keys(self) -> List[Any]:
        """Get all keys."""
        with self._lock:
            return list(self._data.keys())
    
    def values(self) -> List[Any]:
        """Get all values."""
        with self._lock:
            return list(self._data.values())
    
    def items(self) -> List[tuple]:
        """Get all items."""
        with self._lock:
            return list(self._data.items())
    
    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._data.clear()
    
    def size(self) -> int:
        """Get size of dictionary."""
        with self._lock:
            return len(self._data)


class ManagedThread(threading.Thread):
    """Enhanced thread with state management and monitoring."""
    
    def __init__(
        self, 
        target: Callable = None,
        name: str = None,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        heartbeat_interval: float = 5.0
    ):
        super().__init__(target=target, name=name, args=args, kwargs=kwargs or {})
        
        self.logger = get_logger()
        self.state = ThreadState.CREATED
        self.priority = priority
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        
        self.error_count = 0
        self.last_heartbeat = None
        self.metadata = {}
        
        # For managing the target function
        self._target = target
        self._args = args
        self._kwargs = kwargs
    
    def run(self):
        """Run the thread with enhanced monitoring."""
        try:
            self.state = ThreadState.STARTING
            self.started_at = datetime.now()
            self.logger.debug(f"Thread {self.name} starting")
            
            self.state = ThreadState.RUNNING
            self._send_heartbeat()
            
            # Run the target function
            if self._target:
                self._run_with_monitoring()
            
            self.state = ThreadState.COMPLETED
            self.completed_at = datetime.now()
            self.logger.debug(f"Thread {self.name} completed successfully")
            
        except Exception as e:
            self.error_count += 1
            self.state = ThreadState.ERROR
            self.completed_at = datetime.now()
            self.logger.error(f"Thread {self.name} failed: {e}", exception=e)
            raise
    
    def _run_with_monitoring(self):
        """Run target function with pause/stop monitoring."""
        # Start heartbeat timer
        heartbeat_timer = threading.Timer(self.heartbeat_interval, self._heartbeat_worker)
        heartbeat_timer.daemon = True
        heartbeat_timer.start()
        
        try:
            # Run target with monitoring
            if hasattr(self._target, '__call__'):
                # Check for stop/pause periodically if target supports it
                if self._target_supports_monitoring():
                    self._run_monitored_target()
                else:
                    # Run normally
                    result = self._target(*self._args, **self._kwargs)
                    return result
        finally:
            heartbeat_timer.cancel()
    
    def _target_supports_monitoring(self) -> bool:
        """Check if target function supports monitoring (stop_event parameter)."""
        import inspect
        if not hasattr(self._target, '__call__'):
            return False
        
        sig = inspect.signature(self._target)
        return 'stop_event' in sig.parameters or 'pause_event' in sig.parameters
    
    def _run_monitored_target(self):
        """Run target function with monitoring support."""
        # Add monitoring events to kwargs
        enhanced_kwargs = self._kwargs.copy()
        
        import inspect
        sig = inspect.signature(self._target)
        if 'stop_event' in sig.parameters:
            enhanced_kwargs['stop_event'] = self._stop_event
        if 'pause_event' in sig.parameters:
            enhanced_kwargs['pause_event'] = self._pause_event
        
        return self._target(*self._args, **enhanced_kwargs)
    
    def _heartbeat_worker(self):
        """Send periodic heartbeats."""
        while not self._stop_event.is_set() and self.state == ThreadState.RUNNING:
            self._send_heartbeat()
            time.sleep(self.heartbeat_interval)
    
    def _send_heartbeat(self):
        """Send heartbeat signal."""
        self.last_heartbeat = datetime.now()
    
    def pause(self):
        """Pause thread execution."""
        if self.state == ThreadState.RUNNING:
            self.state = ThreadState.PAUSING
            self._pause_event.clear()
            self.state = ThreadState.PAUSED
            self.logger.debug(f"Thread {self.name} paused")
    
    def resume(self):
        """Resume thread execution."""
        if self.state == ThreadState.PAUSED:
            self._pause_event.set()
            self.state = ThreadState.RUNNING
            self.logger.debug(f"Thread {self.name} resumed")
    
    def stop(self, timeout: Optional[float] = None):
        """Stop thread execution."""
        if self.state in [ThreadState.RUNNING, ThreadState.PAUSED]:
            self.state = ThreadState.STOPPING
            self._stop_event.set()
            self._pause_event.set()  # Unblock if paused
            
            if timeout:
                self.join(timeout)
                if self.is_alive():
                    self.logger.warning(f"Thread {self.name} did not stop within {timeout}s")
            
            self.state = ThreadState.STOPPED
            self.logger.debug(f"Thread {self.name} stopped")
    
    def wait_for_pause_point(self):
        """Wait for pause event (to be called within target function)."""
        self._pause_event.wait()
    
    def should_stop(self) -> bool:
        """Check if thread should stop (to be called within target function)."""
        return self._stop_event.is_set()
    
    def get_info(self) -> ThreadInfo:
        """Get thread information."""
        return ThreadInfo(
            thread_id=self.ident or 0,
            name=self.name or "unnamed",
            state=self.state,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error_count=self.error_count,
            last_heartbeat=self.last_heartbeat,
            priority=self.priority,
            metadata=self.metadata.copy()
        )


class SafeQtThread(QThread if PYQT_AVAILABLE else threading.Thread):
    """Thread-safe Qt thread wrapper with enhanced error handling."""
    
    if PYQT_AVAILABLE:
        progress_updated = pyqtSignal(int, str)
        error_occurred = pyqtSignal(str)
        finished_successfully = pyqtSignal()
    
    def __init__(self, target: Callable = None, *args, **kwargs):
        super().__init__()
        self.logger = get_logger()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.should_stop_flag = threading.Event()
        self.error_count = 0
        
        if PYQT_AVAILABLE:
            self.finished.connect(self._on_finished)
    
    def run(self):
        """Run the Qt thread safely."""
        try:
            self.logger.debug(f"Qt thread {self.objectName()} starting")
            
            if self.target:
                # Add stop_event to kwargs if target supports it
                enhanced_kwargs = self.kwargs.copy()
                enhanced_kwargs['stop_event'] = self.should_stop_flag
                
                result = self.target(*self.args, **enhanced_kwargs)
                
                if PYQT_AVAILABLE:
                    self.finished_successfully.emit()
                
                return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Qt thread {self.objectName()} failed: {e}", exception=e)
            
            if PYQT_AVAILABLE:
                self.error_occurred.emit(str(e))
            
            raise
    
    def stop_safely(self, timeout: float = 5.0):
        """Stop thread safely."""
        self.should_stop_flag.set()
        
        if not self.wait(int(timeout * 1000)):  # Convert to milliseconds
            self.logger.warning(f"Qt thread {self.objectName()} did not stop within {timeout}s")
            self.terminate()
            self.wait(1000)  # Wait 1 second for termination
    
    def _on_finished(self):
        """Handle thread finished signal."""
        self.logger.debug(f"Qt thread {self.objectName()} finished")


class ThreadPool:
    """Enhanced thread pool with monitoring and resource management."""
    
    def __init__(
        self, 
        max_workers: int = 4,
        thread_name_prefix: str = "SpygateWorker",
        timeout: Optional[float] = None
    ):
        self.logger = get_logger()
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.timeout = timeout
        
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        self._active_futures: Set[Future] = set()
        self._completed_futures: List[Future] = []
        self._lock = threading.RLock()
        
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
    
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        """Submit a task to the thread pool."""
        with self._lock:
            future = self._executor.submit(fn, *args, **kwargs)
            self._active_futures.add(future)
            self.stats["submitted"] += 1
            
            # Add completion callback
            future.add_done_callback(self._on_future_done)
            
            self.logger.debug(f"Task submitted to thread pool: {fn.__name__}")
            return future
    
    def submit_batch(self, tasks: List[tuple]) -> List[Future]:
        """Submit multiple tasks at once."""
        futures = []
        for task in tasks:
            if len(task) == 1:
                fn = task[0]
                args, kwargs = (), {}
            elif len(task) == 2:
                fn, args = task
                kwargs = {}
            elif len(task) == 3:
                fn, args, kwargs = task
            else:
                raise ValueError("Task tuple must be (fn,), (fn, args), or (fn, args, kwargs)")
            
            future = self.submit(fn, *args, **kwargs)
            futures.append(future)
        
        self.logger.info(f"Submitted batch of {len(tasks)} tasks")
        return futures
    
    def _on_future_done(self, future: Future):
        """Handle completed future."""
        with self._lock:
            if future in self._active_futures:
                self._active_futures.remove(future)
            
            self._completed_futures.append(future)
            
            # Update stats
            if future.cancelled():
                self.stats["cancelled"] += 1
            elif future.exception():
                self.stats["failed"] += 1
                self.logger.error(f"Thread pool task failed: {future.exception()}")
            else:
                self.stats["completed"] += 1
            
            # Cleanup old completed futures (keep last 100)
            if len(self._completed_futures) > 100:
                self._completed_futures = self._completed_futures[-100:]
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all active tasks to complete."""
        if not self._active_futures:
            return True
        
        try:
            for future in as_completed(self._active_futures, timeout=timeout):
                pass
            return True
        except TimeoutError:
            self.logger.warning(f"Not all tasks completed within {timeout}s")
            return False
    
    def cancel_all(self):
        """Cancel all pending tasks."""
        with self._lock:
            cancelled_count = 0
            for future in list(self._active_futures):
                if future.cancel():
                    cancelled_count += 1
            
            self.logger.info(f"Cancelled {cancelled_count} pending tasks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "active_tasks": len(self._active_futures),
                "completed_tasks": len(self._completed_futures),
                **self.stats
            }
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the thread pool."""
        self.logger.info("Shutting down thread pool")
        
        if timeout:
            self.wait_for_completion(timeout)
        
        self._executor.shutdown(wait=wait)
        
        with self._lock:
            self._active_futures.clear()
            self._completed_futures.clear()


class ThreadManager:
    """Central thread management system."""
    
    def __init__(self):
        self.logger = get_logger()
        self._threads: Dict[str, ManagedThread] = {}
        self._qt_threads: Dict[str, SafeQtThread] = {}
        self._thread_pools: Dict[str, ThreadPool] = {}
        self._lock = threading.RLock()
        
        # Monitoring
        self._monitor_thread = None
        self._monitor_interval = 30.0  # seconds
        self._start_monitoring()
    
    def create_thread(
        self,
        name: str,
        target: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        auto_start: bool = False
    ) -> ManagedThread:
        """Create a managed thread."""
        if name in self._threads:
            raise ThreadingError(f"Thread '{name}' already exists")
        
        thread = ManagedThread(
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout
        )
        
        with self._lock:
            self._threads[name] = thread
        
        self.logger.info(f"Created thread: {name}")
        
        if auto_start:
            thread.start()
        
        return thread
    
    def create_qt_thread(
        self,
        name: str,
        target: Callable,
        *args,
        auto_start: bool = False,
        **kwargs
    ) -> Optional[SafeQtThread]:
        """Create a Qt thread (returns None if Qt not available)."""
        if not PYQT_AVAILABLE:
            self.logger.warning("Qt not available, cannot create Qt thread")
            return None
        
        if name in self._qt_threads:
            raise ThreadingError(f"Qt thread '{name}' already exists")
        
        thread = SafeQtThread(target=target, *args, **kwargs)
        thread.setObjectName(name)
        
        with self._lock:
            self._qt_threads[name] = thread
        
        self.logger.info(f"Created Qt thread: {name}")
        
        if auto_start:
            thread.start()
        
        return thread
    
    def create_thread_pool(
        self,
        name: str,
        max_workers: int = 4,
        thread_name_prefix: str = None
    ) -> ThreadPool:
        """Create a thread pool."""
        if name in self._thread_pools:
            raise ThreadingError(f"Thread pool '{name}' already exists")
        
        pool = ThreadPool(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix or f"Pool-{name}"
        )
        
        with self._lock:
            self._thread_pools[name] = pool
        
        self.logger.info(f"Created thread pool: {name} (workers: {max_workers})")
        return pool
    
    def get_thread(self, name: str) -> Optional[ManagedThread]:
        """Get thread by name."""
        return self._threads.get(name)
    
    def get_qt_thread(self, name: str) -> Optional[SafeQtThread]:
        """Get Qt thread by name."""
        return self._qt_threads.get(name)
    
    def get_thread_pool(self, name: str) -> Optional[ThreadPool]:
        """Get thread pool by name."""
        return self._thread_pools.get(name)
    
    def stop_thread(self, name: str, timeout: Optional[float] = None) -> bool:
        """Stop a thread."""
        thread = self._threads.get(name)
        if not thread:
            return False
        
        try:
            thread.stop(timeout)
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop thread {name}: {e}")
            return False
    
    def stop_qt_thread(self, name: str, timeout: float = 5.0) -> bool:
        """Stop a Qt thread."""
        thread = self._qt_threads.get(name)
        if not thread:
            return False
        
        try:
            thread.stop_safely(timeout)
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Qt thread {name}: {e}")
            return False
    
    def stop_all_threads(self, timeout: Optional[float] = None):
        """Stop all managed threads."""
        self.logger.info("Stopping all threads")
        
        # Stop regular threads
        for name, thread in list(self._threads.items()):
            try:
                thread.stop(timeout)
            except Exception as e:
                self.logger.error(f"Error stopping thread {name}: {e}")
        
        # Stop Qt threads
        for name, thread in list(self._qt_threads.items()):
            try:
                thread.stop_safely(timeout or 5.0)
            except Exception as e:
                self.logger.error(f"Error stopping Qt thread {name}: {e}")
        
        # Shutdown thread pools
        for name, pool in list(self._thread_pools.items()):
            try:
                pool.shutdown(timeout=timeout)
            except Exception as e:
                self.logger.error(f"Error shutting down thread pool {name}: {e}")
    
    def cleanup_finished_threads(self):
        """Remove finished threads from tracking."""
        with self._lock:
            # Clean up regular threads
            finished_threads = [
                name for name, thread in self._threads.items()
                if not thread.is_alive()
            ]
            
            for name in finished_threads:
                del self._threads[name]
                self.logger.debug(f"Cleaned up finished thread: {name}")
            
            # Clean up Qt threads
            finished_qt_threads = [
                name for name, thread in self._qt_threads.items()
                if thread.isFinished()
            ]
            
            for name in finished_qt_threads:
                del self._qt_threads[name]
                self.logger.debug(f"Cleaned up finished Qt thread: {name}")
    
    def get_thread_status(self) -> Dict[str, Any]:
        """Get comprehensive thread status."""
        with self._lock:
            regular_threads = {
                name: {
                    "type": "regular",
                    "state": thread.state.value,
                    "alive": thread.is_alive(),
                    "error_count": thread.error_count,
                    "priority": thread.priority,
                    "created_at": thread.created_at.isoformat(),
                    "last_heartbeat": thread.last_heartbeat.isoformat() if thread.last_heartbeat else None
                }
                for name, thread in self._threads.items()
            }
            
            qt_threads = {
                name: {
                    "type": "qt",
                    "running": thread.isRunning(),
                    "finished": thread.isFinished(),
                    "error_count": thread.error_count
                }
                for name, thread in self._qt_threads.items()
            } if PYQT_AVAILABLE else {}
            
            thread_pools = {
                name: pool.get_stats()
                for name, pool in self._thread_pools.items()
            }
            
            return {
                "regular_threads": regular_threads,
                "qt_threads": qt_threads,
                "thread_pools": thread_pools,
                "total_threads": len(regular_threads) + len(qt_threads),
                "monitoring_active": self._monitor_thread is not None and self._monitor_thread.is_alive()
            }
    
    def _start_monitoring(self):
        """Start thread monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            name="ThreadMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Thread monitoring started")
    
    def _monitor_worker(self):
        """Monitor thread health."""
        while True:
            try:
                time.sleep(self._monitor_interval)
                
                # Check for dead threads and clean up
                self.cleanup_finished_threads()
                
                # Check for hung threads (no heartbeat)
                current_time = datetime.now()
                hung_threads = []
                
                for name, thread in self._threads.items():
                    if (thread.last_heartbeat and 
                        current_time - thread.last_heartbeat > timedelta(minutes=5) and
                        thread.is_alive()):
                        hung_threads.append(name)
                
                if hung_threads:
                    self.logger.warning(f"Potentially hung threads detected: {hung_threads}")
                
                # Log periodic status
                status = self.get_thread_status()
                active_count = sum(1 for t in status["regular_threads"].values() if t["alive"])
                active_qt_count = sum(1 for t in status["qt_threads"].values() if t["running"])
                
                self.logger.debug(
                    f"Thread status: {active_count} regular, {active_qt_count} Qt, "
                    f"{len(status['thread_pools'])} pools"
                )
                
            except Exception as e:
                self.logger.error(f"Thread monitoring error: {e}")


# Global thread manager instance
_global_thread_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    """Get or create the global thread manager instance."""
    global _global_thread_manager
    if _global_thread_manager is None:
        _global_thread_manager = ThreadManager()
    return _global_thread_manager


# Convenience functions and decorators
def thread_safe(lock: Optional[threading.Lock] = None):
    """Decorator to make function thread-safe."""
    if lock is None:
        lock = threading.RLock()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def temporary_thread(
    target: Callable,
    name: str = None,
    args: tuple = (),
    kwargs: Dict[str, Any] = None,
    timeout: Optional[float] = None
):
    """Context manager for temporary thread execution."""
    thread_manager = get_thread_manager()
    thread_name = name or f"temp_thread_{int(time.time())}"
    
    thread = thread_manager.create_thread(
        thread_name,
        target,
        args=args,
        kwargs=kwargs or {},
        auto_start=True
    )
    
    try:
        yield thread
        
        # Wait for completion
        thread.join(timeout)
        
        if thread.is_alive():
            thread.stop(5.0)  # Give 5 seconds to stop gracefully
            
    finally:
        # Cleanup
        if thread.is_alive():
            thread_manager.stop_thread(thread_name, 1.0)


def run_in_background(func: Callable = None, *, name: str = None, timeout: Optional[float] = None):
    """Decorator to run function in background thread."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_manager = get_thread_manager()
            thread_name = name or f"{func.__name__}_{int(time.time())}"
            
            thread = thread_manager.create_thread(
                thread_name,
                func,
                args=args,
                kwargs=kwargs,
                timeout=timeout,
                auto_start=True
            )
            
            return thread
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


if __name__ == "__main__":
    # Test the threading utilities
    logger = get_logger()
    manager = get_thread_manager()
    
    logger.info("Testing threading utilities")
    
    # Test counter
    counter = ThreadSafeCounter(0)
    
    def increment_worker():
        for _ in range(100):
            counter.increment()
            time.sleep(0.001)
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = manager.create_thread(f"worker_{i}", increment_worker, auto_start=True)
        threads.append(thread)
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    logger.info(f"Counter final value: {counter.get()}")
    
    # Test thread pool
    pool = manager.create_thread_pool("test_pool", max_workers=3)
    
    def square(x):
        time.sleep(0.1)
        return x * x
    
    futures = [pool.submit(square, i) for i in range(10)]
    
    for i, future in enumerate(futures):
        result = future.result()
        logger.info(f"Square of {i}: {result}")
    
    # Get status
    status = manager.get_thread_status()
    logger.info(f"Thread status: {status}")
    
    # Cleanup
    manager.stop_all_threads(5.0)
    logger.info("Threading test completed")