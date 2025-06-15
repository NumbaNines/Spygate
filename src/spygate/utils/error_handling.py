"""
Comprehensive Error Handling System for SpygateAI.

This module provides robust error handling, recovery mechanisms, and error boundaries
to ensure the application remains stable under all conditions.
"""

import functools
import inspect
import sys
import traceback
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import json

from .logging_config import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE = "database"
    FILE_IO = "file_io"
    VALIDATION = "validation"
    THREADING = "threading"
    GUI = "gui"
    ML_MODEL = "ml_model"
    VIDEO_PROCESSING = "video_processing"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    timestamp: datetime
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    exception_type: str
    message: str
    traceback_str: str
    function_name: str
    module_name: str
    line_number: int
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


class SpygateError(Exception):
    """Base exception class for SpygateAI with enhanced context."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()


class GPUError(SpygateError):
    """GPU-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.GPU, severity=ErrorSeverity.HIGH, **kwargs)


class DatabaseError(SpygateError):
    """Database-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, severity=ErrorSeverity.MEDIUM, **kwargs)


class ValidationError(SpygateError):
    """Input validation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, severity=ErrorSeverity.LOW, **kwargs)


class VideoProcessingError(SpygateError):
    """Video processing errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VIDEO_PROCESSING, severity=ErrorSeverity.MEDIUM, **kwargs)


class MLModelError(SpygateError):
    """Machine learning model errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.ML_MODEL, severity=ErrorSeverity.HIGH, **kwargs)


class ThreadingError(SpygateError):
    """Threading and concurrency errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.THREADING, severity=ErrorSeverity.HIGH, **kwargs)


class GUIError(SpygateError):
    """GUI-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.GUI, severity=ErrorSeverity.MEDIUM, **kwargs)


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can recover from the given error."""
        raise NotImplementedError
    
    def recover(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from the error. Returns True if successful."""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry the operation with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_counts: Dict[str, int] = {}
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        key = f"{error_info.function_name}:{error_info.line_number}"
        return self.retry_counts.get(key, 0) < self.max_retries
    
    def recover(self, error_info: ErrorInfo) -> bool:
        key = f"{error_info.function_name}:{error_info.line_number}"
        retry_count = self.retry_counts.get(key, 0) + 1
        self.retry_counts[key] = retry_count
        
        if retry_count <= self.max_retries:
            import time
            delay = self.base_delay * (2 ** (retry_count - 1))  # Exponential backoff
            time.sleep(delay)
            return True
        return False


class FallbackStrategy(ErrorRecoveryStrategy):
    """Use fallback methods or default values."""
    
    def __init__(self, fallback_values: Dict[str, Any] = None):
        self.fallback_values = fallback_values or {}
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category in [ErrorCategory.NETWORK, ErrorCategory.FILE_IO, ErrorCategory.GPU]
    
    def recover(self, error_info: ErrorInfo) -> bool:
        # This would need to be implemented specific to each use case
        return True


class ResourceCleanupStrategy(ErrorRecoveryStrategy):
    """Clean up resources and retry."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        return error_info.category in [ErrorCategory.GPU, ErrorCategory.DATABASE, ErrorCategory.THREADING]
    
    def recover(self, error_info: ErrorInfo) -> bool:
        try:
            if error_info.category == ErrorCategory.GPU:
                # GPU cleanup
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
            elif error_info.category == ErrorCategory.DATABASE:
                # Database cleanup would be handled by resource manager
                pass
            return True
        except:
            return False


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        self.logger = get_logger()
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 1000
        self.recovery_strategies: List[ErrorRecoveryStrategy] = [
            RetryStrategy(),
            ResourceCleanupStrategy(),
            FallbackStrategy()
        ]
        
        # Error statistics
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_success_rate: Dict[ErrorCategory, float] = {}
        
        # Circuit breaker pattern
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_threshold = 5  # failures before opening circuit
        self.circuit_breaker_timeout = timedelta(minutes=5)
        
        self.lock = threading.RLock()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID."""
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def _extract_error_context(self, frame) -> Dict[str, Any]:
        """Extract context information from the error location."""
        try:
            context = {
                "function": frame.f_code.co_name,
                "filename": frame.f_code.co_filename,
                "line_number": frame.f_lineno,
                "local_vars": {}
            }
            
            # Safely extract local variables (avoid sensitive data)
            for name, value in frame.f_locals.items():
                if not name.startswith('_') and name not in ['password', 'token', 'key']:
                    try:
                        # Convert to string safely
                        str_value = str(value)
                        if len(str_value) < 200:  # Avoid huge objects
                            context["local_vars"][name] = str_value
                    except:
                        context["local_vars"][name] = f"<{type(value).__name__}>"
            
            return context
        except:
            return {"function": "unknown", "filename": "unknown", "line_number": 0, "local_vars": {}}
    
    def _get_error_info(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Extract comprehensive error information."""
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Get the frame where the error occurred
        if exc_traceback:
            frame = exc_traceback.tb_frame
            line_number = exc_traceback.tb_lineno
        else:
            frame = inspect.currentframe()
            line_number = frame.f_lineno if frame else 0
        
        # Extract context from the error location
        error_context = self._extract_error_context(frame) if frame else {}
        if context:
            error_context.update(context)
        
        # Determine category and severity
        category = ErrorCategory.SYSTEM
        severity = ErrorSeverity.MEDIUM
        
        if isinstance(exception, SpygateError):
            category = exception.category
            severity = exception.severity
        else:
            # Classify common exceptions
            if isinstance(exception, (MemoryError, RuntimeError)) and "CUDA" in str(exception):
                category = ErrorCategory.GPU
                severity = ErrorSeverity.HIGH
            elif isinstance(exception, (ConnectionError, TimeoutError)):
                category = ErrorCategory.NETWORK
                severity = ErrorSeverity.MEDIUM
            elif isinstance(exception, (FileNotFoundError, PermissionError, IOError)):
                category = ErrorCategory.FILE_IO
                severity = ErrorSeverity.MEDIUM
            elif isinstance(exception, (ValueError, TypeError)) and "validation" in str(exception).lower():
                category = ErrorCategory.VALIDATION
                severity = ErrorSeverity.LOW
            elif isinstance(exception, threading.ThreadError):
                category = ErrorCategory.THREADING
                severity = ErrorSeverity.HIGH
        
        return ErrorInfo(
            timestamp=datetime.now(),
            error_id=self._generate_error_id(),
            category=category,
            severity=severity,
            exception_type=exc_type.__name__ if exc_type else type(exception).__name__,
            message=str(exception),
            traceback_str=traceback.format_exc(),
            function_name=error_context.get("function", "unknown"),
            module_name=error_context.get("filename", "unknown"),
            line_number=line_number,
            context=error_context
        )
    
    def _check_circuit_breaker(self, operation_key: str) -> bool:
        """Check if circuit breaker is open for the given operation."""
        with self.lock:
            if operation_key not in self.circuit_breakers:
                self.circuit_breakers[operation_key] = {
                    "failure_count": 0,
                    "last_failure": None,
                    "state": "closed"  # closed, open, half-open
                }
            
            breaker = self.circuit_breakers[operation_key]
            
            if breaker["state"] == "open":
                # Check if timeout has passed
                if (datetime.now() - breaker["last_failure"]) > self.circuit_breaker_timeout:
                    breaker["state"] = "half-open"
                    self.logger.info(f"Circuit breaker half-opened for {operation_key}")
                    return False
                return True
            
            return False
    
    def _update_circuit_breaker(self, operation_key: str, success: bool):
        """Update circuit breaker state based on operation result."""
        with self.lock:
            breaker = self.circuit_breakers[operation_key]
            
            if success:
                if breaker["state"] == "half-open":
                    breaker["state"] = "closed"
                    breaker["failure_count"] = 0
                    self.logger.info(f"Circuit breaker closed for {operation_key}")
            else:
                breaker["failure_count"] += 1
                breaker["last_failure"] = datetime.now()
                
                if breaker["failure_count"] >= self.circuit_breaker_threshold:
                    breaker["state"] = "open"
                    self.logger.warning(f"Circuit breaker opened for {operation_key} after {breaker['failure_count']} failures")
    
    def handle_error(
        self, 
        exception: Exception, 
        context: Dict[str, Any] = None,
        attempt_recovery: bool = True
    ) -> Optional[ErrorInfo]:
        """Handle an error with optional recovery attempt."""
        error_info = self._get_error_info(exception, context)
        
        with self.lock:
            # Add to history
            self.error_history.append(error_info)
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # Update statistics
            self.error_counts[error_info.category] = self.error_counts.get(error_info.category, 0) + 1
        
        # Log the error
        log_message = f"[{error_info.error_id}] {error_info.category.value.upper()} ERROR in {error_info.function_name}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exception=exception)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exception=exception)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.debug(log_message)
        
        # Attempt recovery if requested and possible
        if attempt_recovery and error_info.severity != ErrorSeverity.CRITICAL:
            operation_key = f"{error_info.function_name}:{error_info.category.value}"
            
            # Check circuit breaker
            if self._check_circuit_breaker(operation_key):
                self.logger.warning(f"Circuit breaker open for {operation_key}, skipping recovery")
                return error_info
            
            error_info.recovery_attempted = True
            
            for strategy in self.recovery_strategies:
                if strategy.can_recover(error_info):
                    try:
                        if strategy.recover(error_info):
                            error_info.recovery_successful = True
                            self.logger.info(f"[{error_info.error_id}] Recovery successful using {strategy.__class__.__name__}")
                            self._update_circuit_breaker(operation_key, True)
                            break
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery strategy {strategy.__class__.__name__} failed: {recovery_error}")
            
            self._update_circuit_breaker(operation_key, error_info.recovery_successful)
            
            if not error_info.recovery_successful:
                self.logger.warning(f"[{error_info.error_id}] All recovery strategies failed")
        
        return error_info
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self.lock:
            if not self.error_history:
                return {"total_errors": 0}
            
            total_errors = len(self.error_history)
            recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp) < timedelta(hours=1)]
            
            # Calculate recovery success rates
            recovery_stats = {}
            for category in ErrorCategory:
                category_errors = [e for e in self.error_history if e.category == category and e.recovery_attempted]
                if category_errors:
                    successful_recoveries = sum(1 for e in category_errors if e.recovery_successful)
                    recovery_stats[category.value] = successful_recoveries / len(category_errors)
                else:
                    recovery_stats[category.value] = 0.0
            
            # Severity distribution
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = sum(1 for e in self.error_history if e.severity == severity)
            
            return {
                "total_errors": total_errors,
                "recent_errors_1h": len(recent_errors),
                "errors_by_category": dict(self.error_counts),
                "recovery_success_rate": recovery_stats,
                "severity_distribution": severity_counts,
                "circuit_breakers": {k: v["state"] for k, v in self.circuit_breakers.items()},
                "most_common_errors": self._get_most_common_errors(5)
            }
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common error patterns."""
        error_patterns = {}
        
        for error in self.error_history:
            pattern_key = f"{error.category.value}:{error.exception_type}"
            if pattern_key not in error_patterns:
                error_patterns[pattern_key] = {
                    "category": error.category.value,
                    "exception_type": error.exception_type,
                    "count": 0,
                    "latest_message": error.message
                }
            error_patterns[pattern_key]["count"] += 1
            error_patterns[pattern_key]["latest_message"] = error.message
        
        return sorted(error_patterns.values(), key=lambda x: x["count"], reverse=True)[:limit]
    
    def save_error_report(self, filepath: Optional[str] = None):
        """Save comprehensive error report."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"error_report_{timestamp}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "recent_errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "exception_type": e.exception_type,
                    "message": e.message,
                    "function_name": e.function_name,
                    "module_name": e.module_name,
                    "line_number": e.line_number,
                    "recovery_attempted": e.recovery_attempted,
                    "recovery_successful": e.recovery_successful
                }
                for e in self.error_history[-50:]  # Last 50 errors
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Error report saved to {filepath}")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


# Decorator for automatic error handling
def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    attempt_recovery: bool = True,
    reraise: bool = True
):
    """Decorator for automatic error handling with optional recovery."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add function context
                context = {
                    "function_args": str(args)[:200],
                    "function_kwargs": str(kwargs)[:200],
                    "function_name": func.__name__,
                    "module_name": func.__module__
                }
                
                # Handle the error
                error_handler = get_error_handler()
                error_info = error_handler.handle_error(e, context, attempt_recovery)
                
                # If recovery was successful, try the function again
                if error_info and error_info.recovery_successful:
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        # If retry fails, handle the new error but don't attempt recovery again
                        error_handler.handle_error(retry_error, context, False)
                        if reraise:
                            raise
                        return None
                
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


@contextmanager
def error_boundary(
    operation_name: str,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    attempt_recovery: bool = True,
    suppress_errors: bool = False
):
    """Context manager that creates an error boundary around operations."""
    error_handler = get_error_handler()
    logger = get_logger()
    
    logger.debug(f"Entering error boundary: {operation_name}")
    
    try:
        yield
        logger.debug(f"Exiting error boundary: {operation_name} (success)")
    except Exception as e:
        context = {"operation_name": operation_name}
        error_info = error_handler.handle_error(e, context, attempt_recovery)
        
        logger.debug(f"Exiting error boundary: {operation_name} (error: {error_info.error_id if error_info else 'unknown'})")
        
        if not suppress_errors:
            raise


# Validation helpers
def validate_input(value: Any, validators: List[Callable[[Any], bool]], error_message: str = "Validation failed"):
    """Validate input using a list of validator functions."""
    for validator in validators:
        if not validator(value):
            raise ValidationError(f"{error_message}: {value}")


def validate_file_path(path: Union[str, Path], must_exist: bool = True, extensions: List[str] = None):
    """Validate file path."""
    path = Path(path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")
    
    if extensions and path.suffix.lower() not in [ext.lower() for ext in extensions]:
        raise ValidationError(f"Invalid file extension. Expected: {extensions}, got: {path.suffix}")


def validate_gpu_availability():
    """Validate GPU availability."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise GPUError("CUDA is not available")
        if torch.cuda.device_count() == 0:
            raise GPUError("No CUDA devices found")
    except ImportError:
        raise GPUError("PyTorch not available")


if __name__ == "__main__":
    # Test the error handling system
    handler = get_error_handler()
    logger = get_logger()
    
    logger.info("Testing error handling system")
    
    # Test basic error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        handler.handle_error(e, {"test_context": "basic_test"})
    
    # Test error boundary
    with error_boundary("test_operation", suppress_errors=True):
        raise RuntimeError("Test boundary error")
    
    # Test decorator
    @handle_errors(category=ErrorCategory.VALIDATION, reraise=False)
    def test_function():
        raise ValidationError("Test validation error")
    
    result = test_function()
    logger.info(f"Test function result: {result}")
    
    # Test validation
    try:
        validate_file_path("/nonexistent/file.txt")
    except ValidationError as e:
        logger.info(f"Validation error caught: {e}")
    
    # Print statistics
    stats = handler.get_error_statistics()
    logger.info(f"Error statistics: {stats}")
    
    # Save report
    handler.save_error_report("test_error_report.json")