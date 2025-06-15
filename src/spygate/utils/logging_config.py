"""
Comprehensive logging configuration for SpygateAI.

This module provides a centralized logging system with proper configuration,
error tracking, and performance monitoring.
"""

import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import json
import threading


class SpygateFormatter(logging.Formatter):
    """Custom formatter for SpygateAI logs with enhanced context."""
    
    def __init__(self):
        # Set default format and date format for the parent formatter
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.start_time = datetime.now()
    
    def format(self, record):
        # First, call parent format to ensure asctime is populated
        try:
            record.asctime = self.formatTime(record, self.datefmt)
        except Exception:
            record.asctime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add thread information safely
        try:
            record.thread_name = threading.current_thread().name
            record.thread_id = threading.current_thread().ident
        except Exception:
            record.thread_name = "unknown"
            record.thread_id = 0
        
        # Add module context safely
        try:
            record.module_path = getattr(record, 'pathname', 'unknown')
            if hasattr(record, 'pathname') and record.pathname:
                try:
                    record.relative_path = str(Path(record.pathname).relative_to(Path.cwd()))
                except ValueError:
                    # If path is not relative to cwd, just use the filename
                    record.relative_path = Path(record.pathname).name
            else:
                record.relative_path = 'unknown'
        except Exception:
            record.relative_path = 'unknown'
        
        # Add execution time safely
        try:
            elapsed = datetime.now() - self.start_time
            record.elapsed_time = f"{elapsed.total_seconds():.3f}s"
        except Exception:
            record.elapsed_time = "0s"
        
        # Color coding for console output
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[41m', # Red background
            'RESET': '\033[0m'      # Reset
        }
        
        level_color = colors.get(record.levelname, colors['RESET'])
        reset_color = colors['RESET']
        
        # Format message with context - use a simpler, more reliable format
        try:
            base_format = (
                f"{level_color}[{record.levelname:8}]{reset_color} "
                f"{record.asctime} | "
                f"T:{record.thread_name[:10]:>10} | "
                f"{record.relative_path[:30]:>30}:{getattr(record, 'lineno', 0):<4} | "
                f"{record.elapsed_time:>8} | "
                f"{record.getMessage()}"
            )
        except Exception:
            # Fallback to basic format if anything goes wrong
            base_format = f"{record.asctime} - {record.name} - {record.levelname} - {record.getMessage()}"
        
        # Add exception info if present
        try:
            if record.exc_info:
                base_format += f"\n{self.formatException(record.exc_info)}"
        except Exception:
            pass
        
        return base_format


class SpygateLogger:
    """Enhanced logger for SpygateAI with context management and error tracking."""
    
    def __init__(self, name: str = "spygate", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self.error_count = 0
        self.warning_count = 0
        self.start_time = datetime.now()
        
        # Thread-local storage for context
        self._local = threading.local()
    
    def _setup_handlers(self):
        """Set up logging handlers for console and file output."""
        formatter = SpygateFormatter()
        
        # Simple file formatter without color codes
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler with color support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Main log file with rotation
        main_log_file = self.log_dir / "spygate.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance log file
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=2
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(lambda record: hasattr(record, 'performance_metric'))
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(perf_handler)
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to log messages within this block."""
        old_context = getattr(self._local, 'context', {})
        self._local.context = {**old_context, **kwargs}
        try:
            yield
        finally:
            self._local.context = old_context
    
    def _add_context(self, message: str) -> str:
        """Add context to log message if available."""
        context = getattr(self._local, 'context', {})
        if context:
            context_str = " | ".join(f"{k}={v}" for k, v in context.items())
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._add_context(message), extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(self._add_context(message), extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.warning_count += 1
        self.logger.warning(self._add_context(message), extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with context and optional exception."""
        self.error_count += 1
        if exception:
            self.logger.error(self._add_context(message), exc_info=exception, extra=kwargs)
        else:
            self.logger.error(self._add_context(message), extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with context and optional exception."""
        self.error_count += 1
        if exception:
            self.logger.critical(self._add_context(message), exc_info=exception, extra=kwargs)
        else:
            self.logger.critical(self._add_context(message), extra=kwargs)
    
    def performance(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """Log performance metric."""
        message = f"PERF | {metric_name}: {value}{unit}"
        if kwargs:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message += f" | {context}"
        
        self.logger.info(message, extra={'performance_metric': True})
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None, duration: float = None):
        """Log function call with parameters and duration."""
        kwargs = kwargs or {}
        
        # Sanitize sensitive information
        safe_args = [str(arg)[:100] if not isinstance(arg, (int, float, bool)) else arg for arg in args]
        safe_kwargs = {k: (str(v)[:100] if not isinstance(v, (int, float, bool)) else v) for k, v in kwargs.items()}
        
        message = f"CALL | {func_name}({', '.join(map(str, safe_args))}"
        if safe_kwargs:
            message += f", {safe_kwargs}"
        message += ")"
        
        if duration is not None:
            message += f" | Duration: {duration:.3f}s"
        
        self.debug(message)
    
    def log_gpu_memory(self, used_mb: float, total_mb: float, context: str = ""):
        """Log GPU memory usage."""
        usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        message = f"GPU_MEM | {used_mb:.1f}MB / {total_mb:.1f}MB ({usage_percent:.1f}%)"
        if context:
            message += f" | {context}"
        
        if usage_percent > 90:
            self.warning(message)
        else:
            self.performance("gpu_memory_usage", usage_percent, "%", used_mb=used_mb, total_mb=total_mb)
    
    def log_system_stats(self, cpu_percent: float, ram_mb: float, ram_total_mb: float):
        """Log system performance statistics."""
        ram_percent = (ram_mb / ram_total_mb) * 100 if ram_total_mb > 0 else 0
        
        self.performance("cpu_usage", cpu_percent, "%")
        self.performance("ram_usage", ram_percent, "%", used_mb=ram_mb, total_mb=ram_total_mb)
        
        if cpu_percent > 95:
            self.warning(f"High CPU usage: {cpu_percent:.1f}%")
        if ram_percent > 90:
            self.warning(f"High RAM usage: {ram_percent:.1f}% ({ram_mb:.1f}MB/{ram_total_mb:.1f}MB)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = datetime.now() - self.start_time
        return {
            "uptime_seconds": uptime.total_seconds(),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "log_level": self.logger.level,
            "handlers_count": len(self.logger.handlers)
        }
    
    def save_session_summary(self):
        """Save a summary of the logging session."""
        stats = self.get_stats()
        summary_file = self.log_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.info(f"Session summary saved to {summary_file}")


# Global logger instance
_global_logger: Optional[SpygateLogger] = None


def get_logger(name: str = "spygate") -> SpygateLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SpygateLogger(name)
        _global_logger.info("SpygateAI logging system initialized")
    return _global_logger


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> SpygateLogger:
    """Set up logging configuration for SpygateAI."""
    logger = SpygateLogger("spygate", log_dir)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.logger.setLevel(numeric_level)
    
    return logger


@contextmanager
def log_execution_time(logger: SpygateLogger, operation_name: str):
    """Context manager to log execution time of operations."""
    start_time = datetime.now()
    logger.debug(f"Starting {operation_name}")
    
    try:
        yield
        duration = (datetime.now() - start_time).total_seconds()
        logger.performance("execution_time", duration, "s", operation=operation_name)
        logger.debug(f"Completed {operation_name} in {duration:.3f}s")
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed {operation_name} after {duration:.3f}s", exception=e)
        raise


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler for uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to work normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = get_logger()
    logger.critical(
        f"Uncaught exception: {exc_type.__name__}: {exc_value}",
        exception=exc_value
    )


# Install global exception handler
sys.excepthook = handle_uncaught_exception


if __name__ == "__main__":
    # Test the logging system
    logger = setup_logging("DEBUG")
    
    logger.info("Testing SpygateAI logging system")
    logger.debug("Debug message")
    logger.warning("Warning message")
    
    with logger.context(module="test", operation="demo"):
        logger.info("Message with context")
        logger.performance("test_metric", 123.45, "ms")
    
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        logger.error("Caught test exception", exception=e)
    
    logger.log_gpu_memory(1024.5, 8192.0, "during_inference")
    logger.log_system_stats(45.2, 2048.0, 8192.0)
    
    print("\nLogging stats:", logger.get_stats())