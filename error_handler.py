"""
SpygateAI Error Handler & Logging System
Comprehensive error handling with smart recovery and detailed logging
"""

import functools
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


class SpygateLogger:
    """Smart logging system for SpygateAI"""

    def __init__(self, log_dir: str = "logs"):
        """Initialize logging system"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup loggers
        self.setup_loggers()

    def setup_loggers(self):
        """Setup different loggers for different purposes"""

        # Main application logger
        self.app_logger = self._create_logger(
            "spygate_app", self.log_dir / "spygate.log", level=logging.INFO
        )

        # Performance logger
        self.perf_logger = self._create_logger(
            "spygate_performance", self.log_dir / "performance.log", level=logging.INFO
        )

        # Error logger
        self.error_logger = self._create_logger(
            "spygate_errors", self.log_dir / "errors.log", level=logging.ERROR
        )

        # GPU logger
        self.gpu_logger = self._create_logger(
            "spygate_gpu", self.log_dir / "gpu.log", level=logging.DEBUG
        )

    def _create_logger(self, name: str, log_file: Path, level: int = logging.INFO):
        """Create a configured logger"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers
        logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_app_event(self, message: str, level: str = "info"):
        """Log application events"""
        getattr(self.app_logger, level.lower())(message)

    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        message = (
            f"FPS: {metrics.get('fps', 0):.1f} | "
            f"Processing: {metrics.get('processing_time_ms', 0):.1f}ms | "
            f"GPU Memory: {metrics.get('gpu_memory_used_gb', 0):.1f}GB"
        )
        self.perf_logger.info(message)

    def log_error(self, error: Exception, context: str = ""):
        """Log errors with full context"""
        error_message = f"{context} | {type(error).__name__}: {str(error)}"
        self.error_logger.error(error_message)
        self.error_logger.error(f"Traceback:\n{traceback.format_exc()}")

    def log_gpu_event(self, message: str):
        """Log GPU-related events"""
        self.gpu_logger.debug(message)


class ErrorHandler:
    """Smart error handling with recovery strategies"""

    def __init__(self, logger: SpygateLogger):
        self.logger = logger
        self.error_counts = {}
        self.recovery_attempts = {}

    def handle_gpu_error(self, error: Exception) -> bool:
        """
        Handle GPU-related errors with smart recovery

        Returns:
            bool: True if recovery was successful
        """
        error_type = type(error).__name__

        self.logger.log_error(error, "GPU Error")

        if "CUDA out of memory" in str(error):
            self.logger.log_gpu_event("CUDA OOM detected - attempting memory cleanup")
            return self._recover_from_cuda_oom()

        elif "CUDA driver" in str(error):
            self.logger.log_gpu_event("CUDA driver issue detected")
            return self._recover_from_cuda_driver_error()

        return False

    def handle_ocr_error(self, error: Exception) -> bool:
        """Handle OCR-related errors"""
        self.logger.log_error(error, "OCR Error")

        if "pytesseract" in str(error):
            self.logger.log_app_event("PyTesseract error - checking configuration", "warning")
            return self._recover_from_tesseract_error()

        return False

    def handle_model_error(self, error: Exception) -> bool:
        """Handle model loading/inference errors"""
        self.logger.log_error(error, "Model Error")

        if "model" in str(error).lower():
            self.logger.log_app_event("Model error detected - attempting reload", "warning")
            return self._recover_from_model_error()

        return False

    def _recover_from_cuda_oom(self) -> bool:
        """Recover from CUDA out of memory"""
        try:
            import torch

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.log_gpu_event("GPU cache cleared")

                # Force garbage collection
                import gc

                gc.collect()

                self.logger.log_gpu_event("CUDA OOM recovery completed")
                return True

        except Exception as e:
            self.logger.log_error(e, "CUDA OOM Recovery Failed")

        return False

    def _recover_from_cuda_driver_error(self) -> bool:
        """Recover from CUDA driver issues"""
        self.logger.log_gpu_event(
            "CUDA driver recovery not implemented - manual intervention required"
        )
        return False

    def _recover_from_tesseract_error(self) -> bool:
        """Recover from Tesseract configuration issues"""
        try:
            import pytesseract

            # Try to reconfigure Tesseract path
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

            self.logger.log_app_event("Tesseract path reconfigured", "info")
            return True

        except Exception as e:
            self.logger.log_error(e, "Tesseract Recovery Failed")

        return False

    def _recover_from_model_error(self) -> bool:
        """Recover from model loading issues"""
        self.logger.log_app_event(
            "Model recovery not implemented - requires manual reload", "warning"
        )
        return False


def error_handler(retry_count: int = 3, fallback_value: Any = None):
    """
    Decorator for automatic error handling with retries

    Args:
        retry_count: Number of retry attempts
        fallback_value: Value to return if all retries fail
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if attempt < retry_count:
                        # Log retry attempt
                        logger = SpygateLogger()
                        logger.log_app_event(
                            f"Retry {attempt + 1}/{retry_count} for {func.__name__}: {str(e)}",
                            "warning",
                        )
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    else:
                        # Final attempt failed
                        logger = SpygateLogger()
                        logger.log_error(e, f"Final retry failed for {func.__name__}")

            # All retries failed
            if fallback_value is not None:
                return fallback_value
            else:
                raise last_exception

        return wrapper

    return decorator


def gpu_safe(fallback_to_cpu: bool = True):
    """
    Decorator to make functions GPU-safe with CPU fallback

    Args:
        fallback_to_cpu: Whether to attempt CPU fallback on GPU errors
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                if "cuda" in str(e).lower() or "gpu" in str(e).lower():
                    logger = SpygateLogger()
                    error_handler = ErrorHandler(logger)

                    # Attempt GPU recovery
                    if error_handler.handle_gpu_error(e):
                        logger.log_gpu_event(f"GPU recovery successful for {func.__name__}")
                        return func(*args, **kwargs)

                    elif fallback_to_cpu:
                        logger.log_app_event(f"Falling back to CPU for {func.__name__}", "warning")
                        # This would require function-specific CPU fallback logic
                        raise NotImplementedError("CPU fallback not implemented for this function")

                raise e

        return wrapper

    return decorator


# Example usage functions
@error_handler(retry_count=3, fallback_value=None)
def example_ocr_function(image_path: str):
    """Example OCR function with error handling"""
    import pytesseract
    from PIL import Image

    image = Image.open(image_path)
    result = pytesseract.image_to_string(image)
    return result


@gpu_safe(fallback_to_cpu=True)
def example_gpu_function():
    """Example GPU function with safety wrapper"""
    import torch

    # This would be your actual GPU processing
    device = torch.device("cuda")
    tensor = torch.ones(1000, 1000).to(device)
    result = tensor * 2
    return result.cpu().numpy()


if __name__ == "__main__":
    # Demo the logging system
    logger = SpygateLogger()
    error_handler = ErrorHandler(logger)

    logger.log_app_event("SpygateAI started", "info")
    logger.log_performance({"fps": 45.2, "processing_time_ms": 22.1, "gpu_memory_used_gb": 2.3})

    # Simulate an error
    try:
        raise Exception("Test error for demonstration")
    except Exception as e:
        logger.log_error(e, "Demo Error")

    print("✅ Error handling and logging system demo completed")
    print("📁 Check the 'logs' directory for log files")
