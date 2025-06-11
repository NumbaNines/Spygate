"""
SpygateAI Performance Monitor
Real-time performance tracking for video analysis
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import psutil
import torch


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    fps: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_utilization: float = 0.0
    ram_usage: float = 0.0
    ocr_success_rate: float = 0.0
    processing_time: float = 0.0
    frame_count: int = 0


class PerformanceMonitor:
    """Real-time performance monitoring for SpygateAI"""

    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor

        Args:
            window_size: Number of recent measurements to keep for averaging
        """
        self.window_size = window_size
        self.reset_metrics()

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.cuda.current_device()

    def reset_metrics(self):
        """Reset all performance tracking"""
        self.fps_history = deque(maxlen=self.window_size)
        self.processing_times = deque(maxlen=self.window_size)
        self.ocr_results = deque(maxlen=self.window_size)
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

    def start_frame(self):
        """Mark the start of frame processing"""
        self.frame_start_time = time.time()

    def end_frame(self, ocr_success: bool = False):
        """
        Mark the end of frame processing

        Args:
            ocr_success: Whether OCR was successful for this frame
        """
        current_time = time.time()

        # Calculate processing time
        if hasattr(self, "frame_start_time"):
            processing_time = current_time - self.frame_start_time
            self.processing_times.append(processing_time)

        # Calculate FPS
        time_since_last = current_time - self.last_frame_time
        if time_since_last > 0:
            fps = 1.0 / time_since_last
            self.fps_history.append(fps)

        self.last_frame_time = current_time
        self.frame_count += 1
        self.ocr_results.append(ocr_success)

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        metrics = PerformanceMetrics()

        # FPS calculation
        if self.fps_history:
            metrics.fps = sum(self.fps_history) / len(self.fps_history)

        # Processing time
        if self.processing_times:
            metrics.processing_time = sum(self.processing_times) / len(self.processing_times)

        # OCR success rate
        if self.ocr_results:
            metrics.ocr_success_rate = sum(self.ocr_results) / len(self.ocr_results) * 100

        # System metrics
        metrics.cpu_utilization = psutil.cpu_percent()
        metrics.ram_usage = psutil.virtual_memory().percent
        metrics.frame_count = self.frame_count

        # GPU metrics
        if self.gpu_available:
            try:
                # Get GPU memory info
                memory_info = torch.cuda.memory_stats(self.device)
                allocated = memory_info.get("allocated_bytes.all.current", 0)
                reserved = memory_info.get("reserved_bytes.all.current", 0)

                # Convert to GB
                metrics.gpu_memory_used = allocated / (1024**3)
                total_memory = torch.cuda.get_device_properties(self.device).total_memory
                metrics.gpu_memory_total = total_memory / (1024**3)

                # GPU utilization (approximated)
                metrics.gpu_utilization = (
                    (allocated / total_memory) * 100 if total_memory > 0 else 0
                )

            except Exception:
                # GPU metrics unavailable
                pass

        return metrics

    def print_status(self, detailed: bool = False):
        """Print current performance status"""
        metrics = self.get_current_metrics()

        print(f"\n🔥 SpygateAI Performance Monitor")
        print(f"=" * 45)
        print(f"📺 FPS: {metrics.fps:.1f}")
        print(f"⏱️  Processing Time: {metrics.processing_time*1000:.1f}ms")
        print(f"🎯 OCR Success Rate: {metrics.ocr_success_rate:.1f}%")
        print(f"📊 Frames Processed: {metrics.frame_count}")

        if detailed:
            print(f"\n💻 System Resources:")
            print(f"   CPU: {metrics.cpu_utilization:.1f}%")
            print(f"   RAM: {metrics.ram_usage:.1f}%")

            if self.gpu_available:
                print(f"🎮 GPU (RTX 4070 SUPER):")
                print(
                    f"   Memory: {metrics.gpu_memory_used:.1f}GB / {metrics.gpu_memory_total:.1f}GB"
                )
                print(f"   Utilization: {metrics.gpu_utilization:.1f}%")
            else:
                print(f"🎮 GPU: Not Available")

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary as dictionary"""
        metrics = self.get_current_metrics()
        runtime = time.time() - self.start_time

        return {
            "runtime_seconds": runtime,
            "average_fps": metrics.fps,
            "total_frames": metrics.frame_count,
            "average_processing_time_ms": metrics.processing_time * 1000,
            "ocr_success_rate_percent": metrics.ocr_success_rate,
            "gpu_available": self.gpu_available,
            "gpu_memory_used_gb": metrics.gpu_memory_used,
            "cpu_utilization_percent": metrics.cpu_utilization,
            "ram_usage_percent": metrics.ram_usage,
            "performance_rating": self._get_performance_rating(metrics),
        }

    def _get_performance_rating(self, metrics: PerformanceMetrics) -> str:
        """Get overall performance rating"""
        if metrics.fps >= 60:
            return "🟢 Excellent"
        elif metrics.fps >= 30:
            return "🟡 Good"
        elif metrics.fps >= 15:
            return "🟠 Fair"
        else:
            return "🔴 Poor"


# Example usage function
def demo_performance_monitor():
    """Demonstrate the performance monitor"""
    monitor = PerformanceMonitor()

    print("🚀 Starting SpygateAI Performance Monitor Demo")

    # Simulate processing frames
    for i in range(10):
        monitor.start_frame()

        # Simulate processing time
        time.sleep(0.05)  # 50ms processing time

        # Simulate OCR success (80% success rate)
        ocr_success = i % 5 != 0

        monitor.end_frame(ocr_success)

        if i % 3 == 0:
            monitor.print_status()

    print("\n📋 Final Performance Summary:")
    summary = monitor.get_performance_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_performance_monitor()
