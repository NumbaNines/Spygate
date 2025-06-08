"""
Prometheus metrics exporter.

This module provides Prometheus metrics export functionality for system monitoring.
"""

import threading
import time
from typing import Dict, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram
from prometheus_client.core import CollectorRegistry

from ..core.performance_monitor import PerformanceMonitor


class PrometheusExporter:
    """Exports performance metrics in Prometheus format."""
    
    def __init__(
        self,
        monitor: PerformanceMonitor,
        port: int = 8000,
        interval: float = 1.0,
        prefix: str = "spygate_",
    ):
        """Initialize Prometheus exporter.
        
        Args:
            monitor: Performance monitor instance
            port: HTTP server port
            interval: Metrics update interval in seconds
            prefix: Metric name prefix
        """
        self.monitor = monitor
        self.interval = interval
        self.prefix = prefix
        self.registry = CollectorRegistry()
        
        # Create metrics
        self.fps = Gauge(
            f"{prefix}fps",
            "Current frames per second",
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            f"{prefix}memory_usage_mb",
            "Current memory usage in MB",
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            f"{prefix}gpu_memory_mb",
            "Current GPU memory usage in MB",
            registry=self.registry
        )
        
        self.quality_level = Gauge(
            f"{prefix}quality_level",
            "Current quality level (0.0-1.0)",
            registry=self.registry
        )
        
        self.dropped_frames = Counter(
            f"{prefix}dropped_frames_total",
            "Total number of dropped frames",
            registry=self.registry
        )
        
        self.optimization_events = Counter(
            f"{prefix}optimization_events_total",
            "Total number of quality optimization events",
            registry=self.registry
        )
        
        self.frames_processed = Counter(
            f"{prefix}frames_processed_total",
            "Total number of frames processed",
            registry=self.registry
        )
        
        self.processing_time = Histogram(
            f"{prefix}processing_time_seconds",
            "Frame processing time in seconds",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry
        )
        
        # Start server
        start_http_server(port, registry=self.registry)
        
        # Start update thread
        self._stop = False
        self._thread = threading.Thread(target=self._update_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def _update_loop(self):
        """Update metrics periodically."""
        while not self._stop:
            try:
                # Get current stats
                stats = self.monitor.get_performance_stats()
                
                # Update gauges
                self.fps.set(stats["fps"])
                self.memory_usage.set(stats["memory_usage_mb"])
                self.gpu_memory.set(stats.get("gpu_memory_mb", 0))
                self.quality_level.set(stats["quality_level"])
                
                # Update counters
                self.dropped_frames.inc(stats["dropped_frames"])
                self.optimization_events.inc(stats["optimization_events"])
                self.frames_processed.inc(stats["frames_processed"])
                
                # Update histogram
                for time_value in self.monitor.get_metrics_report()["processing_times"]:
                    self.processing_time.observe(time_value)
                    
            except Exception as e:
                import logging
                logging.error(f"Error updating Prometheus metrics: {e}")
                
            time.sleep(self.interval)
    
    def stop(self):
        """Stop the metrics exporter."""
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            
    def __del__(self):
        """Cleanup on deletion."""
        self.stop() 