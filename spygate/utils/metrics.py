"""
Metrics collection and reporting module.

This module provides functionality for collecting, storing, and reporting
various performance and system metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Union
from collections import deque, defaultdict
import threading
import json
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and manages system metrics."""

    def __init__(
        self,
        max_history: int = 3600,  # 1 hour at 1 sample/second
        storage_path: Optional[str] = None,
    ):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of samples to keep in memory
            storage_path: Path to store metrics data (optional)
        """
        self.max_history = max_history
        self.storage_path = storage_path
        
        # Initialize storage
        self._gauges = defaultdict(lambda: deque(maxlen=max_history))
        self._counters = defaultdict(int)
        self._events = defaultdict(lambda: deque(maxlen=max_history))
        self._histograms = defaultdict(lambda: deque(maxlen=max_history))
        
        # Track timestamps
        self._timestamps = defaultdict(lambda: deque(maxlen=max_history))
        
        # Threading
        self._lock = threading.Lock()
        
        # Auto-save if storage path provided
        if storage_path:
            self._setup_storage()
            self._start_auto_save()
            
        logger.info(f"Initialized MetricsCollector with {max_history} samples history")

    def _setup_storage(self):
        """Setup storage directory."""
        storage_dir = Path(self.storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (storage_dir / "gauges").mkdir(exist_ok=True)
        (storage_dir / "events").mkdir(exist_ok=True)
        (storage_dir / "histograms").mkdir(exist_ok=True)
        
        logger.info(f"Setup metrics storage at {storage_dir}")

    def _start_auto_save(self):
        """Start automatic saving of metrics."""
        def auto_save():
            while True:
                try:
                    time.sleep(60)  # Save every minute
                    self.save_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics auto-save: {e}")
                    time.sleep(300)  # Back off on error
                    
        self._save_thread = threading.Thread(
            target=auto_save,
            daemon=True
        )
        self._save_thread.start()

    def record_gauge(self, name: str, value: float):
        """Record a gauge value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        with self._lock:
            timestamp = time.time()
            self._gauges[name].append(value)
            self._timestamps[f"gauge_{name}"].append(timestamp)

    def record_counter(self, name: str, value: int = 1):
        """Increment a counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self._lock:
            self._counters[name] += value

    def record_event(
        self,
        name: str,
        attributes: Optional[Dict] = None,
    ):
        """Record an event.
        
        Args:
            name: Event name
            attributes: Event attributes
        """
        with self._lock:
            timestamp = time.time()
            event = {
                "timestamp": timestamp,
                "attributes": attributes or {},
            }
            self._events[name].append(event)
            self._timestamps[f"event_{name}"].append(timestamp)

    def record_histogram(
        self,
        name: str,
        value: float,
        bucket_size: Optional[float] = None,
    ):
        """Record a histogram value.
        
        Args:
            name: Histogram name
            value: Value to record
            bucket_size: Size of histogram buckets
        """
        with self._lock:
            timestamp = time.time()
            if bucket_size:
                # Round to bucket
                value = round(value / bucket_size) * bucket_size
            self._histograms[name].append(value)
            self._timestamps[f"histogram_{name}"].append(timestamp)

    def get_gauge_stats(
        self,
        name: str,
        window: Optional[float] = None,
    ) -> Dict[str, float]:
        """Get statistics for a gauge.
        
        Args:
            name: Gauge name
            window: Time window in seconds
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            values = list(self._gauges[name])
            timestamps = list(self._timestamps[f"gauge_{name}"])
            
            if not values:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                }
                
            if window:
                # Filter by time window
                cutoff = time.time() - window
                values = [
                    v for v, t in zip(values, timestamps)
                    if t >= cutoff
                ]
                
            return {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    def get_counter_value(self, name: str) -> int:
        """Get current counter value.
        
        Args:
            name: Counter name
            
        Returns:
            Counter value
        """
        with self._lock:
            return self._counters[name]

    def get_events(
        self,
        name: str,
        window: Optional[float] = None,
    ) -> List[Dict]:
        """Get events in time window.
        
        Args:
            name: Event name
            window: Time window in seconds
            
        Returns:
            List of events
        """
        with self._lock:
            events = list(self._events[name])
            if window:
                cutoff = time.time() - window
                events = [
                    e for e in events
                    if e["timestamp"] >= cutoff
                ]
            return events

    def get_histogram_stats(
        self,
        name: str,
        window: Optional[float] = None,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics.
        
        Args:
            name: Histogram name
            window: Time window in seconds
            percentiles: Percentiles to calculate
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            values = list(self._histograms[name])
            timestamps = list(self._timestamps[f"histogram_{name}"])
            
            if not values:
                return {
                    "count": 0,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                }
                
            if window:
                # Filter by time window
                cutoff = time.time() - window
                values = [
                    v for v, t in zip(values, timestamps)
                    if t >= cutoff
                ]
                
            stats = {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            
            if percentiles:
                for p in percentiles:
                    stats[f"p{p}"] = float(np.percentile(values, p))
                    
            return stats

    def save_metrics(self):
        """Save metrics to storage."""
        if not self.storage_path:
            return
            
        with self._lock:
            timestamp = int(time.time())
            storage_dir = Path(self.storage_path)
            
            # Save gauges
            for name, values in self._gauges.items():
                if not values:
                    continue
                    
                gauge_file = storage_dir / "gauges" / f"{name}.json"
                data = {
                    "name": name,
                    "timestamp": timestamp,
                    "values": list(values),
                    "timestamps": list(self._timestamps[f"gauge_{name}"]),
                }
                with open(gauge_file, "w") as f:
                    json.dump(data, f)
                    
            # Save events
            for name, events in self._events.items():
                if not events:
                    continue
                    
                event_file = storage_dir / "events" / f"{name}.json"
                data = {
                    "name": name,
                    "timestamp": timestamp,
                    "events": list(events),
                }
                with open(event_file, "w") as f:
                    json.dump(data, f)
                    
            # Save histograms
            for name, values in self._histograms.items():
                if not values:
                    continue
                    
                hist_file = storage_dir / "histograms" / f"{name}.json"
                data = {
                    "name": name,
                    "timestamp": timestamp,
                    "values": list(values),
                    "timestamps": list(self._timestamps[f"histogram_{name}"]),
                }
                with open(hist_file, "w") as f:
                    json.dump(data, f)
                    
            logger.info("Saved metrics to storage")

    def load_metrics(self):
        """Load metrics from storage."""
        if not self.storage_path:
            return
            
        with self._lock:
            storage_dir = Path(self.storage_path)
            
            # Load gauges
            gauge_dir = storage_dir / "gauges"
            for gauge_file in gauge_dir.glob("*.json"):
                try:
                    with open(gauge_file) as f:
                        data = json.load(f)
                        name = data["name"]
                        self._gauges[name] = deque(
                            data["values"],
                            maxlen=self.max_history
                        )
                        self._timestamps[f"gauge_{name}"] = deque(
                            data["timestamps"],
                            maxlen=self.max_history
                        )
                except Exception as e:
                    logger.error(f"Error loading gauge {gauge_file}: {e}")
                    
            # Load events
            event_dir = storage_dir / "events"
            for event_file in event_dir.glob("*.json"):
                try:
                    with open(event_file) as f:
                        data = json.load(f)
                        name = data["name"]
                        self._events[name] = deque(
                            data["events"],
                            maxlen=self.max_history
                        )
                except Exception as e:
                    logger.error(f"Error loading events {event_file}: {e}")
                    
            # Load histograms
            hist_dir = storage_dir / "histograms"
            for hist_file in hist_dir.glob("*.json"):
                try:
                    with open(hist_file) as f:
                        data = json.load(f)
                        name = data["name"]
                        self._histograms[name] = deque(
                            data["values"],
                            maxlen=self.max_history
                        )
                        self._timestamps[f"histogram_{name}"] = deque(
                            data["timestamps"],
                            maxlen=self.max_history
                        )
                except Exception as e:
                    logger.error(f"Error loading histogram {hist_file}: {e}")
                    
            logger.info("Loaded metrics from storage")

    def cleanup_old_data(self):
        """Remove old data points."""
        with self._lock:
            # Clear empty metrics
            for name in list(self._gauges.keys()):
                if not self._gauges[name]:
                    del self._gauges[name]
                    del self._timestamps[f"gauge_{name}"]
                    
            for name in list(self._events.keys()):
                if not self._events[name]:
                    del self._events[name]
                    
            for name in list(self._histograms.keys()):
                if not self._histograms[name]:
                    del self._histograms[name]
                    del self._timestamps[f"histogram_{name}"]
                    
            # Remove old counters
            for name in list(self._counters.keys()):
                if self._counters[name] == 0:
                    del self._counters[name]

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._gauges.clear()
            self._counters.clear()
            self._events.clear()
            self._histograms.clear()
            self._timestamps.clear()
            logger.info("Reset all metrics") 