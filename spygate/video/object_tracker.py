"""
Object tracking module for video analysis.

This module provides object tracking functionality with support for multiple
tracking algorithms, parallel processing, and memory-efficient operation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

import cv2
import numpy as np
from numba import jit, cuda

from ..core.hardware import HardwareDetector
from ..core.optimizer import TierOptimizer
from ..utils.tracking_hardware import TrackingHardwareManager, TrackingMode
from .algorithm_selector import AlgorithmSelector, SceneComplexity

logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for object tracking."""

    max_objects: int = 30
    min_detection_confidence: float = 0.5
    max_prediction_age: int = 30
    batch_size: int = 16  # Increased for better GPU utilization
    max_thread_workers: int = 4
    enable_parallel: bool = True
    max_frames_in_memory: int = 100
    prediction_buffer_size: int = 30
    cleanup_interval: int = 100
    max_memory_usage: float = 2048.0  # Maximum memory usage in MB
    enable_gpu: bool = True  # Enable GPU acceleration if available
    enable_prefetch: bool = True  # Enable frame prefetching
    prefetch_size: int = 4  # Number of frames to prefetch
    quality_step: float = 0.1  # Quality adjustment step
    min_quality: float = 0.5  # Minimum processing quality
    max_quality: float = 1.0  # Maximum processing quality
    enable_memory_tracking: bool = True  # Track memory usage
    memory_warning_threshold: float = 0.9  # 90% of max memory
    enable_adaptive_quality: bool = True  # Adjust quality based on resources
    enable_cache_compression: bool = True  # Compress cached data
    compression_level: int = 1  # Compression level (1-9)
    max_cached_tracks: int = 1000  # Maximum tracks to keep in memory
    enable_motion_prediction: bool = True  # Use motion prediction
    prediction_horizon: int = 10  # Frames to predict ahead
    enable_track_pruning: bool = True  # Remove stale tracks
    track_pruning_threshold: int = 30  # Frames before pruning
    enable_parallel_tracking: bool = True  # Track objects in parallel
    enable_gpu_acceleration: bool = True  # Use GPU for tracking
    max_gpu_memory: float = 1024.0  # Maximum GPU memory usage in MB
    enable_track_fusion: bool = True  # Merge similar tracks
    fusion_iou_threshold: float = 0.5  # IOU threshold for fusion
    enable_scene_adaptation: bool = True  # Adapt to scene complexity


@jit(nopython=True)
def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


@cuda.jit
def calculate_iou_batch_gpu(bboxes1, bboxes2, results):
    """CUDA kernel for batch IoU calculation."""
    idx = cuda.grid(1)
    if idx < bboxes1.shape[0]:
        x1 = max(bboxes1[idx, 0], bboxes2[idx, 0])
        y1 = max(bboxes1[idx, 1], bboxes2[idx, 1])
        x2 = min(bboxes1[idx, 2], bboxes2[idx, 2])
        y2 = min(bboxes1[idx, 3], bboxes2[idx, 3])
        
        if x2 < x1 or y2 < y1:
            results[idx] = 0.0
            return
            
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (bboxes1[idx, 2] - bboxes1[idx, 0]) * (bboxes1[idx, 3] - bboxes1[idx, 1])
        area2 = (bboxes2[idx, 2] - bboxes2[idx, 0]) * (bboxes2[idx, 3] - bboxes2[idx, 1])
        
        union = area1 + area2 - intersection
        
        results[idx] = intersection / union if union > 0 else 0.0


class ObjectTracker:
    """Tracks objects in video frames with performance optimization."""

    def __init__(self, config: TrackingConfig, hardware: HardwareDetector):
        self.config = config
        self.optimizer = TierOptimizer(hardware)
        self.hardware_manager = TrackingHardwareManager()
        self.algorithm_selector = AlgorithmSelector()
        
        # Initialize tracking state
        self.tracks = {}  # track_id -> track_data
        self.track_history = {}  # track_id -> deque(positions)
        self.track_predictions = {}  # track_id -> predicted_positions
        self.track_ages = {}  # track_id -> frames_since_update
        self.next_track_id = 0
        self.frame_count = 0
        self.last_cleanup = 0
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.memory_usage = 0.0
        self.current_quality = config.max_quality
        
        # Threading and synchronization
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=config.max_thread_workers)
        
        # GPU resources
        if self.config.enable_gpu_acceleration and cuda.is_available():
            cuda.select_device(0)
            self.stream = cuda.stream()
            self.gpu_memory_pool = {}  # buffer_key -> gpu_buffer
        else:
            self.stream = None
            self.gpu_memory_pool = None
            
        logger.info(f"Initialized ObjectTracker with {hardware.tier.name} tier")

    def _track_memory(self, track_id: int, data_size: float):
        """Track memory usage of tracking data."""
        if not self.config.enable_memory_tracking:
            return
            
        self.memory_usage += data_size
        
        # Check memory threshold
        if self.memory_usage > self.config.max_memory_usage * self.config.memory_warning_threshold:
            self._reduce_memory_usage()

    def _reduce_memory_usage(self):
        """Reduce memory usage when threshold is reached."""
        if not self.config.enable_memory_tracking:
            return
            
        logger.info(f"Memory usage ({self.memory_usage:.2f}MB) exceeded warning threshold")
        
        # Remove old tracks
        with self.lock:
            old_tracks = sorted(
                self.track_ages.items(),
                key=lambda x: x[1],
                reverse=True
            )[:len(self.track_ages)//4]  # Remove oldest 25%
            
            for track_id, _ in old_tracks:
                self._remove_track(track_id)
                
        # Force garbage collection
        gc.collect()
        
        # Adjust quality if needed
        if self.config.enable_adaptive_quality:
            self.current_quality = max(
                self.config.min_quality,
                self.current_quality - self.config.quality_step
            )
            logger.info(f"Reduced tracking quality to {self.current_quality:.2f}")

    def _remove_track(self, track_id: int):
        """Remove a track and free its resources."""
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.track_predictions:
            del self.track_predictions[track_id]
        if track_id in self.track_ages:
            del self.track_ages[track_id]
            
        # Free GPU resources
        if self.gpu_memory_pool and track_id in self.gpu_memory_pool:
            self.gpu_memory_pool[track_id].free()
            del self.gpu_memory_pool[track_id]

    @jit(nopython=True)
    def _predict_motion(self, history: np.ndarray) -> np.ndarray:
        """Predict future motion using trajectory analysis."""
        if len(history) < 2:
            return history[-1] if len(history) > 0 else np.zeros(4)
            
        # Calculate velocity using recent positions
        velocities = np.diff(history, axis=0)
        avg_velocity = np.mean(velocities[-min(len(velocities), 5):], axis=0)
        
        # Predict future position
        return history[-1] + avg_velocity

    def _update_track_predictions(self):
        """Update motion predictions for all tracks."""
        if not self.config.enable_motion_prediction:
            return
            
        for track_id in self.tracks:
            history = np.array(list(self.track_history[track_id]))
            if len(history) >= 2:
                predictions = []
                current = history[-1]
                
                # Predict future positions
                for _ in range(self.config.prediction_horizon):
                    current = self._predict_motion(np.vstack([history, current]))
                    predictions.append(current)
                    
                self.track_predictions[track_id] = np.array(predictions)

    def _select_algorithm(self, frame: np.ndarray) -> str:
        """Select tracking algorithm based on scene complexity."""
        if not self.config.enable_scene_adaptation:
            return "default"
            
        # Analyze scene complexity
        motion = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        avg_motion = np.mean(np.abs(motion))
        if avg_motion > 5.0:
            return "complex"
        elif avg_motion > 2.0:
            return "medium"
        else:
            return "simple"

    def _fuse_tracks(self, detections: List[np.ndarray]):
        """Merge similar tracks to prevent duplicates."""
        if not self.config.enable_track_fusion:
            return
            
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[0] + box1[2], box2[0] + box2[2])
            y2 = min(box1[1] + box1[3], box2[1] + box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]
            
            return intersection / (area1 + area2 - intersection)
            
        # Find tracks to merge
        to_merge = []
        track_ids = list(self.tracks.keys())
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track1 = self.tracks[track_ids[i]]
                track2 = self.tracks[track_ids[j]]
                
                if calculate_iou(track1, track2) > self.config.fusion_iou_threshold:
                    to_merge.append((track_ids[i], track_ids[j]))
                    
        # Merge tracks
        for track1_id, track2_id in to_merge:
            if track1_id in self.tracks and track2_id in self.tracks:
                # Keep track with more history
                if len(self.track_history[track1_id]) > len(self.track_history[track2_id]):
                    self._remove_track(track2_id)
                else:
                    self._remove_track(track1_id)

    def update(self, frame: np.ndarray, detections: List[np.ndarray]):
        """Update tracking with new detections."""
        start_time = time.time()
        
        # Select tracking algorithm
        algorithm = self._select_algorithm(frame)
        
        # Update frame count
        self.frame_count += 1
        
        # Process detections
        if self.config.enable_parallel_tracking and len(detections) > 1:
            # Process detections in parallel
            futures = []
            for detection in detections:
                future = self.executor.submit(
                    self._process_detection,
                    detection,
                    algorithm
                )
                futures.append(future)
                
            # Collect results
            results = [f.result() for f in futures]
        else:
            # Process sequentially
            results = [
                self._process_detection(d, algorithm)
                for d in detections
            ]
            
        # Update tracks
        with self.lock:
            for track_id, detection in results:
                if track_id is None:
                    # New track
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    self.tracks[track_id] = detection
                    self.track_history[track_id] = deque(
                        maxlen=self.config.prediction_buffer_size
                    )
                    self.track_ages[track_id] = 0
                else:
                    # Update existing track
                    self.tracks[track_id] = detection
                    self.track_ages[track_id] = 0
                    
                # Update history
                self.track_history[track_id].append(detection)
                
        # Update predictions
        self._update_track_predictions()
        
        # Fuse similar tracks
        self._fuse_tracks(detections)
        
        # Age and prune tracks
        self._age_tracks()
        
        # Cleanup if needed
        if (self.frame_count - self.last_cleanup) >= self.config.cleanup_interval:
            self._cleanup()
            
        # Update performance metrics
        end_time = time.time()
        self.processing_times.append(end_time - start_time)
        
        # Return active tracks
        return self.get_active_tracks()

    def _process_detection(
        self,
        detection: np.ndarray,
        algorithm: str
    ) -> Tuple[Optional[int], np.ndarray]:
        """Process a single detection."""
        best_match = None
        best_score = float('inf')
        
        # Find best matching track
        for track_id, track in self.tracks.items():
            if algorithm == "simple":
                score = np.linalg.norm(detection[:2] - track[:2])
            elif algorithm == "medium":
                score = np.linalg.norm(detection - track)
            else:  # complex
                # Use IoU-based matching
                iou = self._calculate_iou(detection, track)
                score = 1.0 - iou
                
            if score < best_score:
                best_score = score
                best_match = track_id
                
        # Return match or None for new track
        return (best_match, detection) if best_score < 0.5 else (None, detection)

    def _age_tracks(self):
        """Age tracks and remove old ones."""
        if not self.config.enable_track_pruning:
            return
            
        with self.lock:
            # Age all tracks
            for track_id in list(self.track_ages.keys()):
                self.track_ages[track_id] += 1
                
                # Remove old tracks
                if self.track_ages[track_id] > self.config.track_pruning_threshold:
                    self._remove_track(track_id)

    def _cleanup(self):
        """Perform periodic cleanup of resources."""
        with self.lock:
            # Remove old tracks
            while len(self.tracks) > self.config.max_cached_tracks:
                oldest_track = max(
                    self.track_ages.items(),
                    key=lambda x: x[1]
                )[0]
                self._remove_track(oldest_track)
                
            # Reset cleanup counter
            self.last_cleanup = self.frame_count
            
            # Force garbage collection
            if self.config.enable_memory_tracking:
                gc.collect()

    def get_active_tracks(self) -> Dict[int, np.ndarray]:
        """Get currently active tracks."""
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if self.track_ages[track_id] <= self.config.max_prediction_age
        }

    def get_track_predictions(self, track_id: int) -> Optional[np.ndarray]:
        """Get motion predictions for a track."""
        return self.track_predictions.get(track_id)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.processing_times:
            return {
                "avg_processing_time": 0.0,
                "fps": 0.0,
                "memory_usage_mb": self.memory_usage,
                "quality_level": self.current_quality,
                "active_tracks": len(self.tracks)
            }
            
        avg_time = np.mean(self.processing_times)
        return {
            "avg_processing_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "memory_usage_mb": self.memory_usage,
            "quality_level": self.current_quality,
            "active_tracks": len(self.tracks),
            "memory_threshold": self.config.max_memory_usage * self.config.memory_warning_threshold,
            "gpu_memory_mb": self.hardware_manager.get_gpu_memory_usage() if self.stream else 0.0
        }

    def __del__(self):
        """Cleanup resources on deletion."""
        self.executor.shutdown(wait=True)
        if self.stream:
            self.stream.synchronize()
        if self.gpu_memory_pool:
            for buffer in self.gpu_memory_pool.values():
                buffer.free()
        self.tracks.clear()
        self.track_history.clear()
        self.track_predictions.clear()
        self.track_ages.clear()
        gc.collect()

class MultiObjectTracker:
    """
    MultiObjectTracker manages multiple ObjectTracker instances for multi-object tracking.
    Each object is tracked with a unique ID and supports occlusion handling and identity maintenance.
    """
    def __init__(self, tracker_type='KCF', max_lost_frames=30, iou_threshold=0.3, 
                 reidentify_threshold=0.7, max_prediction_frames=10):
        self.tracker_type = tracker_type
        self.trackers = {}  # id -> ObjectTracker
        self.lost = {}      # id -> {'hist': ..., 'bbox': ..., 'frames_lost': int, 'last_velocity': (dx,dy)}
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.reidentify_threshold = reidentify_threshold
        self.max_prediction_frames = max_prediction_frames
        self.frame_count = 0
        self.occlusion_pairs = set()  # Set of (id1, id2) pairs currently in occlusion

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to x1,y1,x2,y2 format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def _detect_occlusions(self):
        """Detect objects that are occluding each other based on IoU."""
        self.occlusion_pairs.clear()
        bboxes = self.get_bboxes()
        for id1 in bboxes:
            for id2 in bboxes:
                if id1 < id2:  # Check each pair only once
                    iou = self._calculate_iou(bboxes[id1], bboxes[id2])
                    if iou > self.iou_threshold:
                        self.occlusion_pairs.add((id1, id2))

    def _predict_bbox(self, obj_id):
        """Predict object's next position based on its velocity."""
        if obj_id not in self.lost:
            return None
        
        lost_info = self.lost[obj_id]
        if 'last_velocity' not in lost_info or lost_info['frames_lost'] > self.max_prediction_frames:
            return None
            
        dx, dy = lost_info['last_velocity']
        x, y, w, h = lost_info['bbox']
        predicted_bbox = (x + dx, y + dy, w, h)
        return predicted_bbox

    def _calculate_velocity(self, tracker):
        """Calculate object's velocity from its last two positions."""
        if len(tracker.history) < 2:
            return None
        
        x1, y1, w1, h1 = tracker.history[-2]
        x2, y2, w2, h2 = tracker.history[-1]
        dx = x2 - x1
        dy = y2 - y1
        return (dx, dy)

    def add(self, obj_id, frame, bbox):
        """Add a new object to track with a given ID and bounding box."""
        tracker = ObjectTracker(tracker_type=self.tracker_type)
        ok = tracker.init(frame, bbox)
        if ok:
            self.trackers[obj_id] = tracker
            # Remove from lost if it was there
            if obj_id in self.lost:
                del self.lost[obj_id]
        return ok

    def mark_lost(self, obj_id, frame):
        """Mark a tracker as lost and store its last known state."""
        tracker = self.trackers.get(obj_id)
        if tracker and tracker.bbox is not None:
            velocity = self._calculate_velocity(tracker)
            hist = tracker.get_histogram(frame)
            self.lost[obj_id] = {
                'hist': hist, 
                'bbox': tracker.bbox,
                'frames_lost': 0,
                'last_velocity': velocity
            }
            del self.trackers[obj_id]

    def try_reidentify(self, frame, bbox, threshold=None):
        """Try to re-identify a lost object using appearance and motion cues."""
        if threshold is None:
            threshold = self.reidentify_threshold
            
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
            
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        
        best_id = None
        best_score = 0
        
        for obj_id, lost_info in list(self.lost.items()):
            # Skip if object has been lost too long
            if lost_info['frames_lost'] > self.max_lost_frames:
                del self.lost[obj_id]
                continue
                
            lost_hist = lost_info['hist']
            if lost_hist is None:
                continue
                
            # Calculate appearance similarity
            appearance_score = cv2.compareHist(hist, lost_hist, cv2.HISTCMP_CORREL)
            
            # Calculate position similarity using predicted position
            position_score = 0.0
            predicted_bbox = self._predict_bbox(obj_id)
            if predicted_bbox is not None:
                iou = self._calculate_iou(bbox, predicted_bbox)
                position_score = iou
            
            # Combine scores (weighted average)
            final_score = 0.7 * appearance_score + 0.3 * position_score
            
            if final_score > best_score and final_score > threshold:
                best_score = final_score
                best_id = obj_id
                
        return best_id

    def update(self, frame):
        """Update all trackers with the new frame, handling occlusions and lost objects."""
        self.frame_count += 1
        
        # Detect occlusions first
        self._detect_occlusions()
        
        results = {}
        lost_ids = []
        
        # Update trackers and handle lost objects
        for obj_id, tracker in list(self.trackers.items()):
            ok, bbox = tracker.update(frame)
            results[obj_id] = (ok, bbox)
            
            if not ok:
                lost_ids.append(obj_id)
            else:
                # Update velocity for active trackers
                velocity = self._calculate_velocity(tracker)
                if velocity is not None:
                    tracker.last_velocity = velocity
        
        # Handle lost objects
        for obj_id in lost_ids:
            self.mark_lost(obj_id, frame)
        
        # Update lost objects
        for obj_id in list(self.lost.keys()):
            lost_info = self.lost[obj_id]
            lost_info['frames_lost'] += 1
            
            # Try to predict position for visualization
            predicted_bbox = self._predict_bbox(obj_id)
            if predicted_bbox is not None:
                results[obj_id] = (False, predicted_bbox)
            
            # Remove if lost for too long
            if lost_info['frames_lost'] > self.max_lost_frames:
                del self.lost[obj_id]
        
        return results

    def remove(self, obj_id):
        """Remove a tracker by its ID."""
        if obj_id in self.trackers:
            del self.trackers[obj_id]

    def get_bboxes(self):
        """Get current bounding boxes for all tracked objects."""
        return {obj_id: tracker.bbox for obj_id, tracker in self.trackers.items() if tracker.initialized}

    def get_paths(self):
        """Get the path history for all tracked objects."""
        return {obj_id: tracker.get_path() for obj_id, tracker in self.trackers.items() if tracker.initialized}

    def _get_color(self, obj_id):
        """Generate a unique color for each object ID."""
        np.random.seed(hash(obj_id) % 2**32)
        color = tuple(int(x) for x in np.random.randint(64, 255, 3))
        return color

    def draw_bboxes(self, frame, thickness=2):
        """Draw bounding boxes for all tracked and predicted objects."""
        output = frame.copy()
        
        # Draw active trackers
        for obj_id, tracker in self.trackers.items():
            if tracker.initialized and tracker.bbox is not None:
                x, y, w, h = [int(v) for v in tracker.bbox]
                color = self._get_color(obj_id)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(output, str(obj_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw occlusion indicator if object is in occlusion
                for pair in self.occlusion_pairs:
                    if obj_id in pair:
                        cv2.putText(output, "Occluded", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        break
        
        # Draw predicted positions for lost objects
        for obj_id in self.lost:
            predicted_bbox = self._predict_bbox(obj_id)
            if predicted_bbox is not None:
                x, y, w, h = [int(v) for v in predicted_bbox]
                color = self._get_color(obj_id)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
                cv2.putText(output, f"{obj_id}?", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        return output

    def draw_paths(self, frame, thickness=2, max_trail_length=30):
        """Draw path trails for all tracked objects with length limitation."""
        output = frame.copy()
        
        for obj_id, tracker in self.trackers.items():
            path = tracker.get_path()
            color = self._get_color(obj_id)
            
            if len(path) > 1:
                # Limit trail length
                if len(path) > max_trail_length:
                    path = path[-max_trail_length:]
                
                # Draw path with fading effect
                for i in range(1, len(path)):
                    x1, y1, w1, h1 = [int(v) for v in path[i-1]]
                    x2, y2, w2, h2 = [int(v) for v in path[i]]
                    center1 = (x1 + w1 // 2, y1 + h1 // 2)
                    center2 = (x2 + w2 // 2, y2 + h2 // 2)
                    
                    # Calculate alpha for fading effect
                    alpha = (i / len(path)) * 0.8 + 0.2
                    line_color = tuple(int(c * alpha) for c in color)
                    
                    cv2.line(output, center1, center2, line_color, thickness)
        
        return output

    def get_statistics(self, fps=1.0):
        """Get speed and direction statistics for all tracked objects."""
        stats = {}
        for obj_id, tracker in self.trackers.items():
            if tracker.initialized:
                speed, direction = tracker.get_speed_and_direction(fps=fps)
                stats[obj_id] = {
                    'speed': speed,
                    'direction': direction,
                    'status': 'tracked',
                    'occluded': any(obj_id in pair for pair in self.occlusion_pairs)
                }
        
        # Include predicted statistics for lost objects
        for obj_id, lost_info in self.lost.items():
            if 'last_velocity' in lost_info:
                dx, dy = lost_info['last_velocity']
                speed = math.hypot(dx, dy) * fps
                direction = math.degrees(math.atan2(dy, dx)) % 360
                stats[obj_id] = {
                    'speed': speed,
                    'direction': direction,
                    'status': 'lost',
                    'frames_lost': lost_info['frames_lost']
                }
        
        return stats 