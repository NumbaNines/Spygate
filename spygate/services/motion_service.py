import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from ..ml.situation_detector import SituationDetector
from ..models.motion_events import MotionEvent, MotionHeatmap, MotionPattern, Situation
from ..video.motion_detector import HardwareAwareMotionDetector
from ..visualization.motion_visualizer import MotionVisualizer


class MotionService:
    """Service for handling motion detection, analysis, and persistence."""

    def __init__(
        self,
        db_session: Session,
        hardware_aware: bool = True,
        store_heatmaps: bool = True,
        store_patterns: bool = True,
        heatmap_interval: int = 300,  # frames
    ):
        """
        Initialize the motion service.

        Args:
            db_session: SQLAlchemy database session
            hardware_aware: Whether to use hardware-aware processing
            store_heatmaps: Whether to store motion heatmaps
            store_patterns: Whether to store motion patterns
            heatmap_interval: Number of frames between heatmap aggregations
        """
        self.db = db_session
        self.hardware_aware = hardware_aware
        self.store_heatmaps = store_heatmaps
        self.store_patterns = store_patterns
        self.heatmap_interval = heatmap_interval

        # Initialize components
        self.motion_detector = HardwareAwareMotionDetector()
        self.situation_detector = SituationDetector()
        self.visualizer = MotionVisualizer()

        # Initialize state
        self.current_video_id = None
        self.frame_count = 0
        self.accumulated_heatmap = None
        self.last_heatmap_frame = 0
        self.pattern_buffer = []

    def process_frame(
        self, frame: np.ndarray, video_id: int, frame_number: int, fps: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Process a video frame for motion detection and analysis.

        Args:
            frame: The video frame to process
            video_id: ID of the video being processed
            frame_number: Current frame number
            fps: Frames per second of the video

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (Visualized frame, Detection results)
        """
        # Update state if video changed
        if self.current_video_id != video_id:
            self._handle_video_change(video_id, frame.shape)

        # Detect motion
        motion_result = self.motion_detector.detect_motion(frame)

        # Detect situations
        situations = self.situation_detector.detect_situations(frame, frame_number, fps)

        # Store results
        event = self._store_motion_event(
            video_id,
            frame_number,
            frame_number / fps,
            motion_result,
            situations["situations"],
        )

        # Update heatmap
        if self.store_heatmaps:
            self._update_heatmap(frame, motion_result, video_id, frame_number, fps)

        # Update pattern detection
        if self.store_patterns:
            self._update_patterns(
                video_id, frame_number, fps, motion_result, situations["situations"]
            )

        # Create visualization
        vis_frame = self.visualizer.update(frame, motion_result, situations["situations"])

        return vis_frame, {
            "motion": motion_result,
            "situations": situations,
            "event_id": event.id if event else None,
        }

    def _handle_video_change(self, video_id: int, frame_shape: tuple[int, int, int]):
        """Handle state reset when processing a new video."""
        self.current_video_id = video_id
        self.frame_count = 0
        self.last_heatmap_frame = 0
        self.pattern_buffer.clear()

        if self.store_heatmaps:
            self.accumulated_heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

        # Initialize detectors for new video
        self.motion_detector.reset()
        self.situation_detector.initialize()
        self.visualizer.reset()

    def _store_motion_event(
        self,
        video_id: int,
        frame_number: int,
        timestamp: float,
        motion_result: dict[str, Any],
        situations: list[dict[str, Any]],
    ) -> Optional[MotionEvent]:
        """Store motion event and situations in database."""
        if not motion_result["motion_detected"]:
            return None

        # Create motion event
        event = MotionEvent(
            video_id=video_id,
            frame_number=frame_number,
            timestamp=timestamp,
            motion_score=motion_result["score"],
            hardware_tier=motion_result.get("hardware_tier"),
            processing_fps=motion_result.get("processing_fps"),
            regions=json.dumps(motion_result.get("regions", [])),
            metadata=json.dumps(motion_result.get("metadata", {})),
        )
        self.db.add(event)

        # Create situations
        for sit in situations:
            situation = Situation(
                motion_event=event,
                type=sit["type"],
                confidence=sit["confidence"],
                field_region=sit.get("details", {}).get("region"),
                direction=sit.get("details", {}).get("direction"),
                speed=sit.get("details", {}).get("speed"),
                duration=sit.get("details", {}).get("duration"),
                details=json.dumps(sit.get("details", {})),
            )
            self.db.add(situation)

        self.db.commit()
        return event

    def _update_heatmap(
        self,
        frame: np.ndarray,
        motion_result: dict[str, Any],
        video_id: int,
        frame_number: int,
        fps: float,
    ):
        """Update and store motion heatmap."""
        # Add current motion to accumulated heatmap
        if motion_result["motion_detected"]:
            mask = np.zeros_like(self.accumulated_heatmap)
            cv2.drawContours(mask, motion_result["contours"], -1, motion_result["score"], -1)
            self.accumulated_heatmap = cv2.add(self.accumulated_heatmap, mask)

        # Store heatmap at intervals
        frames_since_last = frame_number - self.last_heatmap_frame
        if frames_since_last >= self.heatmap_interval:
            # Normalize and compress heatmap
            norm_heatmap = cv2.normalize(
                self.accumulated_heatmap, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Create heatmap record
            heatmap = MotionHeatmap(
                video_id=video_id,
                start_frame=self.last_heatmap_frame,
                end_frame=frame_number,
                start_time=self.last_heatmap_frame / fps,
                end_time=frame_number / fps,
                heatmap_data=json.dumps(norm_heatmap.tolist()),
                resolution=json.dumps({"height": frame.shape[0], "width": frame.shape[1]}),
                metadata=json.dumps(
                    {
                        "max_intensity": float(np.max(self.accumulated_heatmap)),
                        "mean_intensity": float(np.mean(self.accumulated_heatmap)),
                    }
                ),
            )
            self.db.add(heatmap)
            self.db.commit()

            # Reset for next interval
            self.accumulated_heatmap.fill(0)
            self.last_heatmap_frame = frame_number

    def _update_patterns(
        self,
        video_id: int,
        frame_number: int,
        fps: float,
        motion_result: dict[str, Any],
        situations: list[dict[str, Any]],
    ):
        """Update and store motion patterns."""
        if not motion_result["motion_detected"]:
            return

        # Add to pattern buffer
        self.pattern_buffer.append(
            {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "motion": motion_result,
                "situations": situations,
            }
        )

        # Keep buffer size reasonable
        if len(self.pattern_buffer) > 90:  # 3 seconds at 30fps
            self.pattern_buffer.pop(0)

        # Analyze patterns in buffer
        if len(self.pattern_buffer) >= 30:  # 1 second at 30fps
            patterns = self._analyze_pattern_buffer(video_id)

            # Store detected patterns
            for pattern in patterns:
                motion_pattern = MotionPattern(
                    video_id=video_id,
                    start_frame=pattern["start_frame"],
                    end_frame=pattern["end_frame"],
                    pattern_type=pattern["type"],
                    confidence=pattern["confidence"],
                    field_region=pattern.get("field_region"),
                    direction=pattern.get("direction"),
                    speed=pattern.get("speed"),
                    duration=pattern["duration"],
                    trajectory=json.dumps(pattern.get("trajectory", [])),
                    metadata=json.dumps(pattern.get("metadata", {})),
                )
                self.db.add(motion_pattern)

            if patterns:
                self.db.commit()

    def _analyze_pattern_buffer(self, video_id: int) -> list[dict[str, Any]]:
        """Analyze motion pattern buffer for patterns."""
        patterns = []

        # Skip if buffer too small
        if len(self.pattern_buffer) < 2:
            return patterns

        # Get motion centroids from buffer
        centroids = []
        for entry in self.pattern_buffer:
            if entry["motion"]["contours"]:
                moments = cv2.moments(entry["motion"]["contours"][0])
                if moments["m00"] != 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    centroids.append(
                        {
                            "x": cx,
                            "y": cy,
                            "frame": entry["frame_number"],
                            "time": entry["timestamp"],
                        }
                    )

        if len(centroids) < 2:
            return patterns

        # Analyze trajectory
        start_frame = self.pattern_buffer[0]["frame_number"]
        end_frame = self.pattern_buffer[-1]["frame_number"]
        duration = (end_frame - start_frame) / fps

        # Calculate overall motion
        dx = centroids[-1]["x"] - centroids[0]["x"]
        dy = centroids[-1]["y"] - centroids[0]["y"]
        total_distance = np.sqrt(dx * dx + dy * dy)
        speed = total_distance / duration if duration > 0 else 0
        angle = np.arctan2(dy, dx) * 180 / np.pi

        # Determine pattern type
        if speed > 100:  # Fast motion
            pattern_type = "rapid_movement"
            confidence = min(speed / 200, 0.95)
        elif len(centroids) > 10:  # Sustained motion
            pattern_type = "sustained_movement"
            confidence = min(len(centroids) / 30, 0.95)
        else:  # Brief motion
            pattern_type = "brief_movement"
            confidence = 0.7

        # Create pattern record
        patterns.append(
            {
                "type": pattern_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "confidence": confidence,
                "direction": self._classify_direction(angle),
                "speed": speed,
                "duration": duration,
                "trajectory": centroids,
                "metadata": {
                    "angle": angle,
                    "total_distance": total_distance,
                    "centroid_count": len(centroids),
                },
            }
        )

        return patterns

    def _classify_direction(self, angle: float) -> str:
        """Classify motion direction based on angle."""
        if -45 <= angle <= 45:
            return "right"
        elif 45 < angle <= 135:
            return "down"
        elif angle > 135 or angle <= -135:
            return "left"
        else:
            return "up"

    def get_motion_events(
        self,
        video_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        Retrieve motion events for a video within a time range.

        Args:
            video_id: ID of the video
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            min_confidence: Minimum confidence for situations

        Returns:
            List[Dict[str, Any]]: List of motion events with their situations
        """
        query = self.db.query(MotionEvent).filter(MotionEvent.video_id == video_id)

        if start_time is not None:
            query = query.filter(MotionEvent.timestamp >= start_time)
        if end_time is not None:
            query = query.filter(MotionEvent.timestamp <= end_time)

        events = []
        for event in query.all():
            situations = [
                {
                    "type": s.type,
                    "confidence": s.confidence,
                    "field_region": s.field_region,
                    "direction": s.direction,
                    "speed": s.speed,
                    "duration": s.duration,
                    "details": json.loads(s.details) if s.details else {},
                }
                for s in event.situations
                if s.confidence >= min_confidence
            ]

            if situations:  # Only include events with qualifying situations
                events.append(
                    {
                        "id": event.id,
                        "frame_number": event.frame_number,
                        "timestamp": event.timestamp,
                        "motion_score": event.motion_score,
                        "regions": json.loads(event.regions) if event.regions else [],
                        "situations": situations,
                        "metadata": (json.loads(event.metadata) if event.metadata else {}),
                    }
                )

        return events

    def get_motion_heatmap(
        self, video_id: int, start_time: float, end_time: float
    ) -> Optional[np.ndarray]:
        """
        Retrieve aggregated motion heatmap for a time range.

        Args:
            video_id: ID of the video
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Optional[np.ndarray]: Aggregated heatmap or None if no data
        """
        heatmaps = (
            self.db.query(MotionHeatmap)
            .filter(
                MotionHeatmap.video_id == video_id,
                MotionHeatmap.start_time >= start_time,
                MotionHeatmap.end_time <= end_time,
            )
            .all()
        )

        if not heatmaps:
            return None

        # Get frame dimensions from first heatmap
        resolution = json.loads(heatmaps[0].resolution)
        height, width = resolution["height"], resolution["width"]

        # Initialize accumulated heatmap
        accumulated = np.zeros((height, width), dtype=np.float32)

        # Combine heatmaps
        for hm in heatmaps:
            heatmap_data = np.array(json.loads(hm.heatmap_data), dtype=np.uint8)
            accumulated = cv2.add(accumulated, heatmap_data)

        # Normalize final heatmap
        return cv2.normalize(accumulated, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def get_motion_patterns(
        self,
        video_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pattern_types: Optional[list[str]] = None,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        Retrieve motion patterns for a video within a time range.

        Args:
            video_id: ID of the video
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            pattern_types: List of pattern types to include (optional)
            min_confidence: Minimum confidence threshold

        Returns:
            List[Dict[str, Any]]: List of motion patterns
        """
        query = self.db.query(MotionPattern).filter(
            MotionPattern.video_id == video_id,
            MotionPattern.confidence >= min_confidence,
        )

        if start_time is not None:
            query = query.filter(MotionPattern.start_time >= start_time)
        if end_time is not None:
            query = query.filter(MotionPattern.end_time <= end_time)
        if pattern_types:
            query = query.filter(MotionPattern.pattern_type.in_(pattern_types))

        patterns = []
        for pattern in query.all():
            patterns.append(
                {
                    "id": pattern.id,
                    "type": pattern.pattern_type,
                    "start_frame": pattern.start_frame,
                    "end_frame": pattern.end_frame,
                    "confidence": pattern.confidence,
                    "field_region": pattern.field_region,
                    "direction": pattern.direction,
                    "speed": pattern.speed,
                    "duration": pattern.duration,
                    "trajectory": (json.loads(pattern.trajectory) if pattern.trajectory else []),
                    "metadata": (json.loads(pattern.metadata) if pattern.metadata else {}),
                }
            )

        return patterns
