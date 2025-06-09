"""Service for analyzing video content and detecting situations."""

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from ..database import AnalysisJob, AnalysisStatus, Clip, Tag, create_clip, get_db
from ..ml.situation_detector import SituationDetector
from ..video.clip_extractor import ClipExtractor
from ..video.codec_validator import VideoMetadata

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for analyzing video content and detecting situations."""

    CLIPS_DIR = "uploads/clips"
    BUFFER_SECONDS = 5  # Seconds to include before/after detected situation

    def __init__(self):
        """Initialize the analysis service."""
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_jobs: dict[str, bool] = {}
        self._detector = SituationDetector()
        self._detector.initialize()
        self._lock = threading.Lock()

    def start_analysis(self, clip_id: str, video_path: str, metadata: dict[str, Any] = None) -> str:
        """Start a new analysis job.

        Args:
            clip_id: ID of the clip to analyze
            video_path: Path to the video file
            metadata: Optional metadata for the analysis

        Returns:
            str: ID of the created analysis job
        """
        with get_db() as db:
            # Create a new analysis job
            job = AnalysisJob(
                clip_id=clip_id,
                status=AnalysisStatus.PENDING,
                analysis_metadata=metadata or {},
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            # Start the analysis in a background thread
            self._executor.submit(self._run_analysis, job.id, video_path)

            return job.id

    def cancel_analysis(self, job_id: str) -> tuple[bool, str]:
        """Cancel an analysis job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            Tuple[bool, str]: Success status and error message if any
        """
        try:
            with self._lock:
                if job_id in self._active_jobs:
                    self._active_jobs[job_id] = False

            with get_db() as db:
                job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
                if not job:
                    return False, "Job not found"

                # Mark the job as cancelled
                job.status = AnalysisStatus.CANCELLED
                db.commit()

                return True, ""
        except Exception as e:
            logger.error(f"Failed to cancel analysis: {str(e)}")
            return False, f"Failed to cancel analysis: {str(e)}"

    def get_analysis_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get the status of an analysis job.

        Args:
            job_id: ID of the analysis job

        Returns:
            Optional[Dict[str, Any]]: Job status and details if found
        """
        with get_db() as db:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if not job:
                return None

            return {
                "status": job.status.value,
                "metadata": job.analysis_metadata,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            }

    def _run_analysis(self, job_id: str, video_path: str):
        """Run the analysis job.

        Args:
            job_id: ID of the analysis job
            video_path: Path to the video file
        """
        try:
            with get_db() as db:
                # Update job status
                job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
                job.status = AnalysisStatus.IN_PROGRESS
                db.commit()

            # Mark job as active
            self._active_jobs[job_id] = True

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Process frames in batches
            batch_size = 30  # 1 second of frames at 30fps
            frames = []
            frame_number = 0

            while cap.isOpened() and self._active_jobs.get(job_id, False):
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                frame_number += 1

                if len(frames) >= batch_size:
                    situations = self._process_frame_batch(
                        frames, job_id, frame_number - batch_size, fps
                    )
                    if situations:
                        self._create_situation_clips(job_id, situations, video_path, fps)
                    frames = []

            # Process remaining frames
            if frames and self._active_jobs.get(job_id, False):
                situations = self._process_frame_batch(
                    frames, job_id, frame_number - len(frames), fps
                )
                if situations:
                    self._create_situation_clips(job_id, situations, video_path, fps)

            # Update job status
            with get_db() as db:
                job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
                job.status = AnalysisStatus.COMPLETED
                db.commit()

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            self._handle_analysis_error(job_id, str(e))
        finally:
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            if "cap" in locals():
                cap.release()

    def _process_frame_batch(
        self, frames: list[np.ndarray], job_id: str, start_frame: int, fps: float
    ) -> list[dict[str, Any]]:
        """Process a batch of frames.

        Args:
            frames: List of frames to process
            job_id: ID of the analysis job
            start_frame: Starting frame number
            fps: Frames per second

        Returns:
            List[Dict[str, Any]]: List of detected situations
        """
        situations = []
        for i, frame in enumerate(frames):
            frame_number = start_frame + i
            timestamp = frame_number / fps

            # Detect situations in the frame
            result = self._detector.detect_situations(frame, frame_number, timestamp)
            if result["situations"]:
                situations.extend(result["situations"])

        return situations

    def _create_situation_clips(
        self, job_id: str, situations: list[dict[str, Any]], video_path: str, fps: float
    ) -> list[str]:
        """Create clips for detected situations.

        Args:
            job_id: ID of the analysis job
            situations: List of detected situations
            video_path: Path to the video file
            fps: Frames per second

        Returns:
            List[str]: List of created clip IDs
        """
        clip_ids = []

        with get_db() as db:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if not job:
                return clip_ids

            for situation in situations:
                # Calculate clip boundaries (5 seconds before and after)
                frame_number = situation["frame"]
                timestamp = frame_number / fps
                start_time = max(0, timestamp - 5)
                end_time = timestamp + 5

                # Create output path
                clip_filename = f"situation_{job.clip_id}_{frame_number}.mp4"
                output_path = os.path.join(self.CLIPS_DIR, clip_filename)

                # Extract the clip
                success, error = ClipExtractor.extract_segment(
                    source_path=video_path,
                    output_path=output_path,
                    start_time=start_time,
                    end_time=end_time,
                )

                if success:
                    # Create clip record
                    clip = create_clip(
                        db=db,
                        user_id=job.clip.user_id,
                        title=f"Detected: {situation['type']}",
                        description=f"Automatically detected {situation['type']} at {timestamp:.2f}s",
                        file_path=output_path,
                        duration=end_time - start_time,
                        source_clip_id=job.clip_id,
                    )

                    # Add tags
                    tag = db.query(Tag).filter(Tag.name == situation["type"]).first()
                    if not tag:
                        tag = Tag(name=situation["type"])
                        db.add(tag)
                    clip.tags.append(tag)

                    db.commit()
                    clip_ids.append(clip.id)

        return clip_ids

    def _handle_analysis_error(self, job_id: str, error_message: str):
        """Handle an error during analysis.

        Args:
            job_id: ID of the failed job
            error_message: Error message
        """
        with get_db() as db:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = AnalysisStatus.FAILED
                job.error_message = error_message
                db.commit()
