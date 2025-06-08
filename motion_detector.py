import csv
import json
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import psutil


class MotionEventLogger:
    """Handles logging of motion events to CSV, JSON, or SQLite database."""

    def __init__(self, csv_path=None, json_path=None, db_path=None):
        self.csv_path = csv_path
        self.json_path = json_path
        self.db_path = db_path
        if db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS motion_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            roi TEXT,
            magnitude REAL,
            snapshot_path TEXT
        )"""
        )
        conn.commit()
        conn.close()

    def log_event(self, timestamp, roi, magnitude, snapshot_path=None):
        event = {
            "timestamp": timestamp,
            "roi": roi,
            "magnitude": magnitude,
            "snapshot_path": snapshot_path,
        }
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=event.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(event)
        if self.json_path:
            try:
                with open(self.json_path) as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                data = []
            data.append(event)
            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=2)
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """INSERT INTO motion_events (timestamp, roi, magnitude, snapshot_path) VALUES (?, ?, ?, ?)""",
                (timestamp, json.dumps(roi), magnitude, snapshot_path),
            )
            conn.commit()
            conn.close()


class MotionDetector:
    """
    MotionDetector for classic motion detection using frame differencing and background subtraction.
    """

    def __init__(
        self,
        sensitivity: int = 20,
        blur_size: int = 21,
        min_contour_area: int = 500,
        bg_subtractor_type: str = "MOG2",
    ):
        self.sensitivity = sensitivity
        self.blur_size = blur_size
        self.min_contour_area = min_contour_area
        self.last_frame: Optional[np.ndarray] = None
        if bg_subtractor_type == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN()
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def detect_frame_diff(self, frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Detect motion using frame differencing.
        Returns the thresholded image and list of contours.
        """
        masked_frame = self.mask_frame(frame) if self.get_rois() else frame
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self.last_frame is None:
            self.last_frame = gray
            return frame, []

        frame_delta = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(frame_delta, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_frame = gray

        # Filter small contours
        contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        return thresh, contours

    def detect_bg_subtraction(self, frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Detect motion using background subtraction.
        Returns the mask and list of contours.
        """
        masked_frame = self.mask_frame(frame) if self.get_rois() else frame
        fg_mask = self.bg_subtractor.apply(masked_frame)
        thresh = cv2.threshold(fg_mask, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter small contours
        contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        return thresh, contours

    def set_sensitivity(self, sensitivity: int) -> None:
        """Set the sensitivity (threshold value) for motion detection."""
        self.sensitivity = sensitivity

    def set_blur_size(self, blur_size: int) -> None:
        """Set the blur size for preprocessing frames."""
        self.blur_size = blur_size

    def set_min_contour_area(self, min_area: int) -> None:
        """Set the minimum contour area for filtering motion regions."""
        self.min_contour_area = min_area

    def set_rois(self, rois: list) -> None:
        """Set the list of ROIs. Each ROI can be a tuple (x, y, w, h) for rectangles or a list of points for polygons."""
        self.rois = rois

    def get_rois(self) -> list:
        """Get the current list of ROIs."""
        return getattr(self, "rois", [])

    def mask_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI masks to the frame. Only regions inside ROIs are kept for detection."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for roi in self.get_rois():
            if isinstance(roi, tuple) and len(roi) == 4:
                x, y, w, h = roi
                mask[y : y + h, x : x + w] = 255
            elif isinstance(roi, list):
                pts = np.array(roi, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return masked

    def get_config(self) -> dict:
        """Return the current configuration as a dictionary."""
        return {
            "sensitivity": self.sensitivity,
            "blur_size": self.blur_size,
            "min_contour_area": self.min_contour_area,
        }

    def save_config(self, path: str) -> None:
        """Save the current configuration to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2)

    def load_config(self, path: str) -> None:
        """Load configuration from a JSON file and update parameters."""
        import json

        with open(path) as f:
            config = json.load(f)
        self.sensitivity = config.get("sensitivity", self.sensitivity)
        self.blur_size = config.get("blur_size", self.blur_size)
        self.min_contour_area = config.get("min_contour_area", self.min_contour_area)

    def save_rois(self, path: str) -> None:
        """Save the current ROIs to a JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.get_rois(), f, indent=2)

    def load_rois(self, path: str) -> None:
        """Load ROIs from a JSON file."""
        import json

        with open(path) as f:
            self.rois = json.load(f)

    def draw_motion_boxes(
        self, frame: np.ndarray, contours: list, color: tuple = (0, 255, 0), thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes around detected motion areas on the frame."""
        output = frame.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        return output

    def draw_motion_contours(
        self, frame: np.ndarray, contours: list, color: tuple = (0, 0, 255), thickness: int = 2
    ) -> np.ndarray:
        """Draw contours of detected motion areas on the frame."""
        output = frame.copy()
        cv2.drawContours(output, contours, -1, color, thickness)
        return output

    def draw_motion_trails(
        self, frame: np.ndarray, trails: list, color: tuple = (255, 0, 0), thickness: int = 2
    ) -> np.ndarray:
        """Draw motion trails (list of points) on the frame."""
        output = frame.copy()
        for trail in trails:
            for i in range(1, len(trail)):
                cv2.line(output, trail[i - 1], trail[i], color, thickness)
        return output

    def generate_motion_heatmap(
        self, frame_shape: tuple, motion_points: list, alpha: float = 0.5
    ) -> np.ndarray:
        """Generate a heatmap of motion intensity from a list of points."""
        heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
        for pt in motion_points:
            x, y = pt
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                heatmap[y, x] += 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)
        heatmap = np.clip(heatmap / heatmap.max(), 0, 1)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.addWeighted(
            heatmap_color, alpha, np.zeros_like(heatmap_color), 1 - alpha, 0
        )
        return heatmap_color

    def set_event_logger(self, logger: MotionEventLogger):
        """Attach a MotionEventLogger instance to this detector."""
        self.event_logger = logger

    def log_motion_event(self, roi, magnitude, snapshot_path=None):
        """Log a motion event with current timestamp, ROI, magnitude, and optional snapshot."""
        if hasattr(self, "event_logger") and self.event_logger:
            timestamp = datetime.now().isoformat()
            self.event_logger.log_event(timestamp, roi, magnitude, snapshot_path)


class MotionDetector(MotionDetector):
    def __init__(
        self,
        sensitivity: int = 20,
        blur_size: int = 21,
        min_contour_area: int = 500,
        bg_subtractor_type: str = "MOG2",
        resize_width: int = None,
        resize_height: int = None,
        frame_skip: int = 0,
        use_threading: bool = False,
        use_gpu: bool = False,
    ):
        super().__init__(sensitivity, blur_size, min_contour_area, bg_subtractor_type)
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.frame_skip = frame_skip
        self.use_threading = use_threading
        self.use_gpu = use_gpu
        self._frame_count = 0

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if needed."""
        if self.resize_width and self.resize_height:
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        return frame

    def should_process_frame(self) -> bool:
        """Determine if the current frame should be processed (for frame skipping)."""
        self._frame_count += 1
        if self.frame_skip and (self._frame_count % (self.frame_skip + 1)) != 1:
            return False
        return True

    def process_frame(self, frame: np.ndarray, method: str = "diff") -> tuple:
        """Process a frame with optional threading and GPU acceleration."""
        frame = self.preprocess_frame(frame)
        if not self.should_process_frame():
            return None, []
        if self.use_threading:
            result = {}

            def worker():
                if method == "diff":
                    result["out"] = self.detect_frame_diff(frame)
                else:
                    result["out"] = self.detect_bg_subtraction(frame)

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            return result["out"]
        else:
            if method == "diff":
                return self.detect_frame_diff(frame)
            else:
                return self.detect_bg_subtraction(frame)

    def get_resource_usage(self) -> dict:
        """Return current CPU and memory usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }
