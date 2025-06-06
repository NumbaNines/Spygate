import cv2
import numpy as np
import math

class ObjectTracker:
    """
    ObjectTracker wraps OpenCV's tracking API to track a single object across video frames.
    Supports multiple tracker types (KCF, CSRT, etc.).
    """
    SUPPORTED_TYPES = [
        "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "CSRT", "MOSSE"
    ]

    def __init__(self, tracker_type='KCF'):
        if tracker_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type)
        self.initialized = False
        self.bbox = None
        self.history = []

    def _create_tracker(self, tracker_type):
        creator_map = {
            'BOOSTING': ['TrackerBoosting_create'],
            'MIL': ['TrackerMIL_create'],
            'KCF': ['TrackerKCF_create'],
            'TLD': ['TrackerTLD_create'],
            'MEDIANFLOW': ['TrackerMedianFlow_create'],
            'CSRT': ['TrackerCSRT_create'],
            'MOSSE': ['TrackerMOSSE_create'],
        }
        for attr in creator_map[tracker_type]:
            # Try cv2 first
            if hasattr(cv2, attr):
                return getattr(cv2, attr)()
            # Try cv2.legacy if it exists
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, attr):
                return getattr(cv2.legacy, attr)()
        raise ValueError(f"Tracker type {tracker_type} is not available in your OpenCV build.")

    def init(self, frame, bbox):
        """
        Initialize tracker with the first frame and bounding box.
        bbox: (x, y, w, h)
        """
        self.bbox = bbox
        ok = self.tracker.init(frame, bbox)
        self.initialized = ok
        if ok:
            self.history = [bbox]
        return ok

    def update(self, frame):
        """
        Update tracker with a new frame.
        Returns (ok, bbox): ok is True if tracking succeeded, bbox is (x, y, w, h)
        """
        if not self.initialized:
            return False, None
        ok, bbox = self.tracker.update(frame)
        if ok:
            self.bbox = bbox
            self.history.append(bbox)
        return ok, bbox

    def reset(self):
        """
        Reset the tracker to its initial state.
        """
        self.tracker = self._create_tracker(self.tracker_type)
        self.initialized = False
        self.bbox = None
        self.history = []

    def get_path(self):
        """
        Returns the list of bounding boxes tracked so far.
        """
        return self.history 

    def get_speed_and_direction(self, fps=1.0):
        """
        Compute the speed (pixels/sec) and direction (angle in degrees, 0=right, 90=down) of the tracked object.
        Returns (speed, direction) or (None, None) if not enough data.
        """
        if len(self.history) < 2:
            return None, None
        # Use the last two positions
        x1, y1, w1, h1 = self.history[-2]
        x2, y2, w2, h2 = self.history[-1]
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = math.hypot(dx, dy)
        speed = distance * fps  # pixels per second
        angle = math.degrees(math.atan2(dy, dx)) % 360  # 0=right, 90=down
        return speed, angle

    def get_histogram(self, frame):
        """Compute a color histogram for the current bbox in the given frame."""
        if self.bbox is None:
            return None
        x, y, w, h = [int(v) for v in self.bbox]
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

class MultiObjectTracker:
    """
    MultiObjectTracker manages multiple ObjectTracker instances for multi-object tracking.
    Each object is tracked with a unique ID.
    """
    def __init__(self, tracker_type='KCF'):
        self.tracker_type = tracker_type
        self.trackers = {}  # id -> ObjectTracker
        self.lost = {}      # id -> {'hist': ..., 'bbox': ...}

    def add(self, obj_id, frame, bbox):
        """Add a new object to track with a given ID and bounding box."""
        tracker = ObjectTracker(tracker_type=self.tracker_type)
        ok = tracker.init(frame, bbox)
        if ok:
            self.trackers[obj_id] = tracker
        return ok

    def mark_lost(self, obj_id, frame):
        """Mark a tracker as lost and store its last known histogram and bbox."""
        tracker = self.trackers.get(obj_id)
        if tracker and tracker.bbox is not None:
            hist = tracker.get_histogram(frame)
            self.lost[obj_id] = {'hist': hist, 'bbox': tracker.bbox}
            del self.trackers[obj_id]

    def try_reidentify(self, frame, bbox, threshold=0.7):
        """Try to re-identify a lost object by comparing histograms. Returns obj_id or None."""
        x, y, w, h = [int(v) for v in bbox]
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        best_id = None
        best_score = 0
        for obj_id, lost_info in self.lost.items():
            lost_hist = lost_info['hist']
            if lost_hist is None:
                continue
            score = cv2.compareHist(hist, lost_hist, cv2.HISTCMP_CORREL)
            if score > best_score and score > threshold:
                best_score = score
                best_id = obj_id
        return best_id

    def update(self, frame):
        """Update all trackers with the new frame. Returns dict of obj_id -> (ok, bbox)."""
        results = {}
        lost_ids = []
        for obj_id, tracker in list(self.trackers.items()):
            ok, bbox = tracker.update(frame)
            results[obj_id] = (ok, bbox)
            if not ok:
                lost_ids.append(obj_id)
        for obj_id in lost_ids:
            self.mark_lost(obj_id, frame)
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
        """Draw bounding boxes for all tracked objects on the frame."""
        output = frame.copy()
        for obj_id, tracker in self.trackers.items():
            if tracker.initialized and tracker.bbox is not None:
                x, y, w, h = [int(v) for v in tracker.bbox]
                color = self._get_color(obj_id)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(output, str(obj_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return output

    def draw_paths(self, frame, thickness=2):
        """Draw path trails for all tracked objects on the frame."""
        output = frame.copy()
        for obj_id, tracker in self.trackers.items():
            path = tracker.get_path()
            color = self._get_color(obj_id)
            if len(path) > 1:
                for i in range(1, len(path)):
                    x1, y1, w1, h1 = [int(v) for v in path[i-1]]
                    x2, y2, w2, h2 = [int(v) for v in path[i]]
                    center1 = (x1 + w1 // 2, y1 + h1 // 2)
                    center2 = (x2 + w2 // 2, y2 + h2 // 2)
                    cv2.line(output, center1, center2, color, thickness)
        return output

    def get_statistics(self, fps=1.0):
        """
        Get speed and direction statistics for all tracked objects.
        Returns dict of obj_id -> (speed, direction)
        """
        stats = {}
        for obj_id, tracker in self.trackers.items():
            if tracker.initialized:
                stats[obj_id] = tracker.get_speed_and_direction(fps=fps)
        return stats

    def reidentify_and_add(self, frame, bbox):
        """Try to re-identify a lost object and add it back, or add as new if not found."""
        obj_id = self.try_reidentify(frame, bbox)
        if obj_id is not None:
            # Reuse the old ID
            self.add(obj_id, frame, bbox)
            del self.lost[obj_id]
            return obj_id, True
        else:
            # Assign a new ID (e.g., increment or use a UUID)
            new_id = f"obj{len(self.trackers) + len(self.lost) + 1}"
            self.add(new_id, frame, bbox)
            return new_id, False 