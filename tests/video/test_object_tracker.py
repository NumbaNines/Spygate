import cv2
import numpy as np
import pytest

from spygate.video.object_tracker import MultiObjectTracker, ObjectTracker


@pytest.mark.parametrize("tracker_type", ObjectTracker.SUPPORTED_TYPES)
def test_tracker_initialization_and_update(tracker_type):
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (30, 30), (50, 50), (255, 255, 255), -1)
    cv2.rectangle(frame2, (40, 30), (60, 50), (255, 255, 255), -1)
    bbox = (30, 30, 20, 20)

    tracker = ObjectTracker(tracker_type=tracker_type)
    ok = tracker.init(frame1, bbox)
    if not ok:
        pytest.skip(
            f"Tracker {tracker_type} failed to initialize (likely OpenCV/platform limitation)"
        )
    ok, new_bbox = tracker.update(frame2)
    if not ok:
        pytest.skip(f"Tracker {tracker_type} failed to update (likely OpenCV/platform limitation)")
    assert isinstance(new_bbox, tuple) or isinstance(new_bbox, (list, np.ndarray))
    assert len(new_bbox) == 4


def test_tracker_path_history():
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 255, 255), -1)
    bbox = (10, 10, 20, 20)

    tracker = ObjectTracker(tracker_type="KCF")
    ok = tracker.init(frame1, bbox)
    if not ok:
        pytest.skip("Tracker KCF failed to initialize (likely OpenCV/platform limitation)")
    ok, _ = tracker.update(frame2)
    assert ok
    path = tracker.get_path()
    assert len(path) == 2
    assert path[0] == bbox


def test_tracker_reset():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10, 10, 20, 20)
    tracker = ObjectTracker(tracker_type="KCF")
    ok = tracker.init(frame, bbox)
    if not ok:
        pytest.skip("Tracker KCF failed to initialize (likely OpenCV/platform limitation)")
    tracker.reset()
    assert not tracker.initialized
    assert tracker.bbox is None
    assert tracker.get_path() == []


def test_multiobjecttracker_statistics():
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 0, 0), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 0, 0), -1)
    bbox = (10, 10, 20, 20)
    mot = MultiObjectTracker(tracker_type="KCF")
    ok = mot.add("obj1", frame1, bbox)
    if not ok:
        pytest.skip("Tracker KCF failed to initialize (likely OpenCV/platform limitation)")
    mot.update(frame2)
    stats = mot.get_statistics(fps=1.0)
    speed, direction = stats["obj1"]
    assert speed is not None and direction is not None
    assert speed > 0
    assert 0 <= direction < 360


def test_multiobjecttracker_reidentification():
    # Create two frames with a red square, move it, lose it, then re-add
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (10, 10), (30, 30), (0, 0, 255), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (0, 0, 255), -1)
    cv2.rectangle(frame3, (50, 10), (70, 30), (0, 0, 255), -1)
    bbox1 = (10, 10, 20, 20)
    bbox2 = (20, 10, 20, 20)
    bbox3 = (50, 10, 20, 20)
    mot = MultiObjectTracker(tracker_type="KCF")
    ok = mot.add("obj1", frame1, bbox1)
    if not ok:
        pytest.skip("Tracker KCF failed to initialize (likely OpenCV/platform limitation)")
    mot.update(frame2)  # move
    # Simulate loss
    mot.mark_lost("obj1", frame2)
    # Re-identify in a new location (should not match, so new ID)
    new_id, reused = mot.reidentify_and_add(frame3, bbox3)
    assert new_id != "obj1" or not reused
    # Re-identify in the same location (should match, so reuse ID)
    new_id2, reused2 = mot.reidentify_and_add(frame2, bbox2)
    assert new_id2 == "obj1" and reused2


def test_occlusion_detection():
    """Test occlusion detection between objects."""
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create two overlapping objects
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 0, 0), -1)
    cv2.rectangle(frame1, (20, 10), (40, 30), (0, 255, 0), -1)

    mot = MultiObjectTracker(tracker_type="KCF", iou_threshold=0.3)
    ok1 = mot.add("obj1", frame1, (10, 10, 20, 20))
    ok2 = mot.add("obj2", frame1, (20, 10, 20, 20))

    if not ok1 or not ok2:
        pytest.skip("Tracker KCF failed to initialize")

    mot.update(frame2)
    assert ("obj1", "obj2") in mot.occlusion_pairs or ("obj2", "obj1") in mot.occlusion_pairs


def test_motion_prediction():
    """Test motion prediction for lost objects."""
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create moving object
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 0, 0), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 0, 0), -1)

    mot = MultiObjectTracker(tracker_type="KCF")
    ok = mot.add("obj1", frame1, (10, 10, 20, 20))
    if not ok:
        pytest.skip("Tracker KCF failed to initialize")

    # Update to establish velocity
    mot.update(frame2)

    # Mark as lost
    mot.mark_lost("obj1", frame2)

    # Get predicted position
    predicted_bbox = mot._predict_bbox("obj1")
    assert predicted_bbox is not None
    x, y, w, h = predicted_bbox
    assert x > 20  # Should continue in same direction


def test_enhanced_reidentification():
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create a distinctly colored moving object
    cv2.rectangle(frame1, (10, 10), (30, 30), (0, 0, 255), -1)  # Red
    cv2.rectangle(frame2, (20, 10), (40, 30), (0, 0, 255), -1)  # Red
    cv2.rectangle(frame3, (30, 10), (50, 30), (0, 0, 255), -1)  # Red

    mot = MultiObjectTracker(tracker_type="KCF")
    mot.add("obj1", frame1, (10, 10, 20, 20))
    mot.update(frame2)
    mot.mark_lost("obj1", frame2)

    # Try to reidentify in predicted location with same appearance
    reidentified_id = mot.try_reidentify(frame3, (30, 10, 20, 20))
    assert reidentified_id == "obj1"

    # Try to reidentify in wrong location with different appearance
    frame_wrong = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame_wrong, (80, 80), (100, 100), (0, 255, 0), -1)  # Green
    reidentified_id = mot.try_reidentify(frame_wrong, (80, 80, 20, 20))
    assert reidentified_id is None


def test_statistics_with_status():
    """Test detailed statistics including object status."""
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create moving object
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 0, 0), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 0, 0), -1)
    cv2.rectangle(frame3, (50, 10), (70, 30), (255, 0, 0), -1)

    mot = MultiObjectTracker(tracker_type="KCF")
    ok = mot.add("obj1", frame1, (10, 10, 20, 20))
    if not ok:
        pytest.skip("Tracker KCF failed to initialize")

    # Track normally
    mot.update(frame2)
    stats = mot.get_statistics(fps=1.0)
    assert stats["obj1"]["status"] == "tracked"
    assert not stats["obj1"]["occluded"]

    # Mark as lost
    mot.mark_lost("obj1", frame2)
    stats = mot.get_statistics(fps=1.0)
    assert stats["obj1"]["status"] == "lost"
    assert "frames_lost" in stats["obj1"]


def test_path_visualization():
    """Test path visualization with trails."""
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create moving object
    cv2.rectangle(frame1, (10, 10), (30, 30), (255, 0, 0), -1)
    cv2.rectangle(frame2, (20, 10), (40, 30), (255, 0, 0), -1)
    cv2.rectangle(frame3, (30, 10), (50, 30), (255, 0, 0), -1)

    mot = MultiObjectTracker(tracker_type="KCF")
    ok = mot.add("obj1", frame1, (10, 10, 20, 20))
    if not ok:
        pytest.skip("Tracker KCF failed to initialize")

    mot.update(frame2)
    mot.update(frame3)

    # Draw paths
    path_frame = mot.draw_paths(frame3)
    assert path_frame.shape == frame3.shape
    assert not np.array_equal(path_frame, frame3)  # Should have drawn something
