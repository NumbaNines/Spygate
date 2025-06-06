import cv2
import numpy as np
import pytest
from motion_detector import MotionDetector, MotionEventLogger
import os

def generate_test_frame(motion=False, shape=(480, 640, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    if motion:
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    return frame

def test_frame_differencing_detection():
    detector = MotionDetector()
    frame1 = generate_test_frame(motion=False)
    frame2 = generate_test_frame(motion=True)
    detector.detect_frame_diff(frame1)  # Initialize
    _, contours = detector.detect_frame_diff(frame2)
    assert len(contours) > 0

def test_background_subtraction_detection():
    detector = MotionDetector()
    frame1 = generate_test_frame(motion=False)
    frame2 = generate_test_frame(motion=True)
    detector.detect_bg_subtraction(frame1)
    _, contours = detector.detect_bg_subtraction(frame2)
    assert len(contours) > 0

def test_sensitivity_adjustment():
    detector = MotionDetector(sensitivity=10)
    assert detector.sensitivity == 10
    detector.set_sensitivity(30)
    assert detector.sensitivity == 30

def test_roi_selection():
    detector = MotionDetector()
    roi = (50, 50, 100, 100)
    detector.set_rois([roi])
    assert detector.get_rois() == [roi]
    frame = generate_test_frame(motion=True)
    masked = detector.mask_frame(frame)
    assert masked.shape == frame.shape

def test_visualization_methods():
    detector = MotionDetector()
    frame = generate_test_frame(motion=True)
    detector.detect_frame_diff(frame)  # Initialize
    _, contours = detector.detect_frame_diff(frame)
    box_img = detector.draw_motion_boxes(frame, contours)
    contour_img = detector.draw_motion_contours(frame, contours)
    trail_img = detector.draw_motion_trails(frame, [[(100, 100), (110, 110), (120, 120)]])
    heatmap = detector.generate_motion_heatmap(frame.shape, [(100, 100), (110, 110)])
    assert box_img.shape == frame.shape
    assert contour_img.shape == frame.shape
    assert trail_img.shape == frame.shape
    assert heatmap.shape[:2] == frame.shape[:2]

def test_event_logging(tmp_path):
    csv_path = tmp_path / "events.csv"
    json_path = tmp_path / "events.json"
    db_path = tmp_path / "events.db"
    logger = MotionEventLogger(str(csv_path), str(json_path), str(db_path))
    detector = MotionDetector()
    detector.set_event_logger(logger)
    detector.log_motion_event(roi=(0,0,100,100), magnitude=42.0, snapshot_path=None)
    # Check CSV
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 1
    # Check JSON
    with open(json_path, 'r') as f:
        data = f.read()
        assert 'magnitude' in data
    # Check DB
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute('SELECT * FROM motion_events')
    rows = c.fetchall()
    assert len(rows) > 0
    conn.close()

def test_performance_and_resource_usage():
    detector = MotionDetector(resize_width=320, resize_height=240, frame_skip=2)
    frame = generate_test_frame(motion=True, shape=(1080, 1920, 3))
    for _ in range(10):
        detector.process_frame(frame)
    usage = detector.get_resource_usage()
    assert 'cpu_percent' in usage and 'memory_percent' in usage

def test_edge_cases():
    detector = MotionDetector()
    # Low light (almost black frame)
    frame_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.detect_frame_diff(frame_dark)
    _, contours = detector.detect_frame_diff(frame_dark)
    assert isinstance(contours, list)
    # Rapid movement (large difference)
    frame1 = generate_test_frame(motion=False)
    frame2 = generate_test_frame(motion=True)
    detector.detect_frame_diff(frame1)
    _, contours = detector.detect_frame_diff(frame2)
    assert len(contours) > 0 