"""Tests for the enhanced game analyzer integration."""

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ...core.hardware import HardwareDetector, HardwareTier
from ...ml.enhanced_game_analyzer import (
    HARDWARE_TIERS,
    UI_CLASSES,
    EnhancedGameAnalyzer,
    GameState,
    TriangleValidation,
)


def test_game_state_extraction(game_analyzer, sample_frame, mock_yolo_model, mock_ocr):
    """Test complete game state extraction pipeline."""
    # Configure mock detections
    mock_yolo_model.detect_hud_elements.return_value = [
        {"class_name": "hud", "bbox": [0, 0, 1920, 100], "confidence": 0.95},
        {"class_name": "possession_triangle_area", "bbox": [400, 10, 450, 40], "confidence": 0.92},
        {"class_name": "territory_triangle_area", "bbox": [1500, 10, 1550, 40], "confidence": 0.94},
        {"class_name": "preplay_indicator", "bbox": [800, 200, 1000, 250], "confidence": 0.88},
    ]

    # Configure mock OCR responses
    mock_ocr.read_team_names.return_value = ("RAIDERS", "CHIEFS")
    mock_ocr.read_game_situation.return_value = {
        "down": 3,
        "distance": 8,
        "quarter": 2,
        "time": "4:35",
        "score_home": 14,
        "score_away": 7,
    }

    # Extract game state
    game_state = game_analyzer.extract_game_state(sample_frame)

    # Verify all components
    assert game_state["hud_detected"] is True
    assert game_state["team_names"]["home"] == "RAIDERS"
    assert game_state["team_names"]["away"] == "CHIEFS"
    assert game_state["situation"]["down"] == 3
    assert game_state["situation"]["distance"] == 8
    assert game_state["situation"]["quarter"] == 2
    assert game_state["situation"]["time"] == "4:35"
    assert game_state["situation"]["score_home"] == 14
    assert game_state["situation"]["score_away"] == 7
    assert game_state["preplay_active"] is True
    assert game_state["play_call_active"] is False

    # Verify confidence scores
    assert game_state["confidence"]["hud"] == 0.95
    assert game_state["confidence"]["possession"] == 0.92
    assert game_state["confidence"]["territory"] == 0.94
    assert game_state["confidence"]["preplay"] == 0.88
    assert game_state["confidence"]["play_call"] == 0.0


def test_hardware_adaptive_processing(mock_hardware_detector):
    """Test hardware-adaptive processing configuration."""
    # Test HIGH tier configuration
    mock_hardware_detector.tier = HardwareTier.HIGH
    analyzer = EnhancedGameAnalyzer(hardware=mock_hardware_detector)

    assert analyzer.batch_size == 8
    assert analyzer.detection_interval == 2
    assert analyzer.model_size == "m"

    # Test LOW tier configuration
    mock_hardware_detector.tier = HardwareTier.LOW
    analyzer = EnhancedGameAnalyzer(hardware=mock_hardware_detector)

    assert analyzer.batch_size == 2
    assert analyzer.detection_interval == 4
    assert analyzer.model_size == "n"


def test_performance_monitoring(game_analyzer, sample_frame):
    """Test performance monitoring and metrics collection."""
    # Process multiple frames
    for _ in range(10):
        game_analyzer.extract_game_state(sample_frame)

    # Verify metrics collection
    assert len(game_analyzer.performance_metrics["processing_times"]) == 10
    assert len(game_analyzer.performance_metrics["confidence_scores"]) == 10
    assert len(game_analyzer.performance_metrics["detection_counts"]) == 10

    # Verify metrics are within expected ranges
    assert all(0 <= score <= 1 for score in game_analyzer.performance_metrics["confidence_scores"])
    assert all(count >= 0 for count in game_analyzer.performance_metrics["detection_counts"])
    assert all(time >= 0 for time in game_analyzer.performance_metrics["processing_times"])


def test_error_handling(game_analyzer, sample_frame, mock_yolo_model, mock_ocr):
    """Test error handling and recovery."""
    # Simulate model failure
    mock_yolo_model.detect_hud_elements.side_effect = Exception("Model error")

    # Should handle error gracefully
    game_state = game_analyzer.extract_game_state(sample_frame)
    assert game_state["hud_detected"] is False
    assert all(conf == 0.0 for conf in game_state["confidence"].values())

    # Simulate OCR failure
    mock_yolo_model.detect_hud_elements.side_effect = None
    mock_ocr.read_team_names.side_effect = Exception("OCR error")

    # Should handle error gracefully
    game_state = game_analyzer.extract_game_state(sample_frame)
    assert game_state["hud_detected"] is True
    assert game_state["team_names"]["home"] is None
    assert game_state["team_names"]["away"] is None


def test_play_state_detection(game_analyzer):
    """Test play state detection logic."""
    # Create mock frames with different states
    mock_frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)  # Pre-play state
    mock_frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)  # Play active
    mock_frame3 = np.zeros((720, 1280, 3), dtype=np.uint8)  # Play end

    # Configure mock detections for pre-play state
    game_analyzer.model.detect_objects = MagicMock(
        return_value=[
            {"class_name": "preplay_indicator", "bbox": [0, 0, 100, 100], "confidence": 0.95}
        ]
    )

    # Process pre-play frame
    state1 = game_analyzer.process_frame(mock_frame1, timestamp=1.0)
    assert state1.state_indicators["preplay_indicator"] is True
    assert state1.play_active is False
    assert state1.last_preplay_time == 1.0

    # Configure mock detections for play active (no pre-play)
    game_analyzer.model.detect_objects = MagicMock(
        return_value=[{"class_name": "hud", "bbox": [0, 0, 100, 100], "confidence": 0.95}]
    )

    # Process play active frame
    state2 = game_analyzer.process_frame(mock_frame2, timestamp=2.0)
    assert state2.state_indicators["preplay_indicator"] is False
    assert state2.play_active is True
    assert state2.play_start_time == 2.0
    assert state2.play_count == 1

    # Configure mock detections for play end (play call screen appears)
    game_analyzer.model.detect_objects = MagicMock(
        return_value=[
            {"class_name": "play_call_screen", "bbox": [0, 0, 100, 100], "confidence": 0.95}
        ]
    )

    # Process play end frame
    state3 = game_analyzer.process_frame(mock_frame3, timestamp=4.0)
    assert state3.state_indicators["play_call_screen"] is True
    assert state3.play_active is False
    assert state3.play_duration == 2.0  # 4.0 - 2.0
    assert state3.last_playcall_time == 4.0


def test_play_state_edge_cases(game_analyzer):
    """Test edge cases in play state detection."""
    # Test case: Pre-play indicator flickers
    mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Pre-play appears
    game_analyzer.model.detect_objects = MagicMock(
        return_value=[
            {"class_name": "preplay_indicator", "bbox": [0, 0, 100, 100], "confidence": 0.95}
        ]
    )
    state1 = game_analyzer.process_frame(mock_frame, timestamp=1.0)

    # Pre-play briefly disappears (should not trigger play)
    game_analyzer.model.detect_objects = MagicMock(return_value=[])
    state2 = game_analyzer.process_frame(mock_frame, timestamp=1.1)

    # Pre-play reappears quickly
    game_analyzer.model.detect_objects = MagicMock(
        return_value=[
            {"class_name": "preplay_indicator", "bbox": [0, 0, 100, 100], "confidence": 0.95}
        ]
    )
    state3 = game_analyzer.process_frame(mock_frame, timestamp=1.2)

    # Verify no false play detection
    assert state1.play_active is False
    assert state2.play_active is False
    assert state3.play_active is False
    assert state3.play_count == 0  # No play should be counted


def test_clip_timing_validation(game_analyzer):
    """Test the clip timing validation system."""
    # Create mock frames
    mock_frames = []
    mock_timestamps = []
    frame_count = 180  # 6 seconds at 30fps

    # Generate test frames and timestamps
    for i in range(frame_count):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_frames.append(frame)
        mock_timestamps.append(i / 30.0)  # 30fps

    game_analyzer.frame_buffer = mock_frames
    game_analyzer.frame_timestamps = mock_timestamps

    # Test Case 1: Normal Play Sequence
    current_state = GameState()
    current_state.state_indicators["preplay_indicator"] = False
    current_state.last_preplay_time = 1.0
    current_state.play_active = True

    timing = game_analyzer.validate_clip_timing(current_state, 90)  # Frame 90 (3 seconds)
    assert timing["is_valid"] == True
    assert len(timing["validation_methods"]) >= 2
    assert timing["start_frame"] is not None
    assert timing["end_frame"] is not None

    # Test Case 2: Invalid Duration
    timing = game_analyzer.validate_clip_timing(current_state, 10)  # Too short
    assert timing["is_valid"] == False

    # Test Case 3: Multiple Method Agreement
    current_state.current_formation = "Shotgun"
    current_state.previous_formation = "I-Form"
    current_state.down = 1
    current_state.distance = 10

    timing = game_analyzer.validate_clip_timing(current_state, 90)
    assert len(timing["validation_methods"]) >= 3  # At least 3 methods should agree

    # Test Case 4: Buffer Ranges
    clip = game_analyzer.create_clip(30, 90)  # 2 second clip
    assert clip is not None
    assert clip["duration"] >= game_analyzer.clip_config["min_play_duration"]
    assert clip["duration"] <= game_analyzer.clip_config["max_play_duration"]
    assert len(clip["frames"]) > 0

    # Test Case 5: Edge Cases
    clip = game_analyzer.create_clip(0, 10)  # Too short
    assert clip is None

    clip = game_analyzer.create_clip(0, 600)  # Too long
    assert clip is None


def test_hardware_tier_configs():
    """Test hardware tier configurations match PRD specs."""
    assert HARDWARE_TIERS["ULTRA_LOW"]["img_size"] == 320
    assert HARDWARE_TIERS["ULTRA_LOW"]["batch_size"] == 1

    assert HARDWARE_TIERS["LOW"]["img_size"] == 416
    assert HARDWARE_TIERS["LOW"]["batch_size"] == 2

    assert HARDWARE_TIERS["MEDIUM"]["img_size"] == 640
    assert HARDWARE_TIERS["MEDIUM"]["batch_size"] == 4

    assert HARDWARE_TIERS["HIGH"]["img_size"] == 832
    assert HARDWARE_TIERS["HIGH"]["batch_size"] == 8

    assert HARDWARE_TIERS["ULTRA"]["img_size"] == 1280
    assert HARDWARE_TIERS["ULTRA"]["batch_size"] == 16


def test_ui_classes():
    """Test UI classes match PRD requirements."""
    expected_classes = [
        "hud",
        "possession_triangle_area",
        "territory_triangle_area",
        "preplay_indicator",
        "play_call_screen",
    ]
    assert UI_CLASSES == expected_classes


def test_triangle_validation():
    """Test triangle validation with geometric checks."""
    analyzer = EnhancedGameAnalyzer()
    validation = TriangleValidation()

    # Create a perfect equilateral triangle
    height = 100
    base = 100
    points = np.array([[[50, 0]], [[0, height]], [[base, height]]], dtype=np.int32)

    is_valid, confidence = analyzer.validate_triangle(points, validation)
    assert is_valid
    assert confidence > 0.9  # High confidence for perfect triangle


def test_game_state_extraction():
    """Test game state extraction from frame."""
    analyzer = EnhancedGameAnalyzer()

    # Create a mock frame with HUD elements
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Mock detection results
    mock_results = {
        "hud": [(100, 100, 500, 150)],
        "possession_triangle_area": [(50, 120, 70, 140)],
        "territory_triangle_area": [(510, 120, 530, 140)],
    }

    with patch.object(analyzer, "_process_detections", return_value=mock_results):
        game_state = analyzer.analyze_frame(frame)

        assert isinstance(game_state, GameState)
        assert game_state.confidence > 0


def test_hardware_adaptation():
    """Test hardware-adaptive processing."""
    # Test for each hardware tier
    for tier in HardwareTier:
        hardware = MagicMock(spec=HardwareDetector)
        hardware.tier = tier

        analyzer = EnhancedGameAnalyzer(hardware=hardware)
        config = HARDWARE_TIERS[tier.name]

        assert analyzer.model.img_size == config["img_size"]
        assert analyzer.model.batch_size == config["batch_size"]


def test_error_handling():
    """Test error handling and recovery."""
    analyzer = EnhancedGameAnalyzer()

    # Test short gap handling
    with patch.object(analyzer, "_process_detections", side_effect=Exception("Test error")):
        game_state = analyzer.analyze_frame(np.zeros((720, 1280, 3)))
        assert game_state.confidence < 1.0  # Reduced confidence due to error


def test_state_persistence():
    """Test state persistence through gaps."""
    analyzer = EnhancedGameAnalyzer()

    # Create initial state
    initial_state = GameState(down=1, distance=10, yard_line=20, confidence=1.0)

    # Simulate gap
    with patch.object(analyzer, "_process_detections", return_value={}):
        new_state = analyzer.analyze_frame(np.zeros((720, 1280, 3)))

        # Should maintain some state with reduced confidence
        assert new_state.down == initial_state.down
        assert new_state.confidence < initial_state.confidence


def test_validation_methods():
    """Test validation methods for game state."""
    analyzer = EnhancedGameAnalyzer()

    # Test sequence validation
    valid_sequence = {"preplay_indicator": True, "play_call_screen": False, "hud_visible": True}

    assert analyzer._validate_ui_sequence(valid_sequence, 0)


def test_confidence_scoring():
    """Test confidence scoring system."""
    analyzer = EnhancedGameAnalyzer()

    # Test perfect detection
    perfect_detections = {
        "hud": [(100, 100, 500, 150, 0.95)],
        "possession_triangle_area": [(50, 120, 70, 140, 0.90)],
        "territory_triangle_area": [(510, 120, 530, 140, 0.92)],
    }

    confidence = analyzer._calculate_confidence(perfect_detections)
    assert confidence > 0.9  # High confidence for good detections


def test_occlusion_handling():
    """Test handling of partial HUD occlusion."""
    analyzer = EnhancedGameAnalyzer()

    # Test with partially occluded HUD
    occluded_regions = {
        "top_left": True,
        "top_right": False,
        "bottom_left": True,
        "bottom_right": True,
    }

    # Should still function with partial visibility
    assert analyzer.validate_partial_hud(occluded_regions)
