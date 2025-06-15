#!/usr/bin/env python3
"""Test script to verify clip boundary detection fixes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from spygate.ml.enhanced_game_analyzer import GameState, SituationContext
from spygate_desktop_app_faceit_style import AnalysisWorker


def test_play_boundary_detection():
    """Test that play boundaries are properly detected."""
    print("=" * 60)
    print("TESTING PLAY BOUNDARY DETECTION")
    print("=" * 60)

    # Create a dummy worker
    worker = AnalysisWorker("dummy.mp4")

    # Initialize play boundary state
    worker.play_boundary_state = {
        "last_down": None,
        "last_distance": None,
        "last_yard_line": None,
        "last_possession_team": None,
        "last_game_clock": None,
        "active_play_frame": None,
        "active_play_data": None,
    }

    # Test Case 1: Initial play detection
    print("\nTest 1: Initial Play Detection")
    game_state1 = GameState()
    game_state1.down = 1
    game_state1.distance = 10
    game_state1.yard_line = 25

    boundary1 = worker._analyze_play_boundaries(game_state1, 100)
    print(f"Play Started: {boundary1['play_started']}")
    print(f"Clip Should Start: {boundary1['clip_should_start']}")
    print(f"Play Type: {boundary1['play_type']}")
    assert boundary1["play_started"] == True
    assert boundary1["clip_should_start"] == True

    # Test Case 2: Play in progress (no change)
    print("\nTest 2: Play In Progress")
    boundary2 = worker._analyze_play_boundaries(game_state1, 150)
    print(f"Play Started: {boundary2['play_started']}")
    print(f"Play Ended: {boundary2['play_ended']}")
    assert boundary2["play_started"] == False
    assert boundary2["play_ended"] == False

    # Test Case 3: Down change (play ends, new play starts)
    print("\nTest 3: Down Change Detection")
    game_state2 = GameState()
    game_state2.down = 2
    game_state2.distance = 7
    game_state2.yard_line = 28

    boundary3 = worker._analyze_play_boundaries(game_state2, 200)
    print(f"Play Ended: {boundary3['play_ended']}")
    print(f"Clip Should End: {boundary3['clip_should_end']}")
    assert boundary3["play_ended"] == True
    assert boundary3["clip_should_end"] == True

    # Test Case 4: First down achieved
    print("\nTest 4: First Down Detection")
    game_state3 = GameState()
    game_state3.down = 1
    game_state3.distance = 10
    game_state3.yard_line = 40

    # Reset state for new play
    worker.play_boundary_state["active_play_frame"] = None
    boundary4 = worker._analyze_play_boundaries(game_state3, 250)
    print(f"Play Started: {boundary4['play_started']}")
    print(f"Play Type: {boundary4['play_type']}")
    assert boundary4["play_started"] == True

    print("\n✅ All play boundary tests passed!")


def test_clip_duration():
    """Test that clips have appropriate durations."""
    print("\n" + "=" * 60)
    print("TESTING CLIP DURATION LIMITS")
    print("=" * 60)

    worker = AnalysisWorker("dummy.mp4")

    # Test natural boundaries
    game_state = GameState()
    game_state.down = 3
    game_state.distance = 8

    fps = 30
    frame_number = 1000

    start, end = worker._find_natural_clip_boundaries(frame_number, fps, game_state)
    duration_seconds = (end - start) / fps

    print(f"Clip Duration: {duration_seconds:.1f} seconds")
    print(f"Start Frame: {start}, End Frame: {end}")

    # Check constraints
    assert duration_seconds >= 3.0, "Clip too short!"
    assert duration_seconds <= 15.0, "Clip too long!"

    print("✅ Clip duration within acceptable range!")


if __name__ == "__main__":
    test_play_boundary_detection()
    test_clip_duration()
    print("\n🎉 All tests passed! Clip boundary detection is working correctly.")
