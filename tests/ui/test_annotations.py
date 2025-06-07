"""Tests for annotation-related components."""

import os
from datetime import datetime
from uuid import UUID

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton

from src.ui.components.annotation_display import AnnotationDisplay, AnnotationMarker
from src.ui.components.annotation_tool import AnnotationTool
from src.ui.models.annotation import Annotation


@pytest.fixture
def app():
    """Create a QApplication instance."""
    return QApplication([])


@pytest.fixture
def annotation():
    """Create a sample annotation."""
    return Annotation(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        timestamp=5.0,
        text="Test annotation",
        duration=3.0,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        player_name="Test Player",
        color="#3B82F6",
    )


@pytest.fixture
def annotation_tool(app):
    """Create an AnnotationTool instance."""
    return AnnotationTool(current_time=5.0, player_name="Test Player")


@pytest.fixture
def annotation_display(app):
    """Create an AnnotationDisplay instance."""
    return AnnotationDisplay(video_duration=60.0)


def test_annotation_creation():
    """Test creating an annotation."""
    annotation = Annotation.create(
        timestamp=5.0, text="Test annotation", duration=3.0, player_name="Test Player"
    )
    assert isinstance(annotation.id, UUID)
    assert annotation.timestamp == 5.0
    assert annotation.text == "Test annotation"
    assert annotation.duration == 3.0
    assert annotation.player_name == "Test Player"
    assert annotation.color == "#3B82F6"


def test_annotation_tool_creation(annotation_tool):
    """Test AnnotationTool initialization."""
    assert annotation_tool.current_time == 5.0
    assert annotation_tool.player_name == "Test Player"
    assert annotation_tool.text_input is not None
    assert annotation_tool.duration_input is not None
    assert annotation_tool.color_button is not None


def test_annotation_tool_create_annotation(annotation_tool, qtbot):
    """Test creating an annotation with the tool."""
    # Type annotation text
    qtbot.keyClicks(annotation_tool.text_input, "Test annotation")

    # Set duration
    annotation_tool.duration_input.setValue(3)

    # Create annotation
    with qtbot.waitSignal(annotation_tool.annotationCreated) as signal:
        qtbot.mouseClick(
            annotation_tool.findChild(QPushButton, "Create"), Qt.MouseButton.LeftButton
        )

    # Verify signal
    annotation = signal.args[0]
    assert annotation.text == "Test annotation"
    assert annotation.duration == 3.0
    assert annotation.timestamp == 5.0
    assert annotation.player_name == "Test Player"


def test_annotation_display_add_annotation(annotation_display, annotation):
    """Test adding an annotation to the display."""
    annotation_display.add_annotation(annotation)
    assert str(annotation.id) in annotation_display.annotations
    assert str(annotation.id) in annotation_display.markers


def test_annotation_display_remove_annotation(annotation_display, annotation):
    """Test removing an annotation from the display."""
    annotation_display.add_annotation(annotation)
    annotation_display.remove_annotation(annotation)
    assert str(annotation.id) not in annotation_display.annotations
    assert str(annotation.id) not in annotation_display.markers


def test_annotation_marker_click(annotation_display, annotation, qtbot):
    """Test clicking an annotation marker."""
    annotation_display.add_annotation(annotation)
    marker = annotation_display.markers[str(annotation.id)]

    with qtbot.waitSignal(marker.clicked) as signal:
        qtbot.mouseClick(marker, Qt.MouseButton.LeftButton)

    assert signal.args[0] == annotation


def test_annotation_display_update_positions(annotation_display, annotation):
    """Test updating marker positions on resize."""
    annotation_display.resize(200, 20)  # Initial size
    annotation_display.add_annotation(annotation)

    # Get initial position
    marker = annotation_display.markers[str(annotation.id)]
    initial_x = marker.x()

    # Resize and check new position
    annotation_display.resize(400, 20)
    annotation_display.update_marker_positions()
    assert marker.x() != initial_x  # Position should change with width
