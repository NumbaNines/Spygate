import os
import json
import pytest
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
from PyQt6.QtCore import Qt

from spygate.ui.components.tutorial_system import TutorialSystem, TutorialStep
from spygate.ui.components.tutorial_manager import TutorialManager

@pytest.fixture
def app():
    return QApplication([])

@pytest.fixture
def main_window(app):
    window = QMainWindow()
    
    # Add some test widgets
    toolbar = QWidget(window)
    toolbar.setObjectName("main_toolbar")
    
    import_btn = QPushButton("Import", window)
    import_btn.setObjectName("import_button")
    
    timeline = QWidget(window)
    timeline.setObjectName("timeline_view")
    
    analysis = QWidget(window)
    analysis.setObjectName("analysis_panel")
    
    help_btn = QPushButton("Help", window)
    help_btn.setObjectName("help_button")
    
    return window

@pytest.fixture
def tutorial_system(main_window):
    return TutorialSystem(main_window)

@pytest.fixture
def tutorial_manager(tmp_path):
    # Create a temporary tutorials directory
    tutorials_dir = tmp_path / "tutorials"
    tutorials_dir.mkdir()
    
    # Create test tutorial files
    index = {
        "tutorials": ["test_tutorial"],
        "version": "1.0"
    }
    with open(tutorials_dir / "index.json", "w") as f:
        json.dump(index, f)
    
    tutorial = {
        "id": "test_tutorial",
        "title": "Test Tutorial",
        "description": "A test tutorial",
        "skill_level": "beginner",
        "required_features": [],
        "steps": [
            {
                "title": "Step 1",
                "description": "First step",
                "target_widget": "import_button"
            },
            {
                "title": "Step 2",
                "description": "Second step",
                "target_widget": None
            }
        ]
    }
    with open(tutorials_dir / "test_tutorial.json", "w") as f:
        json.dump(tutorial, f)
    
    # Create manager and override tutorial path
    manager = TutorialManager()
    manager.tutorial_path = str(tutorials_dir)
    manager.load_tutorials()
    
    return manager

def test_tutorial_step_creation():
    step = TutorialStep("Test", "Description", None)
    assert step.title == "Test"
    assert step.description == "Description"
    assert step.target_widget is None

def test_tutorial_system_initialization(tutorial_system):
    assert tutorial_system.main_window is not None
    assert tutorial_system.current_step == 0
    assert len(tutorial_system.steps) == 0

def test_tutorial_system_navigation(tutorial_system):
    steps = [
        TutorialStep("Step 1", "First step"),
        TutorialStep("Step 2", "Second step"),
        TutorialStep("Step 3", "Third step")
    ]
    
    tutorial_system.start_tutorial("test", steps)
    assert tutorial_system.current_step == 0
    
    tutorial_system.next_step()
    assert tutorial_system.current_step == 1
    
    tutorial_system.previous_step()
    assert tutorial_system.current_step == 0
    
    # Test boundaries
    tutorial_system.previous_step()  # Should stay at 0
    assert tutorial_system.current_step == 0
    
    tutorial_system.current_step = len(steps) - 1
    tutorial_system.next_step()  # Should complete tutorial
    assert tutorial_system.current_tutorial is None

def test_tutorial_manager_loading(tutorial_manager):
    tutorials = tutorial_manager.get_available_tutorials()
    assert len(tutorials) == 1
    assert tutorials[0].id == "test_tutorial"

def test_tutorial_manager_progress(tutorial_manager):
    tutorial_id = "test_tutorial"
    assert not tutorial_manager.is_tutorial_completed(tutorial_id)
    
    tutorial_manager.mark_tutorial_completed(tutorial_id)
    assert tutorial_manager.is_tutorial_completed(tutorial_id)
    
    # Verify progress was saved
    progress_path = os.path.join(tutorial_manager.tutorial_path, "progress.json")
    with open(progress_path, "r") as f:
        progress = json.load(f)
        assert progress[tutorial_id] is True

def test_tutorial_step_creation_with_widgets(tutorial_manager, main_window):
    steps = tutorial_manager.create_tutorial_steps("test_tutorial", main_window)
    assert len(steps) == 2
    
    # First step should have target widget
    assert steps[0].target_widget is not None
    assert steps[0].target_widget.objectName() == "import_button"
    
    # Second step should not have target widget
    assert steps[1].target_widget is None

def test_tutorial_system_cleanup(tutorial_system):
    steps = [TutorialStep("Test", "Description")]
    tutorial_system.start_tutorial("test", steps)
    
    assert tutorial_system.overlay is not None
    assert tutorial_system.popup is not None
    
    tutorial_system.cleanup()
    assert tutorial_system.overlay is None
    assert tutorial_system.popup is None