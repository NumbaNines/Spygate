import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PyQt6.QtWidgets import QWidget

from .tutorial_system import TutorialStep


@dataclass
class TutorialInfo:
    id: str
    title: str
    description: str
    steps: list[dict]
    required_features: list[str]
    skill_level: str
    completion_reward: Optional[str] = None


class TutorialManager:
    """Manages tutorial content and user progress"""

    def __init__(self):
        self.tutorials: dict[str, TutorialInfo] = {}
        self.completed_tutorials: dict[str, bool] = {}
        self.current_tutorial: Optional[str] = None
        self.tutorial_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "tutorials"
        )

        # Ensure tutorials directory exists
        os.makedirs(self.tutorial_path, exist_ok=True)

        # Load available tutorials
        self.load_tutorials()

        # Load user progress
        self.load_progress()

    def load_tutorials(self):
        """Load all available tutorials from the tutorials directory"""
        try:
            # Load the main tutorials index
            index_path = os.path.join(self.tutorial_path, "index.json")
            if os.path.exists(index_path):
                with open(index_path) as f:
                    index = json.load(f)

                # Load each tutorial
                for tutorial_id in index["tutorials"]:
                    tutorial_path = os.path.join(self.tutorial_path, f"{tutorial_id}.json")
                    if os.path.exists(tutorial_path):
                        with open(tutorial_path) as f:
                            data = json.load(f)
                            self.tutorials[tutorial_id] = TutorialInfo(
                                id=tutorial_id,
                                title=data["title"],
                                description=data["description"],
                                steps=data["steps"],
                                required_features=data.get("required_features", []),
                                skill_level=data.get("skill_level", "beginner"),
                                completion_reward=data.get("completion_reward"),
                            )
        except Exception as e:
            print(f"Error loading tutorials: {e}")

    def load_progress(self):
        """Load user's tutorial progress"""
        progress_path = os.path.join(self.tutorial_path, "progress.json")
        try:
            if os.path.exists(progress_path):
                with open(progress_path) as f:
                    self.completed_tutorials = json.load(f)
        except Exception as e:
            print(f"Error loading tutorial progress: {e}")

    def save_progress(self):
        """Save user's tutorial progress"""
        progress_path = os.path.join(self.tutorial_path, "progress.json")
        try:
            with open(progress_path, "w") as f:
                json.dump(self.completed_tutorials, f)
        except Exception as e:
            print(f"Error saving tutorial progress: {e}")

    def get_available_tutorials(self) -> list[TutorialInfo]:
        """Get list of available tutorials"""
        return list(self.tutorials.values())

    def get_tutorial(self, tutorial_id: str) -> Optional[TutorialInfo]:
        """Get a specific tutorial by ID"""
        return self.tutorials.get(tutorial_id)

    def is_tutorial_completed(self, tutorial_id: str) -> bool:
        """Check if a tutorial has been completed"""
        return self.completed_tutorials.get(tutorial_id, False)

    def mark_tutorial_completed(self, tutorial_id: str):
        """Mark a tutorial as completed"""
        self.completed_tutorials[tutorial_id] = True
        self.save_progress()

    def get_next_tutorial(self) -> Optional[TutorialInfo]:
        """Get the next recommended tutorial based on user progress"""
        for tutorial in self.tutorials.values():
            if not self.is_tutorial_completed(tutorial.id):
                # Check if all required features are available
                if all(
                    self.check_feature_available(feature) for feature in tutorial.required_features
                ):
                    return tutorial
        return None

    def check_feature_available(self, feature: str) -> bool:
        """Check if a required feature is available"""
        # TODO: Implement feature availability checking
        # This should check if the required feature is actually implemented and available
        return True

    def create_tutorial_steps(self, tutorial_id: str, main_window) -> list[TutorialStep]:
        """Create TutorialStep objects for a tutorial"""
        tutorial = self.get_tutorial(tutorial_id)
        if not tutorial:
            return []

        steps = []
        for step_data in tutorial.steps:
            # Find target widget if specified
            target_widget = None
            if "target_widget" in step_data:
                target_widget = self.find_widget_by_name(main_window, step_data["target_widget"])

            step = TutorialStep(
                title=step_data["title"],
                description=step_data["description"],
                target_widget=target_widget,
                highlight_rect=step_data.get("highlight_rect"),
            )
            steps.append(step)

        return steps

    def find_widget_by_name(self, parent, widget_name: str):
        """Find a widget by its object name in the widget hierarchy"""
        if parent.objectName() == widget_name:
            return parent

        for child in parent.findChildren(QWidget):
            if child.objectName() == widget_name:
                return child

        return None
