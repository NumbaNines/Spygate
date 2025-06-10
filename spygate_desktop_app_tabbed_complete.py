#!/usr/bin/env python3
"""
SpygateAI Desktop Application - Complete Tabbed Interface
========================================================

Professional tabbed interface with YouTube-style video processing,
gameplan management, and video preview functionality.
"""

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "spygate"))

try:
    import cv2
    import numpy as np
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class DetectedClip:
    """Represents a detected clip with metadata."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    situation: str
    thumbnail_path: Optional[str] = None
    approved: Optional[bool] = None


# ==================== VIDEO PREVIEW DIALOG ====================


class VideoPreviewDialog(QDialog):
    """Dialog for previewing video clips."""

    def __init__(self, video_path, clip, clip_number, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.clip = clip
        self.clip_number = clip_number
        self.setWindowTitle(f"Preview Clip {clip_number}: {clip.situation}")
        self.setModal(True)
        self.resize(800, 600)

        # Try to import video widgets
        try:
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PyQt6.QtMultimediaWidgets import QVideoWidget

            self.has_multimedia = True
        except ImportError:
            self.has_multimedia = False

        self.setup_ui()

    def setup_ui(self):
        """Set up the preview dialog UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"Clip {self.clip_number}: {self.clip.situation}")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
        """
        )
        layout.addWidget(title)

        if self.has_multimedia:
            # Re-import inside the conditional to make sure classes are available
            from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PyQt6.QtMultimediaWidgets import QVideoWidget

            # Video widget
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(400)
            self.video_widget.setStyleSheet(
                """
                QVideoWidget {
                    background-color: black;
                    border: 2px solid #1ce783;
                    border-radius: 8px;
                }
            """
            )
            layout.addWidget(self.video_widget)

            # Media player
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            self.media_player.setVideoOutput(self.video_widget)

            # Controls
            controls_layout = QHBoxLayout()

            play_btn = QPushButton("▶️ Play")
            play_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #1ce783;
                    color: #e3e3e3;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    font-family: "Minork Sans", sans-serif;
                }
                QPushButton:hover { background-color: #17d474; }
            """
            )
            play_btn.clicked.connect(self.play_clip)
            controls_layout.addWidget(play_btn)

            layout.addLayout(controls_layout)

        else:
            # Fallback for when multimedia is not available
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)

            fallback_label = QLabel(
                """
                🎥 Video Preview Not Available

                PyQt6 multimedia components are not installed.

                Clip Information:
            """
            )
            fallback_label.setStyleSheet(
                """
                QLabel {
                    color: #ccc;
                    font-size: 14px;
                    padding: 20px;
                    background-color: #0b0c0f;
                    border: 2px dashed #666;
                    border-radius: 8px;
                    text-align: center;
                    font-family: "Minork Sans", sans-serif;
                }
            """
            )
            fallback_label.setWordWrap(True)
            fallback_layout.addWidget(fallback_label)

            # Buttons layout
            buttons_layout = QHBoxLayout()

            # External player button
            external_play_btn = QPushButton("🎬 Open in Video Player")
            external_play_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #1ce783;
                    color: #e3e3e3;
                    padding: 15px 30px;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 16px;
                    margin: 10px;
                    font-family: "Minork Sans", sans-serif;
                }
                QPushButton:hover { background-color: #17d474; }
            """
            )
            external_play_btn.clicked.connect(self.open_external_player)
            buttons_layout.addWidget(external_play_btn)

            # Seek instructions button
            seek_btn = QPushButton("📍 Show Seek Instructions")
            seek_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #666;
                    color: #e3e3e3;
                    padding: 15px 30px;
                    border: none;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 16px;
                    margin: 10px;
                    font-family: "Minork Sans", sans-serif;
                }
                QPushButton:hover { background-color: #777; }
            """
            )
            seek_btn.clicked.connect(self.show_seek_instructions)
            buttons_layout.addWidget(seek_btn)

            fallback_layout.addLayout(buttons_layout)
            layout.addWidget(fallback_widget)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #666;
                color: #e3e3e3;
                padding: 10px 30px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #777; }
        """
        )
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def play_clip(self):
        """Play the video clip."""
        if self.has_multimedia:
            from PyQt6.QtCore import QUrl

            self.media_player.setSource(QUrl.fromLocalFile(self.video_path))
            self.media_player.play()

    def open_external_player(self):
        """Open the video in the system's default video player."""
        import os
        import subprocess

        try:
            if os.name == "nt":  # Windows
                subprocess.run(["start", self.video_path], shell=True, check=True)
                self.show_seek_instructions()
            else:
                subprocess.run(
                    ["open" if sys.platform == "darwin" else "xdg-open", self.video_path]
                )
                self.show_seek_instructions()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", f"Could not open video player: {str(e)}")

    def show_seek_instructions(self):
        """Show detailed seek instructions for manual viewing."""
        from PyQt6.QtWidgets import QMessageBox

        start_time = f"{int(self.clip.start_time // 60):02d}:{int(self.clip.start_time % 60):02d}"
        end_time = f"{int(self.clip.end_time // 60):02d}:{int(self.clip.end_time % 60):02d}"
        duration = f"{self.clip.end_time - self.clip.start_time:.1f}"

        message = f"""🎯 Manual Seek Instructions

Clip: {self.clip.situation}
📍 Start Time: {start_time}
📍 End Time: {end_time}
⏱️ Duration: {duration} seconds

To view this clip:
1. Seek to {start_time} in your video player
2. Watch for approximately {duration} seconds
3. The key moment should occur around this timeframe
"""

        QMessageBox.information(self, "Seek Instructions", message)


# ==================== GAMEPLAN DIALOGS ====================


class GameplanCreationDialog(QDialog):
    """Dialog for creating new gameplans with type selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Gameplan")
        self.setModal(True)
        self.resize(500, 400)
        self.gameplan_data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎯 Create New Gameplan")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 0;
        """
        )
        layout.addWidget(title)

        # Gameplan name
        name_label = QLabel("Gameplan Name:")
        name_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-weight: bold;
        """
        )
        layout.addWidget(name_label)

        self.name_input = QLineEdit()
        self.name_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QLineEdit:focus { border-color: #1ce783; }
        """
        )
        layout.addWidget(self.name_input)

        # Gameplan type
        type_label = QLabel("Gameplan Type:")
        type_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-weight: bold;
        """
        )
        layout.addWidget(type_label)

        self.type_combo = QComboBox()
        self.type_combo.addItems(
            [
                "🎯 Opponent-Specific",
                "🛡️ Formation Counter",
                "📊 Situation-Based",
                "🏆 Tournament Prep",
                "📋 General Strategy",
                "⚡ Custom Category",
            ]
        )
        self.type_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QComboBox:focus { border-color: #1ce783; }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #0b0c0f;
                color: #ffffff;
                selection-background-color: #1ce783;
            }
        """
        )
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        layout.addWidget(self.type_combo)

        # Target field (shows/hides based on type)
        self.target_label = QLabel("Target/Opponent:")
        self.target_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-weight: bold;
        """
        )
        self.target_label.hide()
        layout.addWidget(self.target_label)

        self.target_input = QLineEdit()
        self.target_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QLineEdit:focus { border-color: #1ce783; }
        """
        )
        self.target_input.hide()
        layout.addWidget(self.target_input)

        # Description
        desc_label = QLabel("Description (optional):")
        desc_label.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
        """
        )
        layout.addWidget(desc_label)

        self.desc_input = QTextEdit()
        self.desc_input.setMaximumHeight(80)
        self.desc_input.setStyleSheet(
            """
            QTextEdit {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QTextEdit:focus { border-color: #1ce783; }
        """
        )
        layout.addWidget(self.desc_input)

        # Buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #666;
                color: #e3e3e3;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #777; }
        """
        )
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create Gameplan")
        create_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        create_btn.clicked.connect(self.create_gameplan)
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

        # Set initial state
        self.on_type_changed(self.type_combo.currentText())

    def on_type_changed(self, category_type):
        """Show/hide target field based on gameplan type."""
        needs_target = category_type in [
            "🎯 Opponent-Specific",
            "🛡️ Formation Counter",
            "⚡ Custom Category",
        ]

        self.target_label.setVisible(needs_target)
        self.target_input.setVisible(needs_target)

        if category_type == "🎯 Opponent-Specific":
            self.target_label.setText("Opponent Username:")
            self.target_input.setPlaceholderText("e.g., ProPlayer123")
        elif category_type == "🛡️ Formation Counter":
            self.target_label.setText("Formation to Counter:")
            self.target_input.setPlaceholderText("e.g., Shotgun Bunch")
        elif category_type == "⚡ Custom Category":
            self.target_label.setText("Category Name:")
            self.target_input.setPlaceholderText("e.g., My Custom Strategy")

    def create_gameplan(self):
        """Validate and create the gameplan."""
        name = self.name_input.text().strip()
        if not name:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", "Please enter a gameplan name.")
            return

        category_type = self.type_combo.currentText()
        target = self.target_input.text().strip() if self.target_input.isVisible() else None

        # Validate target field if required
        if self.target_input.isVisible() and not target:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", "Please enter the required target/opponent field.")
            return

        # Create gameplan data
        self.gameplan_data = {
            "name": name,
            "category_type": category_type.split(" ", 1)[1],  # Remove emoji
            "target": target,
            "description": self.desc_input.toPlainText().strip(),
            "full_type": category_type,
        }

        print(f"🎯 Dialog validation passed! Accepting dialog.")
        self.accept()


class GameplanManagerDialog(QDialog):
    """Dialog for managing existing gameplans."""

    def __init__(self, categories, parent=None):
        super().__init__(parent)
        self.categories = categories
        self.deleted_categories = []
        self.setWindowTitle("Manage Gameplans")
        self.setModal(True)
        self.resize(400, 500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🗑️ Manage Gameplans")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 0;
        """
        )
        layout.addWidget(title)

        # Category list
        self.category_list = QListWidget()
        self.category_list.setStyleSheet(
            """
            QListWidget {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 6px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background-color: #1ce783;
            }
        """
        )

        # Populate list
        for category in self.categories:
            self.category_list.addItem(category)

        layout.addWidget(self.category_list)

        # Buttons
        button_layout = QHBoxLayout()

        delete_btn = QPushButton("Delete Selected")
        delete_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: #e3e3e3;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """
        )
        delete_btn.clicked.connect(self.delete_selected)
        button_layout.addWidget(delete_btn)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #666;
                color: #e3e3e3;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #777; }
        """
        )
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def delete_selected(self):
        """Delete the selected category."""
        current_item = self.category_list.currentItem()
        if current_item:
            category_text = current_item.text()
            self.deleted_categories.append(category_text)

            row = self.category_list.row(current_item)
            self.category_list.takeItem(row)

            print(f"🗑️ Deleted category: {category_text}")


# ==================== MAIN TABBED APPLICATION ====================


class SpygateDesktopAppTabbed(QMainWindow):
    """Main tabbed desktop application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🏈 SpygateAI Desktop - Professional Analysis Tool")
        self.setMinimumSize(1200, 800)
        self.current_video_path = None

        # Apply dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0b0c0f;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
        """
        )

        self.init_ui()
        print("✅ Core imports successful")

    def init_ui(self):
        """Initialize the user interface."""
        # Central widget with FaceIt-style layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create left sidebar navigation
        self.create_left_sidebar(main_layout)

        # Create main content area
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #0b0c0f;")

        # Create tab content widgets
        self.analysis_widget = self.create_analysis_widget()
        self.gameplan_widget = self.create_gameplan_widget()

        # Add content widgets to stack
        self.content_stack.addWidget(self.analysis_widget)  # 0
        self.content_stack.addWidget(self.create_dashboard_widget())  # 1
        self.content_stack.addWidget(self.gameplan_widget)  # 2
        self.content_stack.addWidget(self.create_learn_widget())  # 3

        main_layout.addWidget(self.content_stack)

        # Set default to Analysis tab
        self.content_stack.setCurrentIndex(0)

    def create_left_sidebar(self, parent_layout):
        """Create the FaceIt-style left sidebar navigation."""
        # Left sidebar
        left_sidebar = QWidget()
        left_sidebar.setFixedWidth(250)
        left_sidebar.setStyleSheet(
            """
            QWidget {
                background-color: #0b0c0f;
                border-right: 1px solid #1a1a1a;
            }
        """
        )

        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Title area
        logo_widget = QWidget()
        logo_widget.setFixedHeight(80)
        logo_widget.setStyleSheet("background-color: #0b0c0f; border-bottom: 1px solid #1a1a1a;")
        
        logo_layout = QHBoxLayout(logo_widget)
        logo_layout.setContentsMargins(20, 0, 20, 0)
        
        logo_label = QLabel("🏈 SpygateAI")
        logo_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 18px;
            font-weight: bold;
        """
        )
        logo_layout.addWidget(logo_label)
        sidebar_layout.addWidget(logo_widget)

        # Navigation area
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(2)

        # Navigation items
        nav_items = [
            ("📊", "Analysis"),
            ("🏠", "Dashboard"), 
            ("🎯", "Gameplan"),
            ("📚", "Learn")
        ]

        # Store nav buttons for selection management
        self.nav_buttons = []
        
        for i, (icon, text) in enumerate(nav_items):
            nav_button = self.create_nav_button(icon, text, i)
            nav_layout.addWidget(nav_button)
            self.nav_buttons.append(nav_button)

        # Set Analysis as default selected
        self.nav_buttons[0].setChecked(True)

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_widget)

        parent_layout.addWidget(left_sidebar)

    def create_nav_button(self, icon, text, index):
        """Create a navigation button with FaceIt styling."""
        button = QPushButton(f"{icon}  {text}")
        button.setFixedHeight(45)
        button.setCheckable(True)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: #767676;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                text-align: left;
                padding-left: 20px;
                border: none;
                border-radius: 0px;
            }}
            QPushButton:hover {{
                background-color: #1a1a1a;
                color: #1ce783;
            }}
            QPushButton:pressed {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
            QPushButton:checked {{
                background-color: #1a1a1a;
                color: #ffffff;
            }}
        """
        )
        
        # Connect button click to handle selection and content switching
        button.clicked.connect(lambda: self.handle_nav_selection(button, index))
        return button

    def handle_nav_selection(self, selected_button, index):
        """Handle navigation tab selection - only one tab selected at a time."""
        for button in self.nav_buttons:
            button.setChecked(False)
        selected_button.setChecked(True)
        
        # Switch content
        self.content_stack.setCurrentIndex(index)

    def create_analysis_widget(self):
        """Create the analysis tab widget."""
        return AnalysisWidget(self)

    def create_dashboard_widget(self):
        """Create the dashboard tab widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header
        header = QLabel("🏠 Dashboard")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
        """
        )
        layout.addWidget(header)

        # Quick stats
        stats_text = QLabel("📈 Quick Stats Coming Soon...")
        stats_text.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 16px;
            padding: 20px;
        """
        )
        layout.addWidget(stats_text)

        layout.addStretch()
        return widget

    def create_gameplan_widget(self):
        """Create the gameplan tab widget."""
        return GameplanWidget(self)

    def create_learn_widget(self):
        """Create the learn tab widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header
        header = QLabel("📚 Learn")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 24px;
            font-weight: bold;
            padding: 20px;
        """
        )
        layout.addWidget(header)

        # Content
        content_text = QLabel("🎓 Learning features coming soon...")
        content_text.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 16px;
            padding: 20px;
        """
        )
        layout.addWidget(content_text)

        layout.addStretch()
        return widget


# ==================== ANALYSIS WIDGET ====================


class AnalysisWidget(QWidget):
    """Analysis Tab - Video processing and clip detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # YouTube-style content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(40, 40, 40, 40)
        self.content_layout.setSpacing(20)

        # Show initial upload state
        self.show_upload_state()

        layout.addWidget(self.content_area)

    def show_upload_state(self):
        """Show the YouTube-style upload interface."""
        # Clear content area
        self.clear_content_area()

        # Upload container (YouTube-style)
        upload_container = QWidget()
        upload_container.setMaximumWidth(600)
        upload_container.setStyleSheet(
            """
            QWidget {
                background-color: #0b0c0f;
                border-radius: 12px;
                padding: 40px;
            }
        """
        )

        container_layout = QVBoxLayout(upload_container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload icon (YouTube-style)
        upload_icon = QLabel("📤")
        upload_icon.setStyleSheet(
            """
            font-size: 48px;
            color: #1ce783;
            font-family: "Minork Sans", sans-serif;
        """
        )
        upload_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(upload_icon)

        # Title
        title = QLabel("Upload your Madden gameplay")
        title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 20px;
            font-weight: bold;
            margin: 20px 0;
        """
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Get AI-powered analysis of your key moments")
        subtitle.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 14px;
            margin-bottom: 30px;
        """
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle)

        # Upload button
        upload_btn = QPushButton("SELECT FILES")
        upload_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 15px 40px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover {
                background-color: #17d474;
            }
        """
        )
        upload_btn.clicked.connect(self.browse_file)
        container_layout.addWidget(upload_btn)

        # Drag and drop hint
        drag_hint = QLabel("or drag and drop video files here")
        drag_hint.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 12px;
            margin-top: 15px;
        """
        )
        drag_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(drag_hint)

        # Center the upload container
        center_layout = QHBoxLayout()
        center_layout.addStretch()
        center_layout.addWidget(upload_container)
        center_layout.addStretch()

        self.content_layout.addStretch()
        self.content_layout.addLayout(center_layout)
        self.content_layout.addStretch()

    def clear_content_area(self):
        """Clear all widgets from the content area."""
        # Stop any running timers first
        if hasattr(self, "progress_timer") and self.progress_timer.isActive():
            self.progress_timer.stop()

        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def browse_file(self):
        """Open file browser for video selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Madden Gameplay Video", "", "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv)"
        )

        if file_path:
            self.process_video_file(file_path)

    def process_video_file(self, file_path):
        """Process the selected video file."""
        print(f"🎬 Processing video: {file_path}")

        # Store current video path for preview functionality
        if self.parent_app:
            self.parent_app.current_video_path = file_path

        # Show processing interface
        self.show_video_interface(file_path)

        # Simulate analysis (in a real app, this would be actual video processing)
        QTimer.singleShot(3000, lambda: self.show_analysis_results(file_path))

    def show_video_interface(self, file_path):
        """Show video processing interface."""
        # Clear content area
        self.clear_content_area()

        # Video title section
        filename = file_path.split("/")[-1]
        video_title = QLabel(filename)
        video_title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 8px;
        """
        )
        self.content_layout.addWidget(video_title)

        # Stats line (YouTube-style)
        stats_line = QLabel("🎬 Analyzing gameplay • AI detection in progress • Please wait...")
        stats_line.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 14px;
            margin-bottom: 24px;
        """
        )
        self.content_layout.addWidget(stats_line)

        # Progress indicator
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #666;
                border-radius: 8px;
                text-align: center;
                background-color: #0b0c0f;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
            }
            QProgressBar::chunk {
                background-color: #1ce783;
                border-radius: 6px;
            }
        """
        )
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        # Animate progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(50)

        self.content_layout.addWidget(progress_container)
        self.content_layout.addStretch()

    def update_progress(self):
        """Update progress bar animation."""
        if hasattr(self, "progress_bar") and self.progress_bar is not None:
            try:
                current_value = self.progress_bar.value()
                if current_value < 100:
                    self.progress_bar.setValue(current_value + 2)
                else:
                    self.progress_timer.stop()
            except RuntimeError:
                # Progress bar has been deleted, stop timer
                self.progress_timer.stop()

    def show_analysis_results(self, file_path):
        """Show analysis results for the processed video."""
        # Create sample detected clips
        sample_clips = [
            DetectedClip(
                start_frame=5400,
                end_frame=5700,
                start_time=90.0,
                end_time=95.0,
                confidence=0.92,
                situation="3rd & 8 - Clutch Moment",
            ),
            DetectedClip(
                start_frame=15300,
                end_frame=15600,
                start_time=255.0,
                end_time=260.0,
                confidence=0.88,
                situation="Red Zone Offense",
            ),
            DetectedClip(
                start_frame=26400,
                end_frame=26700,
                start_time=440.0,
                end_time=445.0,
                confidence=0.85,
                situation="3rd & Long - Midfield",
            ),
        ]

        # Store current video path
        self.current_video_path = file_path

        # Show YouTube-style clips interface
        self.show_clips_interface(sample_clips, file_path)

    def show_clips_interface(self, clips, file_path):
        """Show YouTube-style clips interface."""
        # Clear content area
        self.clear_content_area()

        # Video title section
        filename = file_path.split("/")[-1]
        video_title = QLabel(filename)
        video_title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 8px;
        """
        )
        self.content_layout.addWidget(video_title)

        # Stats line (YouTube-style)
        stats_line = QLabel(f"{len(clips)} key moments found • 95.2% success rate • Just now")
        stats_line.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 14px;
            margin-bottom: 24px;
        """
        )
        self.content_layout.addWidget(stats_line)

        # Clips grid (YouTube-style)
        clips_scroll = QScrollArea()
        clips_scroll.setWidgetResizable(True)
        clips_scroll.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """
        )

        clips_widget = QWidget()
        clips_layout = QVBoxLayout(clips_widget)
        clips_layout.setSpacing(16)

        # Create clips
        for i, clip in enumerate(clips, 1):
            clip_card = self.create_youtube_clip_card(clip, i)
            clips_layout.addWidget(clip_card)

        # Bulk actions
        bulk_actions = self.create_bulk_actions()
        clips_layout.addWidget(bulk_actions)

        clips_scroll.setWidget(clips_widget)
        self.content_layout.addWidget(clips_scroll)

    def create_youtube_clip_card(self, clip, clip_number):
        """Create a YouTube-style clip card."""
        card = QWidget()
        card.setStyleSheet(
            """
            QWidget {
                background-color: #0b0c0f;
                border-radius: 8px;
                padding: 16px;
            }
            QWidget:hover {
                background-color: #1f1f1f;
            }
        """
        )

        layout = QHBoxLayout(card)

        # Thumbnail placeholder (YouTube-style)
        thumbnail = QLabel()
        thumbnail.setFixedSize(160, 90)
        thumbnail.setStyleSheet(
            """
            QLabel {
                background-color: #333;
                border-radius: 4px;
                border: 2px solid #1ce783;
            }
        """
        )

        # Thumbnail content
        thumb_layout = QVBoxLayout()
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumbnail.setLayout(thumb_layout)

        # Play icon
        play_icon = QLabel("▶️")
        play_icon.setStyleSheet(
            """
            font-size: 24px;
            color: #ffffff;
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 20px;
            padding: 8px;
        """
        )
        play_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        play_icon.mousePressEvent = lambda event: self.preview_clip(clip, clip_number)
        thumb_layout.addWidget(play_icon, alignment=Qt.AlignmentFlag.AlignCenter)

        # Duration label
        duration = f"{clip.end_time - clip.start_time:.1f}s"
        duration_label = QLabel(duration)
        duration_label.setStyleSheet(
            """
            QLabel {
                background-color: rgba(0, 0, 0, 0.8);
                color: #ffffff;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                font-family: "Minork Sans", sans-serif;
            }
        """
        )
        thumb_layout.addWidget(
            duration_label, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )

        layout.addWidget(thumbnail)

        # Clip info section
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(16, 0, 0, 0)

        # Clip title
        clip_title = QLabel(f"Clip {clip_number}: {clip.situation}")
        clip_title.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 4px;
        """
        )
        info_layout.addWidget(clip_title)

        # Clip details
        time_range = f"{clip.start_time:.1f}s - {clip.end_time:.1f}s"
        confidence_text = f"Confidence: {clip.confidence:.1%}"
        details = QLabel(f"⏱️ {time_range} • 🎯 {confidence_text}")
        details.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 13px;
            margin-bottom: 12px;
        """
        )
        info_layout.addWidget(details)

        # Action buttons (compact, inline)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        # Preview button (primary)
        preview_btn = QPushButton("👁️ Preview")
        preview_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 6px 12px;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover {
                background-color: #17d474;
            }
        """
        )
        preview_btn.clicked.connect(lambda: self.preview_clip(clip, clip_number))
        actions_layout.addWidget(preview_btn)

        # Mini approve/reject buttons
        approve_mini = QPushButton("✅")
        approve_mini.setFixedSize(30, 30)
        approve_mini.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(76, 175, 80, 0.9);
                color: #e3e3e3;
                border: none;
                border-radius: 15px;
                font-size: 14px;
                margin-left: 4px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(76, 175, 80, 1.0);
            }
        """
        )
        approve_mini.clicked.connect(lambda: self.approve_clip(clip, clip_number, card))
        actions_layout.addWidget(approve_mini)

        reject_mini = QPushButton("❌")
        reject_mini.setFixedSize(30, 30)
        reject_mini.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(244, 67, 54, 0.9);
                color: #e3e3e3;
                border: none;
                border-radius: 15px;
                font-size: 12px;
                margin-left: 4px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover {
                background-color: rgba(244, 67, 54, 1.0);
            }
        """
        )
        reject_mini.clicked.connect(lambda: self.reject_clip(clip, clip_number, card))
        actions_layout.addWidget(reject_mini)

        actions_layout.addStretch()
        info_layout.addLayout(actions_layout)

        layout.addLayout(info_layout)

        return card

    def create_bulk_actions(self):
        """Create bulk action buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 20, 0, 0)

        # Approve all
        approve_all_btn = QPushButton("✅ Approve All")
        approve_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #45a049; }
        """
        )
        layout.addWidget(approve_all_btn)

        # Reject all
        reject_all_btn = QPushButton("❌ Reject All")
        reject_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """
        )
        layout.addWidget(reject_all_btn)

        layout.addStretch()

        # Export approved
        export_btn = QPushButton("📁 Export Approved")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        layout.addWidget(export_btn)

        return widget

    def preview_clip(self, clip, clip_number):
        """Preview a specific clip."""
        print(
            f"👁️ Previewing Clip {clip_number}: {clip.situation} ({clip.start_time:.1f}s-{clip.end_time:.1f}s)"
        )

        # Create and show video preview dialog
        if self.parent_app and hasattr(self.parent_app, "current_video_path"):
            preview_dialog = VideoPreviewDialog(
                self.parent_app.current_video_path, clip, clip_number, self
            )
            preview_dialog.exec()
        else:
            # Fallback for when we don't have the current video path
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Preview",
                f"Clip {clip_number}: {clip.situation}\n"
                f"Time: {clip.start_time:.1f}s - {clip.end_time:.1f}s\n"
                f"Confidence: {clip.confidence:.1%}",
            )

    def approve_clip(self, clip, clip_number, widget):
        """Approve/keep a specific clip."""
        clip.approved = True
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #1b2e1b;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 16px;
            }
        """
        )
        print(f"✅ Approved Clip {clip_number}: {clip.situation}")

    def reject_clip(self, clip, clip_number, widget):
        """Reject/delete a specific clip."""
        clip.approved = False
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #2e1b1b;
                border: 2px solid #f44336;
                border-radius: 8px;
                padding: 16px;
            }
        """
        )
        print(f"❌ Rejected Clip {clip_number}: {clip.situation}")

    # Drag and Drop functionality
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle drop events."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Process the first video file
            video_extensions = (".mp4", ".avi", ".mov", ".wmv", ".mkv")
            for file in files:
                if file.lower().endswith(video_extensions):
                    self.process_video_file(file)
                    break


# ==================== GAMEPLAN WIDGET ====================


class GameplanWidget(QWidget):
    """Gameplan Tab - Strategy organization and opponent prep."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel("🎯 Gameplan Management")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 24px;
            font-weight: bold;
            padding: 20px 0;
        """
        )
        layout.addWidget(header)

        # Action buttons
        buttons_layout = QHBoxLayout()

        new_gameplan_btn = QPushButton("🎯 New Gameplan")
        new_gameplan_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        new_gameplan_btn.clicked.connect(self.create_new_gameplan)
        buttons_layout.addWidget(new_gameplan_btn)

        save_gameplan_btn = QPushButton("💾 Save Gameplan")
        save_gameplan_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #666;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #777; }
        """
        )
        save_gameplan_btn.clicked.connect(self.save_gameplan)
        buttons_layout.addWidget(save_gameplan_btn)

        export_btn = QPushButton("📁 Export")
        export_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #666;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #777; }
        """
        )
        export_btn.clicked.connect(self.export_gameplan)
        buttons_layout.addWidget(export_btn)

        buttons_layout.addStretch()

        manage_btn = QPushButton("🗑️ Manage")
        manage_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                font-family: "Minork Sans", sans-serif;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """
        )
        manage_btn.clicked.connect(self.manage_gameplans)
        buttons_layout.addWidget(manage_btn)

        layout.addLayout(buttons_layout)

        # Categories list
        categories_label = QLabel("📋 Your Gameplans")
        categories_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: "Minork Sans", sans-serif;
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
        """
        )
        layout.addWidget(categories_label)

        self.category_list = QListWidget()
        self.category_list.setStyleSheet(
            """
            QListWidget {
                background-color: #0b0c0f;
                border: 2px solid #666;
                border-radius: 8px;
                padding: 10px;
                color: #ffffff;
                font-family: "Minork Sans", sans-serif;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
                border-radius: 4px;
                margin: 2px 0;
            }
            QListWidget::item:hover {
                background-color: #1a1a1a;
            }
            QListWidget::item:selected {
                background-color: #1ce783;
                color: #000;
            }
        """
        )

        # Populate with sample data
        sample_categories = [
            "🎯 Opponent: ProPlayer123",
            "🎯 Opponent: TopGun99",
            "📊 By Situation",
            "├── 3rd & Long",
            "├── Red Zone Offense",
            "├── 2-Minute Drill",
            "└── Goal Line Defense",
            "🏆 Tournament Prep",
            "🎮 My Tendencies",
        ]

        for category in sample_categories:
            self.category_list.addItem(category)

        self.category_list.itemClicked.connect(self.on_category_selected)
        layout.addWidget(self.category_list)

        layout.addStretch()

    def create_new_gameplan(self):
        """Create a new gameplan."""
        print("🎯 Opening gameplan creation dialog...")
        dialog = GameplanCreationDialog(self)
        result = dialog.exec()

        print(f"🎯 Dialog result: {result} (Accepted = 1)")

        if result == QDialog.DialogCode.Accepted and dialog.gameplan_data:
            print(f"🎯 Gameplan data received: {dialog.gameplan_data}")
            self.add_gameplan_to_list(dialog.gameplan_data)

            # Print success message
            if dialog.gameplan_data["target"]:
                print(
                    f"🎯 Created {dialog.gameplan_data['category_type'].lower()} gameplan: {dialog.gameplan_data['name']}"
                )
                print(f"   Target: {dialog.gameplan_data['target']}")
            else:
                print(
                    f"🎯 Created {dialog.gameplan_data['category_type'].lower()} gameplan: {dialog.gameplan_data['name']}"
                )
                print(f"   Category: {dialog.gameplan_data['category_type']}")

    def add_gameplan_to_list(self, gameplan_data):
        """Add a new gameplan to the category list."""
        print(f"📁 Adding gameplan to list - data: {gameplan_data}")
        print(f"📁 Category list current count: {self.category_list.count()}")

        # Create display text based on gameplan type
        category_type = gameplan_data["category_type"]
        name = gameplan_data["name"]
        target = gameplan_data["target"]

        if category_type == "Opponent-Specific":
            display_text = f"🎯 Opponent: {target}"
        elif category_type == "Formation Counter":
            display_text = f"🛡️ Counter: {target}"
        elif category_type == "Custom Category":
            display_text = f"⚡ Custom: {target}"
        else:
            # General Strategy, Tournament Prep, Situation-Based
            emoji = gameplan_data["full_type"].split(" ")[0]  # Get emoji from full_type
            display_text = f"{emoji} {category_type}: {name}"

        print(f"📁 Display text created: {display_text}")

        # Add to list
        self.category_list.addItem(display_text)
        print(f"📁 Item added! New count: {self.category_list.count()}")
        print(f"📁 Added to category list: {display_text}")

    def save_gameplan(self):
        """Save current gameplan."""
        print("💾 Saving gameplan...")
        # TODO: Implement actual save functionality

    def export_gameplan(self):
        """Export gameplan."""
        print("📁 Exporting gameplan...")
        # TODO: Implement actual export functionality

    def manage_gameplans(self):
        """Open gameplan management dialog."""
        # Get current categories
        categories = []
        for i in range(self.category_list.count()):
            categories.append(self.category_list.item(i).text())

        dialog = GameplanManagerDialog(categories, self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            self.update_categories_from_dialog(dialog.deleted_categories)

    def update_categories_from_dialog(self, deleted_categories):
        """Update the category list after management dialog."""
        for category in deleted_categories:
            # Find and remove the category from the list
            for i in range(self.category_list.count()):
                if self.category_list.item(i).text() == category:
                    self.category_list.takeItem(i)
                    break

        print("📁 Category list updated from management dialog")

    def on_category_selected(self, item):
        """Handle category selection."""
        category_name = item.text()
        print(f"📋 Selected category: {category_name}")
        # TODO: Load and display the selected gameplan


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = SpygateDesktopAppTabbed()
    window.show()

    sys.exit(app.exec())
