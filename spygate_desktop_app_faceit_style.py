#!/usr/bin/env python3
"""
SpygateAI Desktop Application - FaceIt Style Layout
===================================================

Exact FaceIt layout with SpygateAI functionality and content.
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


class SpygateDesktopFaceItStyle(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop - FaceIt Style")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark background
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #0b0c0f;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """)

        self.current_content = "analysis"  # Track current tab
        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout (3-column like FaceIt)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar
        self.create_left_sidebar(main_layout)

        # Main Content Area
        self.create_main_content(main_layout)

        # Right Sidebar
        self.create_right_sidebar(main_layout)

    def create_left_sidebar(self, parent_layout):
        # Left sidebar frame
        left_sidebar = QFrame()
        left_sidebar.setFixedWidth(250)
        left_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
                border-right: 1px solid #2a2a2a;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Header area
        header_widget = QWidget()
        header_widget.setFixedHeight(80)
        header_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: #0b0c0f;
                border-bottom: 1px solid #2a2a2a;
            }}
        """
        )
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 0, 20, 0)

        logo_label = QLabel("🏈 SPYGATE")
        logo_label.setStyleSheet(
            f"""
            QLabel {{
                color: #1ce783;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )
        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        sidebar_layout.addWidget(header_widget)

        # Navigation items (SpygateAI specific)
        nav_items = [
            ("📊", "Analysis"),
            ("🏠", "Dashboard"),
            ("🎯", "Gameplan"),
            ("📚", "Learn"),
            ("🎬", "Clips"),
            ("📈", "Stats"),
            ("⚙️", "Settings"),
        ]

        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(5)

        # Store nav buttons for selection management
        self.nav_buttons = []
        
        for icon, text in nav_items:
            nav_button = self.create_nav_button(icon, text)
            nav_layout.addWidget(nav_button)
            self.nav_buttons.append(nav_button)

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_widget)

        parent_layout.addWidget(left_sidebar)

    def create_nav_button(self, icon, text):
        button = QPushButton(f"{icon}  {text}")
        button.setFixedHeight(45)
        button.setCheckable(True)  # Make button checkable for selected state
        
        # Set first button (Analysis) as selected by default
        if text == "Analysis":
            button.setChecked(True)
            
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
            QPushButton:checked {{
                color: #ffffff;
                background-color: #1a1a1a;
            }}
            QPushButton:hover {{
                background-color: #1a1a1a;
                color: #1ce783;
            }}
            QPushButton:pressed {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
        """
        )
        
        # Connect button click to handle selection
        button.clicked.connect(lambda: self.handle_nav_selection(button, text.lower()))
        return button

    def handle_nav_selection(self, selected_button, content_type):
        """Handle navigation tab selection - only one tab selected at a time"""
        for button in self.nav_buttons:
            button.setChecked(False)
        selected_button.setChecked(True)
        
        # Update main content based on selection
        self.current_content = content_type
        self.update_main_content()

    def create_main_content(self, parent_layout):
        # Main content area
        self.main_content = QFrame()
        self.main_content.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
            }}
        """
        )

        self.content_layout = QVBoxLayout(self.main_content)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.content_layout.setSpacing(20)

        # Initial content (Analysis)
        self.update_main_content()

        parent_layout.addWidget(self.main_content, 1)  # Takes remaining space

    def update_main_content(self):
        """Update main content based on current selection"""
        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        if self.current_content == "analysis":
            self.create_analysis_content()
        elif self.current_content == "dashboard":
            self.create_dashboard_content()
        elif self.current_content == "gameplan":
            self.create_gameplan_content()
        elif self.current_content == "learn":
            self.create_learn_content()
        else:
            self.create_default_content()

    def create_analysis_content(self):
        """Create the analysis tab content with FaceIt styling"""
        # Header
        header = QLabel("Latest Analysis")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 10px 0;
        """
        )
        self.content_layout.addWidget(header)

        # YouTube-style upload area
        upload_container = QWidget()
        upload_container.setMaximumWidth(600)
        upload_container.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 12px;
                padding: 40px;
            }
        """
        )

        container_layout = QVBoxLayout(upload_container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Upload icon
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
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        """
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Drag and drop video files here, or click to browse")
        subtitle.setStyleSheet(
            """
            color: #767676;
            font-family: "Minork Sans", sans-serif;
            font-size: 14px;
            margin-bottom: 20px;
        """
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(subtitle)

        # Browse button
        browse_btn = QPushButton("Browse Files")
        browse_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1ce783;
                color: #e3e3e3;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-family: "Minork Sans", sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #17d474; }
        """
        )
        browse_btn.clicked.connect(self.browse_file)
        container_layout.addWidget(browse_btn)

        # Center the upload container
        centered_layout = QHBoxLayout()
        centered_layout.addStretch()
        centered_layout.addWidget(upload_container)
        centered_layout.addStretch()

        upload_widget = QWidget()
        upload_widget.setLayout(centered_layout)
        self.content_layout.addWidget(upload_widget)

        self.content_layout.addStretch()

    def create_dashboard_content(self):
        """Create dashboard content"""
        header = QLabel("🏠 Dashboard")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px 0;
        """
        )
        self.content_layout.addWidget(header)

        stats_text = QLabel("📈 Quick Stats Coming Soon...")
        stats_text.setStyleSheet(
            """
            color: #767676;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 16px;
        """
        )
        self.content_layout.addWidget(stats_text)
        self.content_layout.addStretch()

    def create_gameplan_content(self):
        """Create gameplan content"""
        header = QLabel("🎯 Gameplan")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px 0;
        """
        )
        self.content_layout.addWidget(header)

        gameplan_text = QLabel("🎮 Strategy management coming soon...")
        gameplan_text.setStyleSheet(
            """
            color: #767676;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 16px;
        """
        )
        self.content_layout.addWidget(gameplan_text)
        self.content_layout.addStretch()

    def create_learn_content(self):
        """Create learn content"""
        header = QLabel("📚 Learn")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px 0;
        """
        )
        self.content_layout.addWidget(header)

        learn_text = QLabel("🎓 Learning features coming soon...")
        learn_text.setStyleSheet(
            """
            color: #767676;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 16px;
        """
        )
        self.content_layout.addWidget(learn_text)
        self.content_layout.addStretch()

    def create_default_content(self):
        """Create default content for unknown tabs"""
        header = QLabel("Coming Soon...")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px 0;
        """
        )
        self.content_layout.addWidget(header)
        self.content_layout.addStretch()

    def create_right_sidebar(self, parent_layout):
        # Right sidebar
        right_sidebar = QFrame()
        right_sidebar.setFixedWidth(300)
        right_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
                border-left: 1px solid #2a2a2a;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(right_sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)

        # Watch section (like FaceIt)
        self.create_watch_section(sidebar_layout)

        # Recent clips section
        self.create_recent_clips_section(sidebar_layout)

        sidebar_layout.addStretch()
        parent_layout.addWidget(right_sidebar)

    def create_watch_section(self, parent_layout):
        # Watch header
        watch_header = QLabel("Watch")
        watch_header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        parent_layout.addWidget(watch_header)

        # Watch items
        watch_items = [
            "🎬 Pro Gameplay Analysis",
            "📺 Tutorial Videos", 
            "🏆 Tournament Highlights"
        ]

        for item in watch_items:
            watch_item = QLabel(item)
            watch_item.setStyleSheet(
                """
                color: #767676;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                padding: 8px 0;
            """
            )
            parent_layout.addWidget(watch_item)

    def create_recent_clips_section(self, parent_layout):
        # Recent clips header
        clips_header = QLabel("Recent Analysis")
        clips_header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin: 20px 0 10px 0;
        """
        )
        parent_layout.addWidget(clips_header)

        # Placeholder clips
        clip_items = [
            "📊 3rd & Long Analysis",
            "🏃 Red Zone Breakdown",
            "⏱️ 2-Minute Drill Study"
        ]

        for item in clip_items:
            clip_item = QLabel(item)
            clip_item.setStyleSheet(
                """
                color: #767676;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                padding: 8px 0;
            """
            )
            parent_layout.addWidget(clip_item)

    def browse_file(self):
        """Browse for video files"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Madden Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv);;All Files (*)",
        )
        if file_path:
            print(f"🎬 Processing video: {file_path}")
            # Here you would integrate the actual video processing logic


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = SpygateDesktopFaceItStyle()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 