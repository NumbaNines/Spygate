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
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: #0b0c0f;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

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
        """Create gameplan content with Play Planner"""
        # Header with toggle button
        header_layout = QHBoxLayout()
        
        header = QLabel("🎯 Play Planner")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px 0;
        """
        )
        header_layout.addWidget(header)
        
        # Toggle button for sidebar
        self.sidebar_toggle_btn = QPushButton("⚙️ Hide Controls")
        self.sidebar_toggle_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #34e89a;
            }
            QPushButton:pressed {
                background-color: #1fc47d;
            }
        """
        )
        self.sidebar_toggle_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addStretch()
        header_layout.addWidget(self.sidebar_toggle_btn)
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        self.content_layout.addWidget(header_widget)

        # Main Play Planner layout
        self.planner_layout = QHBoxLayout()

        # Football field area (main center)
        self.field_widget = self.create_play_planner_field()
        self.planner_layout.addWidget(self.field_widget, 3)  # Takes more space by default

        # Controls area (right side)
        self.controls_widget = self.create_formation_controls()
        self.planner_layout.addWidget(self.controls_widget, 1)  # Takes less space

        # Add to main layout
        self.planner_container = QWidget()
        self.planner_container.setLayout(self.planner_layout)
        self.content_layout.addWidget(self.planner_container)
        
        # Store sidebar visibility state
        self.sidebar_visible = True

    def toggle_sidebar(self):
        """Toggle the visibility of the formation controls sidebar"""
        if self.sidebar_visible:
            # Hide the sidebar
            self.controls_widget.hide()
            self.sidebar_toggle_btn.setText("⚙️ Show Controls")
            self.sidebar_visible = False
            # Field takes up all available space
            self.planner_layout.setStretch(0, 1)  # Field widget index 0, stretch factor 1
        else:
            # Show the sidebar
            self.controls_widget.show()
            self.sidebar_toggle_btn.setText("⚙️ Hide Controls")
            self.sidebar_visible = True
            # Reset to normal proportions: field takes 3 parts, controls take 1 part
            self.planner_layout.setStretch(0, 3)  # Field widget
            self.planner_layout.setStretch(1, 1)  # Controls widget

    def create_play_planner_field(self):
        """Create the interactive 120-yard NFL football field using QGraphicsScene"""
        # Create graphics view and scene
        self.field_view = QGraphicsView()
        self.field_scene = QGraphicsScene()
        self.field_view.setScene(self.field_scene)

        # Style the graphics view
        self.field_view.setStyleSheet(
            """
            QGraphicsView {
                background-color: #151515;
                border: 2px solid #29d28c;
                border-radius: 8px;
            }
        """
        )

        # Set scene dimensions (120 yards + end zones) - Vertical orientation like Madden
        # Made wider for better layout proportions
        field_width = 800  # ~53.3 yards width * 15 pixels per yard (increased from 600)
        field_height = 1200  # 120 yards * 10 pixels per yard
        self.field_scene.setSceneRect(0, 0, field_width, field_height)

        # Draw the football field
        self.draw_football_field()

        # Add player icons
        self.add_player_icons()

        # Apply default Gun Bunch formation
        self.apply_formation_preset("Gun Bunch")

        # Enable drag and drop
        self.field_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

        return self.field_view

    def draw_football_field(self):
        """Draw the football field with yard markers, hash marks, and goal lines - Vertical orientation like Madden"""
        field_width = 800  # Width (53.3 yards) - increased for wider layout
        field_height = 1200  # Height (120 yards)

        # Field background - Updated color scheme
        field_rect = QGraphicsRectItem(0, 0, field_width, field_height)
        field_rect.setBrush(QBrush(QColor("#151515")))  # Dark background as requested
        field_rect.setPen(QPen(QColor("#29d28c"), 2))  # Green border
        self.field_scene.addItem(field_rect)

        # End zones (darker) - Now top and bottom
        end_zone_height = 100  # 10 yards

        # Top end zone
        top_endzone = QGraphicsRectItem(0, 0, field_width, end_zone_height)
        top_endzone.setBrush(QBrush(QColor("#0f0f0f")))  # Even darker for end zones
        top_endzone.setPen(QPen(QColor("#29d28c"), 2))  # Green border
        self.field_scene.addItem(top_endzone)

        # Bottom end zone
        bottom_endzone = QGraphicsRectItem(
            0, field_height - end_zone_height, field_width, end_zone_height
        )
        bottom_endzone.setBrush(QBrush(QColor("#0f0f0f")))  # Even darker for end zones
        bottom_endzone.setPen(QPen(QColor("#29d28c"), 2))  # Green border
        self.field_scene.addItem(bottom_endzone)

        # Yard lines (every 5 yards per NFL specifications) - Horizontal lines
        for yard in range(0, 121, 5):
            y = yard * 10
            # Thicker lines for major yard markers (every 10 yards)
            line_width = 3 if yard % 10 == 0 else 1
            yard_line = QGraphicsLineItem(0, y, field_width, y)
            yard_line.setPen(QPen(QColor("#29d28c"), line_width))  # Green yard lines
            self.field_scene.addItem(yard_line)

            # Yard numbers (only on 10-yard intervals)
            if 10 <= yard <= 110 and yard % 10 == 0:
                # Calculate proper yard number display
                if yard <= 50:
                    display_num = yard
                else:
                    display_num = 110 - yard
                
                if yard != 50:  # Don't duplicate 50-yard line number
                    yard_text = QGraphicsTextItem(str(display_num))
                    yard_text.setDefaultTextColor(QColor("#29d28c"))  # Green yard numbers
                    yard_text.setFont(QFont("Minork Sans", 12, QFont.Weight.Bold))
                    yard_text.setPos(field_width // 2 - 15, y - 20)
                    self.field_scene.addItem(yard_text)

        # 50-yard line (special) - Horizontal line at midfield
        fifty_line = QGraphicsLineItem(0, 600, field_width, 600)
        fifty_line.setPen(QPen(QColor("#29d28c"), 4))  # Green 50-yard line (most prominent)
        self.field_scene.addItem(fifty_line)
        
        # Add "50" yard marker
        fifty_text = QGraphicsTextItem("50")
        fifty_text.setDefaultTextColor(QColor("#29d28c"))
        fifty_text.setFont(QFont("Minork Sans", 14, QFont.Weight.Bold))
        fifty_text.setPos(field_width // 2 - 20, 580)
        self.field_scene.addItem(fifty_text)

        # Hash marks - NFL specification: 20 yards from each sideline
        # Field width represents 53.3 yards, hash marks at 20 yards from each side
        left_hash_pos = int((20.0 / 53.3) * field_width)  # ~300px from left
        right_hash_pos = int((33.3 / 53.3) * field_width)  # ~500px from left (20 yards from right)
        hash_length = 6  # Short hash mark length
        
        # Hash marks on every 5-yard line (excluding goal lines and 50-yard line)
        for yard in range(5, 116, 5):
            y = yard * 10
            # Left hash mark
            hash1 = QGraphicsLineItem(left_hash_pos - hash_length//2, y, left_hash_pos + hash_length//2, y)
            hash1.setPen(QPen(QColor("#29d28c"), 2))  # Green hash marks
            self.field_scene.addItem(hash1)

            # Right hash mark  
            hash2 = QGraphicsLineItem(right_hash_pos - hash_length//2, y, right_hash_pos + hash_length//2, y)
            hash2.setPen(QPen(QColor("#29d28c"), 2))  # Green hash marks
            self.field_scene.addItem(hash2)

        # NFL Special Field Markings
        
        # 2-yard line conversion markers (3-foot lines parallel to goal line at center)
        # Top end zone 2-yard line
        conversion_line_top = QGraphicsLineItem(field_width // 2 - 18, 120, field_width // 2 + 18, 120)
        conversion_line_top.setPen(QPen(QColor("#29d28c"), 2))
        self.field_scene.addItem(conversion_line_top)
        
        # Bottom end zone 2-yard line
        conversion_line_bottom = QGraphicsLineItem(field_width // 2 - 18, 1080, field_width // 2 + 18, 1080)
        conversion_line_bottom.setPen(QPen(QColor("#29d28c"), 2))
        self.field_scene.addItem(conversion_line_bottom)
        
        # 35-yard line kickoff spots (small X marks at center)
        # Top 35-yard line (y = 350)
        x_mark_top = QGraphicsTextItem("X")
        x_mark_top.setDefaultTextColor(QColor("#29d28c"))
        x_mark_top.setFont(QFont("Minork Sans", 10, QFont.Weight.Bold))
        x_mark_top.setPos(field_width // 2 - 5, 340)
        self.field_scene.addItem(x_mark_top)
        
        # Bottom 35-yard line (y = 850) 
        x_mark_bottom = QGraphicsTextItem("X")
        x_mark_bottom.setDefaultTextColor(QColor("#29d28c"))
        x_mark_bottom.setFont(QFont("Minork Sans", 10, QFont.Weight.Bold))
        x_mark_bottom.setPos(field_width // 2 - 5, 840)
        self.field_scene.addItem(x_mark_bottom)
        
        # 50-yard line logo area (placeholder for team/league logos)
        logo_area = QGraphicsEllipseItem(field_width // 2 - 40, 580, 80, 40)
        logo_area.setBrush(QBrush(QColor("#29d28c", 30)))  # Semi-transparent green
        logo_area.setPen(QPen(QColor("#29d28c"), 1, Qt.PenStyle.DashLine))
        self.field_scene.addItem(logo_area)
        
        # Red zone indicators (20-yard lines)
        # Top red zone (y = 200)
        red_zone_top = QGraphicsRectItem(0, 195, field_width, 10)
        red_zone_top.setBrush(QBrush(QColor(220, 20, 60, 40)))  # Semi-transparent red
        red_zone_top.setPen(QPen(QColor(220, 20, 60), 1))
        self.field_scene.addItem(red_zone_top)
        
        # Bottom red zone (y = 1000) 
        red_zone_bottom = QGraphicsRectItem(0, 995, field_width, 10)
        red_zone_bottom.setBrush(QBrush(QColor(220, 20, 60, 40)))  # Semi-transparent red
        red_zone_bottom.setPen(QPen(QColor(220, 20, 60), 1))
        self.field_scene.addItem(red_zone_bottom)

    def add_player_icons(self):
        """Add draggable player icons to the field"""
        self.offensive_players = {}
        self.defensive_players = {}

        # Calculate scaling factor for the wider field (800px vs 600px)
        scale_factor = 800 / 600  # 1.333
        
        # Offensive formation (11 players) - Scaled for wider field
        offensive_positions = [
            ("QB", int(300 * scale_factor), 750),  # Quarterback deeper in own territory
            ("RB", int(300 * scale_factor), 710),  # Running back behind QB
            ("WR1", int(100 * scale_factor), 750),  # Split end
            ("WR2", int(500 * scale_factor), 750),  # Flanker
            ("WR3", int(150 * scale_factor), 650),  # Slot receiver
            ("TE", int(350 * scale_factor), 790),  # Tight end
            ("LT", int(250 * scale_factor), 780),  # Left tackle
            ("LG", int(275 * scale_factor), 780),  # Left guard
            ("C", int(300 * scale_factor), 780),  # Center
            ("RG", int(325 * scale_factor), 780),  # Right guard
            ("RT", int(350 * scale_factor), 780),  # Right tackle
        ]

        for pos, x, y in offensive_positions:
            player = self.create_player_icon(
                pos, x, y, QColor("#bfbfc1"), True
            )  # Light gray for offense indicators
            self.offensive_players[pos] = player

        # Defensive formation (11 players) - Scaled for wider field
        defensive_positions = [
            ("DE", int(200 * scale_factor), 420),  # Defensive end
            ("DT", int(280 * scale_factor), 420),  # Defensive tackle
            ("NT", int(320 * scale_factor), 420),  # Nose tackle
            ("OLB", int(150 * scale_factor), 450),  # Outside linebacker
            ("MLB", int(300 * scale_factor), 450),  # Middle linebacker
            ("CB1", int(100 * scale_factor), 500),  # Cornerback 1
            ("CB2", int(500 * scale_factor), 500),  # Cornerback 2
            ("FS", int(200 * scale_factor), 550),  # Free safety
            ("SS", int(400 * scale_factor), 550),  # Strong safety
            ("DE2", int(380 * scale_factor), 420),  # Defensive end 2
            ("LB", int(450 * scale_factor), 450),  # Linebacker
        ]

        for pos, x, y in defensive_positions:
            player = self.create_player_icon(pos, x, y, QColor(220, 20, 60), False)  # Crimson
            self.defensive_players[pos] = player

    def create_player_icon(self, position, x, y, color, is_offensive):
        """Create a draggable player icon"""
        # Create player circle
        player = QGraphicsEllipseItem(0, 0, 30, 30)
        player.setBrush(QBrush(color))
        player.setPen(QPen(QColor(255, 255, 255), 2))
        player.setPos(x - 15, y - 15)  # Center the circle

        # Make draggable
        player.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        player.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        # Add position label (centered inside circle)
        label = QGraphicsTextItem(position)
        label.setDefaultTextColor(QColor("#29d28c"))  # Green text for position labels
        label.setFont(QFont("Minork Sans", 8, QFont.Weight.Bold))

        # Center the text within the 30x30 circle
        text_rect = label.boundingRect()
        label.setPos(
            (30 - text_rect.width()) / 2,  # Center horizontally within circle
            (30 - text_rect.height()) / 2,  # Center vertically within circle
        )
        label.setParentItem(player)

        # Store metadata
        player.setData(0, position)  # Position name
        player.setData(1, is_offensive)  # True if offensive player
        player.setData(2, color)  # Original color

        self.field_scene.addItem(player)
        return player

    def create_formation_controls(self):
        """Create formation controls for the right sidebar"""
        controls_widget = QWidget()
        controls_widget.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
                padding: 20px;
            }
        """
        )

        layout = QVBoxLayout(controls_widget)
        layout.setSpacing(15)

        # Controls header
        controls_header = QLabel("Formation Controls")
        controls_header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            margin-bottom: 10px;
        """
        )
        layout.addWidget(controls_header)

        # Formation presets section
        presets_label = QLabel("Formation Presets:")
        presets_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        """
        )
        layout.addWidget(presets_label)

        # Formation preset buttons
        formations = [
            "I-Formation",
            "Shotgun",
            "Gun Bunch",
            "Gun Trips Te",
            "Gun Normal Y Off Close",
            "Pistol",
            "Spread",
            "Singleback",
            "Wildcat",
        ]

        for formation in formations:
            btn = QPushButton(formation)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2a2a2a;
                    color: #e3e3e3;
                    padding: 8px 12px;
                    border: none;
                    border-radius: 4px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 12px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #1ce783;
                    color: #0b0c0f;
                }
            """
            )
            btn.clicked.connect(lambda checked, f=formation: self.apply_formation_preset(f))
            layout.addWidget(btn)

        # Personnel packages section
        personnel_label = QLabel("Personnel Packages:")
        personnel_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            margin-top: 15px;
        """
        )
        layout.addWidget(personnel_label)

        # Personnel buttons
        personnel = [
            "11 Personnel",
            "12 Personnel",
            "21 Personnel",
            "Nickel Defense",
            "Dime Defense",
            "Quarter Defense",
        ]

        for package in personnel:
            btn = QPushButton(package)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2a2a2a;
                    color: #e3e3e3;
                    padding: 8px 12px;
                    border: none;
                    border-radius: 4px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 12px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #1ce783;
                    color: #0b0c0f;
                }
            """
            )
            btn.clicked.connect(lambda checked, p=package: self.apply_personnel_package(p))
            layout.addWidget(btn)

        # Action buttons section
        actions_label = QLabel("Actions:")
        actions_label.setStyleSheet(
            """
            color: #ffffff;
            font-family: 'Minork Sans', Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            margin-top: 15px;
        """
        )
        layout.addWidget(actions_label)

        # Action buttons with accent color
        action_buttons = [
            ("Save Play", self.save_play),
            ("Load Play", self.load_play),
            ("Clear Field", self.clear_field),
            ("Reset Positions", self.reset_positions),
        ]

        for btn_text, btn_func in action_buttons:
            btn = QPushButton(btn_text)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #1ce783;
                    color: #e3e3e3;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 6px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #17d474;
                }
            """
            )
            btn.clicked.connect(btn_func)
            layout.addWidget(btn)

        layout.addStretch()
        return controls_widget

    def apply_formation_preset(self, formation):
        """Apply a formation preset to offensive players"""
        print(f"🏈 Applying formation: {formation}")

        # Define formation templates with player positions (x, y coordinates) - FLIPPED TO ATTACK UPFIELD
        formations = {
            "I-Formation": {
                "QB": (300, 750),
                "RB": (300, 710),
                "FB": (300, 730),  # Fullback in I-Formation
                "WR1": (100, 780),
                "WR2": (500, 780),
                "WR3": (150, 780),
                "TE": (350, 780),
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
            "Shotgun": {
                "QB": (300, 720),  # QB deeper in shotgun
                "RB": (250, 720),  # RB beside QB
                "WR1": (100, 780),  # Split end
                "WR2": (500, 780),  # Flanker
                "WR3": (150, 720),  # Slot receiver
                "TE": (375, 780),  # Tight end
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
            "Gun Bunch": {
                "QB": (297, 347),  # QB in shotgun formation
                "RB": (336, 348),  # Running back positioned behind/beside QB
                "WR1": (111, 299),  # Split end on left side
                "WR2": (414, 300),  # Flanker on right side
                "WR3": (446, 309),  # Bunch formation with TE
                "TE": (381, 309),  # Tight end in bunch with WR3
                "LT": (250, 300),  # Offensive line on line of scrimmage
                "LG": (275, 300),
                "C": (300, 300),
                "RG": (325, 300),
                "RT": (350, 300),
            },
            "Gun Trips Te": {
                "QB": (309, 348),  # QB in shotgun formation
                "RB": (352, 348),  # Running back positioned beside QB
                "WR1": (36, 300),  # Wide receiver far left
                "WR2": (186, 308),  # Trips formation receiver
                "WR3": (108, 309),  # Trips formation receiver
                "TE": (412, 301),  # Tight end on right side
                "LT": (246, 300),  # Offensive line on line of scrimmage
                "LG": (280, 299),
                "C": (312, 300),
                "RG": (344, 300),
                "RT": (377, 300),
            },
            "Gun Normal Y Off Close": {
                "QB": (309, 348),  # QB in shotgun formation
                "RB": (352, 348),  # Running back positioned beside QB
                "WR1": (91, 299),  # Wide receiver left side
                "WR2": (472, 299),  # Wide receiver right side
                "WR3": (181, 307),  # Slot receiver
                "TE": (415, 309),  # Tight end Y Off formation
                "LT": (246, 309),  # Offensive line with varied positioning
                "LG": (279, 305),
                "C": (312, 300),
                "RG": (344, 303),
                "RT": (379, 309),
            },
            "Pistol": {
                "QB": (300, 740),  # QB closer to line than shotgun
                "RB": (300, 710),  # RB directly behind QB
                "WR1": (100, 780),
                "WR2": (500, 780),
                "WR3": (150, 740),
                "TE": (375, 780),
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
            "Spread": {
                "QB": (300, 720),  # QB in shotgun
                "RB": (300, 700),  # RB behind QB
                "WR1": (80, 780),  # Wide spread
                "WR2": (520, 780),  # Wide spread
                "WR3": (150, 740),  # Slot left
                "TE": (450, 740),  # Slot right (TE as receiver)
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
            "Singleback": {
                "QB": (300, 750),
                "RB": (300, 720),  # RB directly behind QB
                "WR1": (100, 780),
                "WR2": (500, 780),
                "WR3": (150, 780),
                "TE": (375, 780),
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
            "Wildcat": {
                "QB": (250, 720),  # QB as receiver in Wildcat
                "RB": (300, 750),  # RB takes snap
                "WR1": (100, 780),
                "WR2": (500, 780),
                "WR3": (350, 720),  # Additional back
                "TE": (375, 780),
                "LT": (250, 780),
                "LG": (275, 780),
                "C": (300, 780),
                "RG": (325, 780),
                "RT": (350, 780),
            },
        }

        # Apply the formation if it exists
        if formation in formations:
            formation_positions = formations[formation]
            
            # Calculate scaling factor for the wider field (800px vs 600px)
            scale_factor = 800 / 600  # 1.333

            # Move each offensive player to their formation position
            for position, player in self.offensive_players.items():
                if position in formation_positions:
                    original_x, new_y = formation_positions[position]
                    # Scale the x-coordinate for the wider field, keep y-coordinate the same
                    new_x = int(original_x * scale_factor)
                    # Adjust for player icon center (15px offset since icon is 30x30)
                    player.setPos(new_x - 15, new_y - 15)
                    print(f"  Moved {position} to ({new_x}, {new_y})")

            print(f"✅ Applied {formation} formation!")
        else:
            print(f"❌ Formation '{formation}' not found!")

    def apply_personnel_package(self, package):
        """Apply a personnel package to defensive players"""
        print(f"🛡️ Applying personnel: {package}")
        # Personnel-specific positioning logic would go here

    def save_play(self):
        """Save the current play setup"""
        print("💾 Saving play...")
        # Save current player positions

    def load_play(self):
        """Load a saved play setup"""
        print("📂 Loading play...")
        # Load saved player positions

    def clear_field(self):
        """Clear all routes and annotations from the field"""
        print("🧹 Clearing field...")
        # Clear any drawn routes or annotations

    def reset_positions(self):
        """Reset all players to default positions"""
        print("🔄 Resetting positions...")
        # Reset players to original formation positions

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
        watch_items = ["🎬 Pro Gameplay Analysis", "📺 Tutorial Videos", "🏆 Tournament Highlights"]

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
        clip_items = ["📊 3rd & Long Analysis", "🏃 Red Zone Breakdown", "⏱️ 2-Minute Drill Study"]

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
