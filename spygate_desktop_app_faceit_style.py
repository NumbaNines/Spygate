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

from profile_picture_manager import ProfilePictureManager, is_emoji_profile

# Import user database
from user_database import User, UserDatabase

try:
    import cv2
    import numpy as np
    from PIL import Image
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

        # Make window frameless for custom controls
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Enable window dragging
        self.drag_pos = QPoint()

        # Setup keyboard shortcuts
        self.setup_shortcuts()

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

        self.current_content = "dashboard"  # Track current tab

        # Initialize user database and current user
        self.user_db = UserDatabase()
        self.profile_manager = ProfilePictureManager()
        self.current_user = self.user_db.get_user_by_username("NumbaNines")
        if not self.current_user:
            print("❌ User not found! Creating user...")
            from user_database import setup_demo_user

            self.current_user = setup_demo_user()

        print(
            f"👤 Logged in as: {self.current_user.display_name} ({self.current_user.subscription_type})"
        )

        # Update last login
        self.user_db.update_last_login(self.current_user.user_id)

        # Initialize formation data
        self.players = {}
        self.formation_presets = self.load_formation_presets()

        self.init_ui()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # F11 for fullscreen toggle
        fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        else:
            self.showFullScreen()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout to accommodate header
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top Header Bar
        self.create_header_bar(main_layout)

        # Main content layout (3-column like FaceIt)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Left Sidebar
        self.create_left_sidebar(content_layout)

        # Main Content Area
        self.create_main_content(content_layout)

        # Right Sidebar
        self.create_right_sidebar(content_layout)

        # Add content layout to main layout
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

    def create_header_bar(self, parent_layout):
        """Create top header bar with user controls in top-right"""
        header_bar = QFrame()
        header_bar.setFixedHeight(50)
        header_bar.setStyleSheet(
            """
            QFrame {
                background-color: #0b0c0f;
            }
        """
        )

        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 5, 20, 5)
        header_layout.setSpacing(0)

        # Left side - could add breadcrumbs or app title here if needed
        header_layout.addStretch()

        # Right side - Window controls and User controls
        self.create_window_controls(header_layout)
        self.create_user_controls(header_layout)

        parent_layout.addWidget(header_bar)

    def create_window_controls(self, parent_layout):
        """Create custom window control buttons (minimize, maximize, close)"""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(1)

        # Minimize button
        min_btn = QPushButton("−")
        min_btn.setFixedSize(30, 30)
        min_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #565656;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """
        )
        min_btn.setToolTip("Minimize")
        min_btn.clicked.connect(self.showMinimized)
        controls_layout.addWidget(min_btn)

        # Maximize/Restore button
        self.max_btn = QPushButton("□")
        self.max_btn.setFixedSize(30, 30)
        self.max_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #565656;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
        """
        )
        self.max_btn.setToolTip("Maximize")
        self.max_btn.clicked.connect(self.toggle_maximize)
        controls_layout.addWidget(self.max_btn)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #767676;
                border: none;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e74c3c;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """
        )
        close_btn.setToolTip("Close")
        close_btn.clicked.connect(self.close)
        controls_layout.addWidget(close_btn)

        # Add some spacing before profile picture
        controls_layout.addSpacing(15)

        parent_layout.addLayout(controls_layout)

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.isFullScreen():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        elif self.isMaximized():
            self.showNormal()
            self.max_btn.setText("□")
            self.max_btn.setToolTip("Maximize")
        else:
            self.showMaximized()
            self.max_btn.setText("⧉")
            self.max_btn.setToolTip("Restore")

    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if event.buttons() == Qt.MouseButton.LeftButton and not self.drag_pos.isNull():

            # Only allow dragging from the top area (header bar)
            if event.position().y() < 50:  # Header bar height
                if self.isFullScreen():
                    # In fullscreen, restore to normal first then move
                    self.showNormal()
                    self.max_btn.setText("□")
                    self.max_btn.setToolTip("Maximize")
                    # Position window under cursor
                    self.move(event.globalPosition().toPoint() - QPoint(self.width() // 2, 25))
                elif not self.isMaximized():
                    # Normal window dragging
                    self.move(self.pos() + event.globalPosition().toPoint() - self.drag_pos)

                self.drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for window dragging"""
        self.drag_pos = QPoint()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to toggle fullscreen/restore"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Only allow double-click from the top area (header bar)
            if event.position().y() < 50:  # Header bar height
                if self.isFullScreen():
                    self.showNormal()
                    self.max_btn.setText("□")
                    self.max_btn.setToolTip("Maximize")
                elif self.isMaximized():
                    self.showFullScreen()
                else:
                    self.showMaximized()
                    self.max_btn.setText("⧉")
                    self.max_btn.setToolTip("Restore")
        super().mouseDoubleClickEvent(event)

    def create_user_controls(self, parent_layout):
        """Create circular profile picture button"""
        # Get user's profile picture from database
        profile_pic = self.get_display_profile_picture()

        # Create a container with padding to position profile pic lower
        profile_container = QWidget()
        profile_container.setFixedSize(50, 50)
        profile_layout = QVBoxLayout(profile_container)
        profile_layout.setContentsMargins(0, 5, 0, 5)  # Balanced padding
        profile_layout.setSpacing(0)

        # Circular Profile Picture button
        if is_emoji_profile(self.current_user.profile_picture if self.current_user else "🏈"):
            # Emoji profile picture
            profile_btn = QPushButton(profile_pic)
            profile_btn.setFixedSize(40, 40)
            profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    color: #e3e3e3;
                    border: 2px solid #565656;
                    border-radius: 20px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 20px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )
        else:
            # Custom image profile picture
            profile_btn = QPushButton()
            profile_btn.setFixedSize(40, 40)

            # Load and set custom image
            pixmap = self.load_profile_pixmap(profile_pic)
            if pixmap:
                profile_btn.setIcon(QIcon(pixmap))
                profile_btn.setIconSize(QSize(36, 36))

            profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    border: 2px solid #565656;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )

        profile_btn.setToolTip("Profile & Settings")
        profile_btn.clicked.connect(self.show_settings_dialog)

        # Add profile button to container
        profile_layout.addWidget(profile_btn)
        profile_layout.addStretch()

        # Add container to parent layout
        parent_layout.addWidget(profile_container)

        # Store reference to profile button for later updates
        self.profile_btn = profile_btn

    def get_display_profile_picture(self):
        """Get the profile picture for display (emoji or file path)"""
        if not self.current_user:
            return "🏈"  # Default football emoji

        # If no profile picture set, default to football
        if not self.current_user.profile_picture:
            return "🏈"

        return self.current_user.profile_picture

    def load_profile_pixmap(self, image_path: str) -> Optional[QPixmap]:
        """Load a custom profile picture as a circular QPixmap"""
        try:
            if not Path(image_path).exists():
                return None

            # Load image with PIL for processing
            with Image.open(image_path) as img:
                # Convert to QPixmap
                img_array = np.array(img)
                height, width, channel = img_array.shape
                bytes_per_line = 3 * width

                if channel == 4:  # RGBA
                    q_image = QImage(
                        img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888
                    )
                else:  # RGB
                    q_image = QImage(
                        img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
                    )

                pixmap = QPixmap.fromImage(q_image)

                # Create circular mask
                circular_pixmap = QPixmap(40, 40)
                circular_pixmap.fill(Qt.GlobalColor.transparent)

                painter = QPainter(circular_pixmap)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setBrush(
                    QBrush(
                        pixmap.scaled(
                            40,
                            40,
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(0, 0, 40, 40)
                painter.end()

                return circular_pixmap

        except Exception as e:
            print(f"❌ Error loading profile picture: {e}")
            return None

    def get_default_profile_picture(self):
        """Get a default profile picture based on user preferences or random selection"""
        # Default profile pictures - various styles
        default_profiles = [
            "🏈",  # Football (main theme)
            "👤",  # Generic user
            "🎯",  # Target (strategy theme)
            "⚡",  # Lightning (speed/power)
            "🔥",  # Fire (intensity)
            "💪",  # Strength
            "🧠",  # Brain (intelligence/strategy)
            "🏆",  # Trophy (winning)
            "⭐",  # Star (excellence)
            "🎮",  # Gaming controller
        ]

        # For now, return the football as default - later this could be user-configurable
        return default_profiles[0]  # "🏈"

    def upload_custom_profile_picture(self):
        """Upload and set a custom profile picture"""
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Profile Picture",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;All Files (*)",
        )

        if not file_path:
            return

        print(f"📤 Uploading profile picture: {file_path}")

        # Validate the image
        if not self.profile_manager.validate_image(file_path):
            QMessageBox.warning(
                self,
                "Invalid Image",
                "Please select a valid image file (PNG, JPG, GIF, BMP, or WEBP).",
                QMessageBox.StandardButton.Ok,
            )
            return

        # Process and save the image
        if self.current_user:
            # Clean up old custom profile pictures
            if (
                hasattr(self.current_user, "profile_picture_type")
                and self.current_user.profile_picture_type == "custom"
            ):
                self.profile_manager.cleanup_old_pictures(
                    self.current_user.user_id, self.current_user.profile_picture
                )

            # Process the new image
            processed_path = self.profile_manager.process_image(
                file_path, self.current_user.user_id
            )

            if processed_path:
                # Update database
                success = self.user_db.update_user_profile_picture(
                    self.current_user.user_id, processed_path, "custom"
                )

                if success:
                    # Update current user object
                    self.current_user.profile_picture = processed_path
                    self.current_user.profile_picture_type = "custom"

                    # Update UI
                    self.refresh_profile_button()
                    print(f"✅ Custom profile picture uploaded successfully!")

                    QMessageBox.information(
                        self,
                        "Success!",
                        "Your profile picture has been updated successfully!",
                        QMessageBox.StandardButton.Ok,
                    )
                else:
                    print("❌ Failed to update profile picture in database")
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to save your profile picture. Please try again.",
                        QMessageBox.StandardButton.Ok,
                    )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Failed to process your image. Please try a different image.",
                    QMessageBox.StandardButton.Ok,
                )

    def update_profile_picture(self, new_pic, pic_type="emoji"):
        """Update the profile picture button and save to database"""
        # Save to database if user is logged in
        if self.current_user:
            # Clean up old custom profile pictures if switching to emoji
            if (
                pic_type == "emoji"
                and hasattr(self.current_user, "profile_picture_type")
                and self.current_user.profile_picture_type == "custom"
            ):
                self.profile_manager.cleanup_old_pictures(self.current_user.user_id)

            success = self.user_db.update_user_profile_picture(
                self.current_user.user_id, new_pic, pic_type
            )

            if success:
                # Update current user object
                self.current_user.profile_picture = new_pic
                self.current_user.profile_picture_type = pic_type

                # Update UI
                self.refresh_profile_button()
                print(f"✅ Profile picture updated to {new_pic}")
            else:
                print("❌ Failed to update profile picture in database")

    def refresh_profile_button(self):
        """Refresh the profile button with current user's picture"""
        if not hasattr(self, "profile_btn"):
            return

        profile_pic = self.get_display_profile_picture()

        if is_emoji_profile(self.current_user.profile_picture if self.current_user else "🏈"):
            # Update to emoji
            self.profile_btn.setText(profile_pic)
            self.profile_btn.setIcon(QIcon())  # Clear any icon
            self.profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    color: #e3e3e3;
                    border: 2px solid #565656;
                    border-radius: 20px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                    font-size: 20px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )
        else:
            # Update to custom image
            self.profile_btn.setText("")  # Clear text
            pixmap = self.load_profile_pixmap(profile_pic)
            if pixmap:
                self.profile_btn.setIcon(QIcon(pixmap))
                self.profile_btn.setIconSize(QSize(36, 36))

            self.profile_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #565656;
                    border: 2px solid #565656;
                    border-radius: 20px;
                }
                QPushButton:hover {
                    border-color: #29d28c;
                    background-color: #666666;
                }
                QPushButton:pressed {
                    border-color: #1fc47d;
                    background-color: #4a4a4a;
                }
            """
            )

    def show_settings_dialog(self):
        """Show profile dropdown menu"""
        print("👤 Opening profile dropdown...")

        # Create dropdown menu
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 8px 0px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-size: 14px;
                min-width: 280px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 12px 20px;
                margin: 2px 8px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2a2a2a;
                color: #29d28c;
            }
            QMenu::item:pressed {
                background-color: #29d28c;
                color: #151515;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2a2a2a;
                margin: 8px 16px;
            }
        """
        )

        # Profile section with real user data
        display_name = self.current_user.display_name if self.current_user else "Guest"
        username = f"@{self.current_user.username}" if self.current_user else "@guest"

        profile_section = QLabel(display_name)
        profile_section.setStyleSheet(
            """
            QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 12px 20px 4px 20px;
            }
        """
        )

        # User info with subscription status
        subscription_status = ""
        if self.current_user and self.current_user.is_premium:
            subscription_status = f" • {self.current_user.subscription_type.upper()}"

        user_info = QLabel(f"{username}{subscription_status}")
        user_info.setStyleSheet(
            """
            QLabel {
                color: #767676;
                font-size: 12px;
                padding: 0px 20px 8px 20px;
            }
        """
        )

        # Create custom widget actions for labels
        profile_widget_action = QWidgetAction(menu)
        profile_widget_action.setDefaultWidget(profile_section)
        menu.addAction(profile_widget_action)

        user_widget_action = QWidgetAction(menu)
        user_widget_action.setDefaultWidget(user_info)
        menu.addAction(user_widget_action)

        menu.addSeparator()

        # Profile Picture submenu
        profile_pic_menu = menu.addMenu("🖼️ Change Profile Picture")
        profile_pic_menu.setStyleSheet(menu.styleSheet())  # Inherit parent style

        # Upload custom picture option
        upload_action = profile_pic_menu.addAction("📤 Upload Custom Picture...")
        upload_action.triggered.connect(self.upload_custom_profile_picture)

        profile_pic_menu.addSeparator()

        # Default emoji options
        default_profiles = self.profile_manager.get_default_emoji_profiles()

        for pic, name in default_profiles:
            action = profile_pic_menu.addAction(f"{pic} {name}")
            action.triggered.connect(lambda checked, p=pic: self.update_profile_picture(p, "emoji"))

        menu.addSeparator()

        # Account management with subscription-specific options
        if self.current_user and not self.current_user.is_premium:
            upgrade_action = menu.addAction("⭐ Upgrade to Premium")
            upgrade_action.triggered.connect(self.show_upgrade_info)
        else:
            manage_sub_action = menu.addAction("⭐ Manage Subscription")
            manage_sub_action.triggered.connect(self.show_subscription_info)

        profile_action = menu.addAction("👤 Manage Profile")
        profile_action.triggered.connect(self.show_profile_info)

        purchases_action = menu.addAction("💳 Purchases and memberships")
        purchases_action.triggered.connect(self.show_purchases_info)

        menu.addSeparator()

        # App settings
        settings_action = menu.addAction("⚙️ Settings")
        settings_action.triggered.connect(self.show_app_settings)

        help_action = menu.addAction("❓ Help")
        help_action.triggered.connect(self.show_help)

        feedback_action = menu.addAction("📝 Send feedback")
        feedback_action.triggered.connect(self.show_feedback)

        menu.addSeparator()

        # Sign out
        signout_action = menu.addAction("🚪 Sign out")
        signout_action.triggered.connect(self.sign_out)

        # Position menu to appear below and aligned to the right edge of the profile button
        # This keeps it within the app window bounds
        button_global_pos = self.profile_btn.mapToGlobal(self.profile_btn.rect().bottomRight())
        menu_pos = QPoint(
            button_global_pos.x() - 280, button_global_pos.y() + 5
        )  # 280px is menu width
        menu.exec(menu_pos)

    def show_upgrade_info(self):
        """Show upgrade information"""
        print("⭐ Showing upgrade options...")

        # Create upgrade dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Upgrade to Premium")
        dialog.setFixedSize(450, 500)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #34e89a;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("🏆 Upgrade to SpygateAI Premium")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Features list
        features_text = """
        ✅ Unlimited video analysis
        ✅ Advanced AI coaching insights
        ✅ Custom formation builder
        ✅ Export clips and highlights
        ✅ Priority customer support
        ✅ Beta features access
        """

        features_label = QLabel(features_text)
        features_label.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(features_label)

        # Pricing
        price_label = QLabel("Only $19.99/month")
        price_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #29d28c; text-align: center;"
        )
        price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(price_label)

        # Buttons
        button_layout = QHBoxLayout()

        upgrade_btn = QPushButton("Upgrade Now")
        upgrade_btn.clicked.connect(lambda: self.process_upgrade(dialog))
        button_layout.addWidget(upgrade_btn)

        cancel_btn = QPushButton("Maybe Later")
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        cancel_btn.clicked.connect(dialog.close)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def process_upgrade(self, dialog):
        """Process the upgrade to premium"""
        if self.current_user:
            # Create premium subscription
            subscription_id = self.user_db.create_subscription(
                self.current_user.user_id, "premium", 19.99, 12
            )

            # Refresh current user data
            self.current_user = self.user_db.get_user_by_id(self.current_user.user_id)

            print(f"✅ Upgraded to Premium! Subscription ID: {subscription_id}")

            # Show success message
            QMessageBox.information(
                self,
                "Upgrade Successful!",
                "🎉 Welcome to SpygateAI Premium!\n\nYou now have access to all premium features.",
                QMessageBox.StandardButton.Ok,
            )

            dialog.close()

    def show_subscription_info(self):
        """Show subscription management information"""
        print("⭐ Showing subscription management...")

        if not self.current_user:
            return

        # Get subscription details
        subscription_status = self.user_db.check_subscription_status(self.current_user.user_id)

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Subscription")
        dialog.setFixedSize(450, 400)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        header = QLabel("📋 Subscription Details")
        header.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Current plan
        plan_info = f"""
        Current Plan: {subscription_status.get('plan', 'Unknown').upper()}
        Status: {subscription_status.get('status', 'Unknown').upper()}
        """

        if subscription_status.get("expires_at"):
            from datetime import datetime

            expires_date = datetime.fromisoformat(
                subscription_status["expires_at"].replace("Z", "")
            )
            plan_info += f"Expires: {expires_date.strftime('%B %d, %Y')}"

        plan_label = QLabel(plan_info)
        plan_label.setStyleSheet(
            "font-size: 14px; background-color: #2a2a2a; padding: 15px; border-radius: 8px;"
        )
        layout.addWidget(plan_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_profile_info(self):
        """Show profile management"""
        print("👤 Showing profile management...")

        dialog = QDialog(self)
        dialog.setWindowTitle("Profile Settings")
        dialog.setFixedSize(400, 300)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        if self.current_user:
            profile_info = f"""
            Username: {self.current_user.username}
            Display Name: {self.current_user.display_name}
            Email: {self.current_user.email}
            Member Since: {self.current_user.created_at[:10]}
            Last Login: {self.current_user.last_login[:10]}
            """

            info_label = QLabel(profile_info)
            info_label.setStyleSheet(
                "font-size: 14px; background-color: #2a2a2a; padding: 15px; border-radius: 8px;"
            )
            layout.addWidget(info_label)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_purchases_info(self):
        """Show purchases and subscriptions"""
        print("💳 Showing purchases...")

        dialog = QDialog(self)
        dialog.setWindowTitle("Purchase History")
        dialog.setFixedSize(500, 400)
        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        header = QLabel("💳 Purchase History")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        if self.current_user:
            subscriptions = self.user_db.get_user_subscriptions(self.current_user.user_id)

            if subscriptions:
                for sub in subscriptions:
                    sub_info = f"""
                    Plan: {sub.plan_type.upper()}
                    Price: ${sub.price_paid:.2f}
                    Started: {sub.started_at[:10]}
                    Status: {sub.status.upper()}
                    """

                    sub_label = QLabel(sub_info)
                    sub_label.setStyleSheet(
                        "font-size: 12px; background-color: #2a2a2a; padding: 10px; border-radius: 6px; margin-bottom: 5px;"
                    )
                    layout.addWidget(sub_label)
            else:
                no_purchases = QLabel("No purchase history found.")
                no_purchases.setStyleSheet("font-size: 14px; color: #767676; text-align: center;")
                layout.addWidget(no_purchases)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    def show_app_settings(self):
        """Show application settings"""
        print("⚙️ Showing app settings...")

    def show_help(self):
        """Show help information"""
        print("❓ Showing help...")

    def show_feedback(self):
        """Show feedback form"""
        print("📝 Showing feedback form...")

    def sign_out(self):
        """Sign out user"""
        print("🚪 Signing out...")

    def create_left_sidebar(self, parent_layout):
        # Left sidebar frame
        left_sidebar = QFrame()
        left_sidebar.setFixedWidth(250)
        left_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Header area
        header_widget = QWidget()
        header_widget.setFixedHeight(50)
        header_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: #0b0c0f;
            }}
        """
        )
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 5, 20, 5)
        header_layout.setSpacing(0)

        # Try to load custom logo, fallback to text logo
        logo_widget = self.create_logo_widget()
        header_layout.addWidget(logo_widget)
        header_layout.addStretch()

        sidebar_layout.addWidget(header_widget)

        # Navigation items (SpygateAI specific)
        nav_items = [
            ("", "Dashboard"),
            ("", "Analysis"),
            ("", "Gameplan"),
            ("", "Learn"),
            ("", "Clips"),
            ("", "Stats"),
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

    def create_logo_widget(self):
        """Create logo widget - tries to load image logo, falls back to text"""
        logo_paths = [
            "assets/logo/spygate-logo.png",
            "assets/logo/spygate_logo.png",
            "assets/spygate-logo.png",
            "logo.png",
        ]

        # Try to load image logo
        for logo_path in logo_paths:
            if Path(logo_path).exists():
                try:
                    logo_label = QLabel()
                    pixmap = QPixmap(logo_path)

                    if not pixmap.isNull():
                        # Scale logo to fit nicely in header (max 180x50)
                        scaled_pixmap = pixmap.scaled(
                            180,
                            50,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        logo_label.setPixmap(scaled_pixmap)
                        logo_label.setToolTip("SpygateAI Desktop")
                        print(f"✅ Loaded logo from: {logo_path}")
                        return logo_label

                except Exception as e:
                    print(f"❌ Failed to load logo from {logo_path}: {e}")
                    continue

        # Fallback to text logo
        print("📝 Using text logo (no image found)")
        logo_label = QLabel("🏈 SPYGATE")
        logo_label.setStyleSheet(
            """
            QLabel {
                color: #1ce783;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }
        """
        )
        logo_label.setToolTip("SpygateAI Desktop")
        return logo_label

    def load_formation_presets(self):
        """Load formation presets from JSON file or return defaults"""
        presets_file = Path("assets/formations/formation_presets.json")

        # Default formations if file doesn't exist
        default_formations = {
            "Gun Bunch": {
                "description": "3 WR bunch formation",
                "positions": {
                    "QB": (396, 347),
                    "RB": (448, 348),
                    "WR1": (148, 299),
                    "WR2": (552, 300),
                    "WR3": (594, 309),
                    "TE": (508, 309),
                    "LT": (333, 300),
                    "LG": (366, 300),
                    "C": (400, 300),
                    "RG": (433, 300),
                    "RT": (466, 300),
                },
            },
            "I-Formation": {
                "description": "Traditional I-Formation",
                "positions": {
                    "QB": (396, 347),
                    "RB": (396, 380),
                    "FB": (396, 365),
                    "WR1": (148, 299),
                    "WR2": (644, 299),
                    "TE": (508, 309),
                    "LT": (333, 300),
                    "LG": (366, 300),
                    "C": (400, 300),
                    "RG": (433, 300),
                    "RT": (466, 300),
                },
            },
        }

        try:
            if presets_file.exists():
                with open(presets_file) as f:
                    formations = json.load(f)
                print(f"✅ Loaded formations from: {presets_file}")
                return formations
        except Exception as e:
            print(f"❌ Error loading formations: {e}")

        print("📝 Using default formations")
        return default_formations

    def browse_file(self):
        """Open file browser to select video files for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files (*)",
        )

        if file_path:
            print(f"🎬 Processing video: {file_path}")
            # Here you would integrate with your video processing logic
            QMessageBox.information(
                self,
                "Video Selected",
                f"Selected video: {Path(file_path).name}\n\nVideo processing will be implemented here.",
                QMessageBox.StandardButton.Ok,
            )

    def show_play_builder(self):
        """Show the play builder interface"""
        print("🏈 Opening Play Builder...")
        # This would open your formation editor or play builder
        QMessageBox.information(
            self,
            "Play Builder",
            "Play Builder interface will be implemented here.\n\nThis will integrate with your formation editor.",
            QMessageBox.StandardButton.Ok,
        )

    def create_nav_button(self, icon, text):
        button = QPushButton(text if not icon else f"{icon}  {text}")
        button.setFixedHeight(45)
        button.setCheckable(True)  # Make button checkable for selected state

        # Set first button (Dashboard) as selected by default
        if text == "Dashboard":
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
        self.update_right_sidebar()  # Update right sidebar based on current tab

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
        # Stop zoom timer if it exists before clearing content
        if hasattr(self, "zoom_timer") and self.zoom_timer.isActive():
            self.zoom_timer.stop()

        # Clear existing content
        for i in reversed(range(self.content_layout.count())):
            item = self.content_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        if self.current_content == "analysis":
            content_widget = self.create_analysis_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "dashboard":
            content_widget = self.create_dashboard_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "gameplan":
            content_widget = self.create_gameplan_content()
            self.content_layout.addWidget(content_widget)
        elif self.current_content == "learn":
            content_widget = self.create_learn_content()
            self.content_layout.addWidget(content_widget)
        else:
            content_widget = self.create_default_content()
            self.content_layout.addWidget(content_widget)

    def create_analysis_content(self):
        """Create the analysis tab content with FaceIt styling"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

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
        layout.addWidget(header)

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
        layout.addWidget(upload_widget)

        layout.addStretch()
        return content

    def create_dashboard_content(self):
        """Create comprehensive dashboard content"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Welcome header with user info
        welcome_layout = QHBoxLayout()

        welcome_text = (
            f"Welcome back, {self.current_user.display_name if self.current_user else 'Guest'}!"
        )
        welcome_header = QLabel(welcome_text)
        welcome_header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        welcome_layout.addWidget(welcome_header)

        # Premium badge if applicable
        if self.current_user and self.current_user.is_premium:
            premium_badge = QLabel("⭐ PREMIUM")
            premium_badge.setStyleSheet(
                """
                color: #29d28c;
                font-size: 12px;
                font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
                background-color: rgba(41, 210, 140, 0.2);
                padding: 4px 8px;
                border-radius: 4px;
                margin-left: 15px;
            """
            )
            welcome_layout.addWidget(premium_badge)

        welcome_layout.addStretch()

        welcome_widget = QWidget()
        welcome_widget.setLayout(welcome_layout)
        layout.addWidget(welcome_widget)

        # Quick Action Buttons Row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(15)

        # Upload Video Button (primary action)
        upload_btn = QPushButton("📤 Upload New Video")
        upload_btn.setFixedHeight(50)
        upload_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #29d28c;
                color: #151515;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #34e89a;
            }
            QPushButton:pressed {
                background-color: #1fc47d;
            }
        """
        )
        upload_btn.clicked.connect(self.browse_file)
        actions_row.addWidget(upload_btn)

        # Play Builder Button
        play_builder_btn = QPushButton("🏈 Play Builder")
        play_builder_btn.setFixedHeight(50)
        play_builder_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
            border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        play_builder_btn.clicked.connect(self.show_play_builder)
        actions_row.addWidget(play_builder_btn)

        # View Analysis Button
        analysis_btn = QPushButton("📊 View Analysis")
        analysis_btn.setFixedHeight(50)
        analysis_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #565656;
                color: #e3e3e3;
                border: none;
                border-radius: 8px;
                padding: 15px 25px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """
        )
        analysis_btn.clicked.connect(lambda: self.switch_to_tab("analysis"))
        actions_row.addWidget(analysis_btn)

        actions_widget = QWidget()
        actions_widget.setLayout(actions_row)
        layout.addWidget(actions_widget)

        # Stats Cards Row
        stats_row = QHBoxLayout()
        stats_row.setSpacing(15)

        # Create stat cards
        stat_cards = [
            ("📹", "Videos Analyzed", "23", "#29d28c"),
            ("⏱️", "Hours Processed", "45.2", "#1ce783"),
            ("🏈", "Formations Used", "12", "#17d474"),
            ("📈", "Win Rate", "67%", "#34e89a"),
        ]

        for icon, title, value, color in stat_cards:
            card = self.create_stat_card(icon, title, value, color)
            stats_row.addWidget(card)

        stats_widget = QWidget()
        stats_widget.setLayout(stats_row)
        layout.addWidget(stats_widget)

        # Content Row (Recent Activity + Performance Charts)
        content_row = QHBoxLayout()
        content_row.setSpacing(20)

        # Recent Activity Panel
        recent_activity = self.create_recent_activity_panel()
        content_row.addWidget(recent_activity, 1)

        # Performance Summary Panel
        performance_panel = self.create_performance_panel()
        content_row.addWidget(performance_panel, 1)

        content_widget = QWidget()
        content_widget.setLayout(content_row)
        layout.addWidget(content_widget)

        # Premium Features Showcase (if premium user)
        if self.current_user and self.current_user.is_premium:
            premium_panel = self.create_premium_features_panel()
            layout.addWidget(premium_panel)

        layout.addStretch()
        return content

    def create_stat_card(self, icon, title, value, color):
        """Create a stat card widget"""
        card = QWidget()
        card.setFixedHeight(100)
        card.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        # Icon and value row
        top_row = QHBoxLayout()

        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 24px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        top_row.addWidget(icon_label)

        top_row.addStretch()

        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 20px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        top_row.addWidget(value_label)

        top_widget = QWidget()
        top_widget.setLayout(top_row)
        layout.addWidget(top_widget)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            """
            color: #767676;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(title_label)

        return card

    def create_recent_activity_panel(self):
        """Create recent activity panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("🕒 Recent Activity")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Activity items
        activities = [
            ("📹", "Analyzed 'Saints vs Panthers' - 4th Quarter", "2 hours ago"),
            ("🏈", "Created Gun Bunch formation", "Yesterday"),
            ("📊", "Generated Red Zone report", "Yesterday"),
            ("🎬", "Exported highlight reel", "2 days ago"),
            ("📈", "Updated win rate stats", "3 days ago"),
        ]

        for icon, description, time in activities:
            activity_item = self.create_activity_item(icon, description, time)
            layout.addWidget(activity_item)

        layout.addStretch()
        return panel

    def create_activity_item(self, icon, description, time):
        """Create an activity item"""
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(10)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            """
            color: #29d28c;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(icon_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(desc_label)

        layout.addStretch()

        # Time
        time_label = QLabel(time)
        time_label.setStyleSheet(
            """
            color: #767676;
            font-size: 10px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(time_label)

        return item

    def create_performance_panel(self):
        """Create performance summary panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        header = QLabel("📈 Performance Summary")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        # Performance metrics
        metrics = [
            ("🎯", "Red Zone Efficiency", "72%", "#29d28c"),
            ("🏃", "3rd Down Conversion", "45%", "#1ce783"),
            ("⚡", "Big Play Rate", "18%", "#17d474"),
            ("🛡️", "Defensive Stops", "62%", "#34e89a"),
            ("⏱️", "Time of Possession", "58%", "#29d28c"),
        ]

        for icon, metric, value, color in metrics:
            metric_item = self.create_metric_item(icon, metric, value, color)
            layout.addWidget(metric_item)

        layout.addStretch()
        return panel

    def create_metric_item(self, icon, metric, value, color):
        """Create a performance metric item"""
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(15)

        # Icon
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 16px;
                    font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(icon_label)

        # Metric name
        metric_label = QLabel(metric)
        metric_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 12px;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(metric_label)

        layout.addStretch()

        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 14px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(value_label)

        return item

    def create_premium_features_panel(self):
        """Create premium features showcase panel"""
        panel = QWidget()
        panel.setStyleSheet(
            """
            QWidget {
                background-color: rgba(41, 210, 140, 0.1);
                border-radius: 8px;
                border: 1px solid #29d28c;
                }
            """
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()

        header = QLabel("⭐ Premium Features Active")
        header.setStyleSheet(
            """
            color: #29d28c;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Subscription status
        if self.current_user:
            status_label = QLabel(f"{self.current_user.subscription_type.upper()} Plan")
            status_label.setStyleSheet(
                """
                color: #29d28c;
                    font-size: 12px;
                    font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                background-color: rgba(41, 210, 140, 0.2);
                padding: 4px 8px;
                border-radius: 4px;
            """
            )
            header_layout.addWidget(status_label)

        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)

        # Feature highlights
        features_layout = QHBoxLayout()
        features_layout.setSpacing(15)

        features = [
            ("🚀", "Unlimited Analysis"),
            ("🎯", "Advanced AI Coaching"),
            ("📊", "Custom Reports"),
            ("🏆", "Beta Features Access"),
        ]

        for icon, feature in features:
            feature_item = QWidget()
            feature_layout = QHBoxLayout(feature_item)
            feature_layout.setContentsMargins(0, 0, 0, 0)
            feature_layout.setSpacing(8)

            icon_label = QLabel(icon)
            icon_label.setStyleSheet(
                """
                color: #29d28c;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            feature_layout.addWidget(icon_label)

            text_label = QLabel(feature)
            text_label.setStyleSheet(
                """
                color: #ffffff;
                font-size: 11px;
                font-family: 'Minork Sans', Arial, sans-serif;
            """
            )
            feature_layout.addWidget(text_label)

            features_layout.addWidget(feature_item)

        features_widget = QWidget()
        features_widget.setLayout(features_layout)
        layout.addWidget(features_widget)

        return panel

    def switch_to_gameplan_and_play_builder(self):
        """Switch to gameplan tab and launch play builder"""
        # Switch to gameplan tab first
        self.switch_to_tab("gameplan")

        # Small delay to ensure tab switch is complete, then show play builder
        QTimer.singleShot(100, self.show_play_builder)

    def switch_to_tab(self, tab_name):
        """Switch to a specific tab"""
        # Find and activate the corresponding navigation button
        for button in self.nav_buttons:
            button.setChecked(False)
            if button.text().lower().find(tab_name) != -1:
                button.setChecked(True)

        # Update content
        self.current_content = tab_name
        self.update_main_content()
        self.update_right_sidebar()

    def create_gameplan_content(self):
        """Create gameplan tab content placeholder"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        header = QLabel("Game Planning & Formation Builder")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        placeholder = QLabel("🏈 Formation Builder and Game Planning tools will be integrated here")
        placeholder.setStyleSheet(
            """
            color: #767676;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
            text-align: center;
            padding: 40px;
        """
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()
        return content

    def create_learn_content(self):
        """Create learn tab content placeholder"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        header = QLabel("Learning Center")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
        """
        )
        layout.addWidget(header)

        placeholder = QLabel("📚 Tutorials, guides, and learning resources will be available here")
        placeholder.setStyleSheet(
            """
            color: #767676;
            font-size: 16px;
            font-family: 'Minork Sans', Arial, sans-serif;
            text-align: center;
            padding: 40px;
        """
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()
        return content

    def create_default_content(self):
        """Create default content for unimplemented tabs"""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        placeholder = QLabel("🚧 This feature is coming soon!")
        placeholder.setStyleSheet(
            """
            color: #767676;
            font-size: 18px;
            font-family: 'Minork Sans', Arial, sans-serif;
            text-align: center;
            padding: 40px;
        """
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

        layout.addStretch()
        return content

    def create_right_sidebar(self, parent_layout):
        """Create collapsible right sidebar"""
        # Right sidebar frame
        self.right_sidebar = QFrame()
        self.right_sidebar.setFixedWidth(300)
        self.right_sidebar.setStyleSheet(
            """
            QFrame {
                background-color: #0b0c0f;
            }
        """
        )

        self.right_sidebar_layout = QVBoxLayout(self.right_sidebar)
        self.right_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.right_sidebar_layout.setSpacing(0)

        # Update right sidebar content
        self.update_right_sidebar()

        parent_layout.addWidget(self.right_sidebar)

    def update_right_sidebar(self):
        """Update right sidebar content based on current tab"""
        # Clear existing content
        for i in reversed(range(self.right_sidebar_layout.count())):
            item = self.right_sidebar_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        # Add content based on current tab
        if self.current_content == "dashboard":
            self.create_dashboard_sidebar()
        elif self.current_content == "analysis":
            self.create_analysis_sidebar()
        else:
            self.create_default_sidebar()

    def create_dashboard_sidebar(self):
        """Create dashboard-specific sidebar content"""
        header = QLabel("Quick Actions")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(header)

        # Recent files or quick actions would go here
        placeholder = QLabel("Dashboard sidebar content")
        placeholder.setStyleSheet(
            """
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(placeholder)
        self.right_sidebar_layout.addStretch()

    def create_analysis_sidebar(self):
        """Create analysis-specific sidebar content"""
        header = QLabel("Analysis Tools")
        header.setStyleSheet(
            """
            color: #ffffff;
            font-size: 16px;
            font-weight: bold;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(header)

        placeholder = QLabel("Analysis tools and settings")
        placeholder.setStyleSheet(
            """
                color: #767676;
                font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(placeholder)
        self.right_sidebar_layout.addStretch()

    def create_default_sidebar(self):
        """Create default sidebar content"""
        placeholder = QLabel("Sidebar content")
        placeholder.setStyleSheet(
            """
                color: #767676;
                font-size: 14px;
            font-family: 'Minork Sans', Arial, sans-serif;
            padding: 20px;
        """
        )
        self.right_sidebar_layout.addWidget(placeholder)
        self.right_sidebar_layout.addStretch()


class ZoomableGraphicsView(QGraphicsView):
    """Enhanced Graphics View with comprehensive zoom and navigation controls"""

    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.0
        self.zoom_step = 0.15
        self.min_zoom = 0.25
        self.max_zoom = 5.0
        self.pan_mode = False
        self.grid_visible = True
        self.snap_to_grid = False
        self.last_pan_point = QPointF()

        # Setup the view
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setOptimizationFlags(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

        # Enable mouse tracking for pan operations
        self.setMouseTracking(True)

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for zoom and navigation"""
        # Zoom shortcuts
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+="), self, self.zoom_in)  # Alternative
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
        QShortcut(QKeySequence("Ctrl+9"), self, self.fit_to_view)

        # Navigation shortcuts
        QShortcut(QKeySequence("Space"), self, self.toggle_pan_mode)
        QShortcut(QKeySequence("Ctrl+G"), self, self.toggle_grid)
        QShortcut(QKeySequence("Ctrl+Shift+G"), self, self.toggle_snap)
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom with Ctrl modifier"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom in/out with Ctrl+scroll
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Normal scroll behavior
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for pan mode"""
        if self.pan_mode and event.button() == Qt.MouseButton.LeftButton:
            self.last_pan_point = event.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for pan mode"""
        if self.pan_mode and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self.last_pan_point
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            self.last_pan_point = event.position()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for pan mode"""
        if self.pan_mode:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(event)

    def zoom_in(self):
        """Zoom in by zoom_step"""
        if self.zoom_factor < self.max_zoom:
            factor = 1 + self.zoom_step
            self.scale(factor, factor)
            self.zoom_factor *= factor

    def zoom_out(self):
        """Zoom out by zoom_step"""
        if self.zoom_factor > self.min_zoom:
            factor = 1 / (1 + self.zoom_step)
            self.scale(factor, factor)
            self.zoom_factor *= factor

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.resetTransform()
        self.zoom_factor = 1.0

    def fit_to_view(self):
        """Fit entire scene to view"""
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            transform = self.transform()
            self.zoom_factor = transform.m11()  # Get scaling factor

    def toggle_pan_mode(self):
        """Toggle pan/hand tool mode"""
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def toggle_grid(self):
        """Toggle grid visibility"""
        self.grid_visible = not self.grid_visible
        self.viewport().update()

    def toggle_snap(self):
        """Toggle snap to grid"""
        self.snap_to_grid = not self.snap_to_grid

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.window().isFullScreen():
            self.window().showNormal()
        else:
            self.window().showFullScreen()

    def get_zoom_percentage(self):
        """Get current zoom as percentage"""
        return int(self.zoom_factor * 100)


class FloatingZoomControls(QWidget):
    """Floating zoom control widget with comprehensive features"""

    # Signals for communication with parent
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    reset_zoom_requested = pyqtSignal()
    fit_view_requested = pyqtSignal()
    pan_mode_toggled = pyqtSignal(bool)
    grid_toggled = pyqtSignal(bool)
    snap_toggled = pyqtSignal(bool)
    fullscreen_toggled = pyqtSignal()
    mini_map_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_percentage = 100
        self.pan_mode = False
        self.grid_visible = True
        self.snap_enabled = False
        self.mini_map_visible = False

        self.setup_ui()
        self.setup_styling()

    def setup_ui(self):
        """Setup the floating control UI with clear icons"""
        self.setFixedSize(220, 120)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Core Zoom Controls Row
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)

        # Zoom Out button
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(28, 28)
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out_requested.emit)
        zoom_row.addWidget(self.zoom_out_btn)

        # Zoom percentage display
        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(40)
        self.zoom_label.setToolTip("Current Zoom Level")
        zoom_row.addWidget(self.zoom_label)

        # Zoom In button
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(28, 28)
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in_requested.emit)
        zoom_row.addWidget(self.zoom_in_btn)

        # Reset button
        self.reset_btn = QPushButton("◯")
        self.reset_btn.setFixedSize(28, 28)
        self.reset_btn.setToolTip("Reset Zoom (100%)")
        self.reset_btn.clicked.connect(self.reset_zoom_requested.emit)
        zoom_row.addWidget(self.reset_btn)

        # Fit to view button
        self.fit_btn = QPushButton("⬜")
        self.fit_btn.setFixedSize(28, 28)
        self.fit_btn.setToolTip("Fit Field to View")
        self.fit_btn.clicked.connect(self.fit_view_requested.emit)
        zoom_row.addWidget(self.fit_btn)

        layout.addLayout(zoom_row)

        # Navigation Controls Row
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        # Pan/Hand tool toggle
        self.pan_btn = QPushButton("✋")
        self.pan_btn.setFixedSize(28, 28)
        self.pan_btn.setCheckable(True)
        self.pan_btn.setToolTip("Pan/Drag Tool")
        self.pan_btn.clicked.connect(self.toggle_pan_mode)
        nav_row.addWidget(self.pan_btn)

        # Grid toggle
        self.grid_btn = QPushButton("⊞")
        self.grid_btn.setFixedSize(28, 28)
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.setToolTip("Toggle Grid")
        self.grid_btn.clicked.connect(self.toggle_grid)
        nav_row.addWidget(self.grid_btn)

        # Snap to grid toggle
        self.snap_btn = QPushButton("⊡")
        self.snap_btn.setFixedSize(28, 28)
        self.snap_btn.setCheckable(True)
        self.snap_btn.setToolTip("Snap to Grid")
        self.snap_btn.clicked.connect(self.toggle_snap)
        nav_row.addWidget(self.snap_btn)

        # Mini-map toggle
        self.map_btn = QPushButton("◐")
        self.map_btn.setFixedSize(28, 28)
        self.map_btn.setCheckable(True)
        self.map_btn.setToolTip("Mini-Map Overview")
        self.map_btn.clicked.connect(self.toggle_mini_map)
        nav_row.addWidget(self.map_btn)

        # Fullscreen toggle
        self.fullscreen_btn = QPushButton("⤢")
        self.fullscreen_btn.setFixedSize(28, 28)
        self.fullscreen_btn.setToolTip("Toggle Fullscreen")
        self.fullscreen_btn.clicked.connect(self.fullscreen_toggled.emit)
        nav_row.addWidget(self.fullscreen_btn)

        layout.addLayout(nav_row)

    def setup_styling(self):
        """Apply styling to the floating controls"""
        # Main container styling with #565656 background
        self.setStyleSheet(
            f"""
            FloatingZoomControls {{
                background-color: rgba(86, 86, 86, 220);
                border: 1px solid #29d28c;
                border-radius: 12px;
            }}

            QPushButton {{
                background-color: #565656;
                color: #e3e3e3;
                border: 1px solid #404040;
                border-radius: 6px;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 14px;
            }}

            QPushButton:hover {{
                background-color: #6a6a6a;
                border-color: #29d28c;
            }}

            QPushButton:pressed {{
                background-color: #4a4a4a;
            }}

            QPushButton:checked {{
                background-color: #29d28c;
                color: #151515;
                border-color: #1fc47d;
            }}

            QLabel {{
                color: #e3e3e3;
                font-family: 'Minork Sans', Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                background-color: transparent;
            }}
        """
        )

    def toggle_pan_mode(self):
        """Toggle pan mode and emit signal"""
        self.pan_mode = not self.pan_mode
        self.pan_mode_toggled.emit(self.pan_mode)

    def toggle_grid(self):
        """Toggle grid visibility and emit signal"""
        self.grid_visible = not self.grid_visible
        self.grid_toggled.emit(self.grid_visible)

    def toggle_snap(self):
        """Toggle snap to grid and emit signal"""
        self.snap_enabled = not self.snap_enabled
        self.snap_toggled.emit(self.snap_enabled)

    def toggle_mini_map(self):
        """Toggle mini-map visibility and emit signal"""
        self.mini_map_visible = not self.mini_map_visible
        self.mini_map_toggled.emit(self.mini_map_visible)

    def update_zoom_display(self, percentage):
        """Update the zoom percentage display"""
        self.zoom_percentage = percentage
        self.zoom_label.setText(f"{percentage}%")

    def update_pan_mode(self, enabled):
        """Update pan mode button state"""
        self.pan_mode = enabled
        self.pan_btn.setChecked(enabled)

    def update_grid_state(self, visible):
        """Update grid button state"""
        self.grid_visible = visible
        self.grid_btn.setChecked(visible)

    def update_snap_state(self, enabled):
        """Update snap button state"""
        self.snap_enabled = enabled
        self.snap_btn.setChecked(enabled)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SpygateDesktopFaceItStyle()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
