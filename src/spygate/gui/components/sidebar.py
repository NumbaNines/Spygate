"""
Spygate - Sidebar Component
Navigation sidebar with collapsible functionality
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class Sidebar(QFrame):
    """Collapsible sidebar for navigation."""

    # Navigation signals
    home_clicked = pyqtSignal()
    upload_clicked = pyqtSignal()
    clips_clicked = pyqtSignal()
    analytics_clicked = pyqtSignal()
    playbooks_clicked = pyqtSignal()
    community_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.expanded = True
        self.setFixedWidth(200)  # Initial width

        # Define icons for each button (temporary examples)
        self.icons = {
            "home": "🏠",
            "upload": "⬆️",
            "clips": "🎬",
            "analytics": "📊",
            "playbooks": "📖",
            "community": "👥",
            "settings": "⚙️",
        }

        self._setup_ui()

    def _setup_ui(self):
        """Set up the sidebar UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Logo section
        logo_widget = QWidget()
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(16, 16, 16, 16)

        logo_label = QLabel("Spygate")
        logo_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            color: #FFFFFF;
        """
        )
        logo_layout.addWidget(logo_label)

        layout.addWidget(logo_widget)

        # Navigation buttons
        self.nav_buttons = {
            "home": ("Home", self.home_clicked),
            "upload": ("Upload", self.upload_clicked),
            "clips": ("Clips", self.clips_clicked),
            "analytics": ("Analytics", self.analytics_clicked),
            "playbooks": ("Playbooks", self.playbooks_clicked),
            "community": ("Community", self.community_clicked),
            "settings": ("Settings", self.settings_clicked),
        }

        for key, (text, signal) in self.nav_buttons.items():
            # Create container widget for button content
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(16, 8, 16, 8)
            container_layout.setSpacing(16)

            # Create icon label (initially hidden)
            icon_label = QLabel(self.icons[key])
            icon_label.setFixedSize(24, 24)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet(
                """
                QLabel {
                    font-size: 16px;
                }
            """
            )
            icon_label.hide()  # Initially hidden in expanded state
            container_layout.addWidget(icon_label)

            # Create text label
            text_label = QLabel(text)
            text_label.setStyleSheet(
                """
                QLabel {
                    color: #D1D5DB;
                    font-size: 14px;
                    font-weight: 500;
                }
            """
            )
            container_layout.addWidget(text_label)

            # Add stretch to push content to the left
            container_layout.addStretch()

            # Create button
            btn = QPushButton()
            btn.setCheckable(True)
            btn.clicked.connect(signal)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    border: none;
                    border-radius: 4px;
                    background: transparent;
                    margin: 2px 8px;
                    outline: none;
                    min-height: 44px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.05);
                }
                QPushButton:checked {
                    background: rgba(255, 255, 255, 0.1);
                }
                QPushButton:checked QLabel {
                    color: #FFFFFF;
                    font-weight: 600;
                }
            """
            )

            # Set the container as the button's content
            btn.setLayout(container_layout)
            layout.addWidget(btn)

            # Store references
            setattr(self, f"{key}_button", btn)
            setattr(self, f"{key}_icon", icon_label)
            setattr(self, f"{key}_text", text_label)

        # Add stretch to push buttons to top
        layout.addStretch()

        # Toggle button at bottom
        self.toggle_button = QPushButton("◀")  # Left arrow
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setStyleSheet(
            """
            QPushButton {
                padding: 12px;
                border: none;
                border-radius: 4px;
                color: #D1D5DB;
                background: transparent;
                margin: 2px 8px;
                outline: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.05);
                color: #FFFFFF;
            }
            QPushButton:focus {
                outline: none;
            }
        """
        )
        layout.addWidget(self.toggle_button)

        # Apply styles
        self._apply_styles()

    def _apply_styles(self):
        """Apply styles to the sidebar."""
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-right: 1px solid #3B3B3B;
            }
            QFrame:focus {
                outline: none;
            }
        """
        )

    def toggle_sidebar(self):
        """Toggle sidebar expansion state."""
        if self.expanded:
            self.setFixedWidth(70)
            self.toggle_button.setText("▶")

            # Hide text, show icons
            for key in self.nav_buttons:
                text_label = getattr(self, f"{key}_text")
                icon_label = getattr(self, f"{key}_icon")
                text_label.hide()
                icon_label.show()
        else:
            self.setFixedWidth(200)
            self.toggle_button.setText("◀")

            # Show text, hide icons
            for key in self.nav_buttons:
                text_label = getattr(self, f"{key}_text")
                icon_label = getattr(self, f"{key}_icon")
                text_label.show()
                icon_label.hide()

        self.expanded = not self.expanded

    def set_active(self, section):
        """Set the active navigation section."""
        # Reset all buttons
        for key in self.nav_buttons:
            button = getattr(self, f"{key}_button")
            button.setChecked(False)

        # Set active button
        if section in self.nav_buttons:
            button = getattr(self, f"{section}_button")
            button.setChecked(True)
