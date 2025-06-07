"""
Spygate - Clip Card Component
Displays a video clip with thumbnail, title, player name, and tags
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ClipCard(QFrame):
    """Card widget for displaying a video clip."""

    # Signals
    clicked = pyqtSignal()  # Emitted when the card is clicked

    def __init__(self, clip_data):
        """Initialize the clip card.

        Args:
            clip_data (dict): Data for the clip including:
                - title (str): Clip title
                - player_name (str): Name of the player ("Self" or "Opponent: Name")
                - tags (list): List of tags
                - thumbnail_path (str): Path to thumbnail image
                - has_mistakes (bool): Whether the clip has detected mistakes
        """
        super().__init__()
        self.clip_data = clip_data
        self.title = clip_data.get("title", "Untitled Clip")
        self.player_name = clip_data.get("player_name", "Unknown")
        self.tags = clip_data.get("tags", [])

        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI components."""
        # Set frame style
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setFixedSize(320, 240)  # Fixed size for consistent grid

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Thumbnail
        thumbnail = QLabel()
        if "thumbnail_path" in self.clip_data:
            pixmap = QPixmap(self.clip_data["thumbnail_path"])
            thumbnail.setPixmap(
                pixmap.scaled(320, 180, Qt.AspectRatioMode.KeepAspectRatio)
            )
        else:
            # Use placeholder image
            thumbnail.setText("No Thumbnail")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail.setStyleSheet("background-color: #2A2A2A;")
        layout.addWidget(thumbnail)

        # Info section
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)

        # Title and mistake indicator
        title_layout = QHBoxLayout()
        title_label = QLabel(self.title)
        title_label.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        title_layout.addWidget(title_label)

        if self.clip_data.get("has_mistakes", False):
            mistake_label = QLabel("●")  # Red dot for mistakes
            mistake_label.setStyleSheet("color: #EF4444; font-size: 16px;")
            mistake_label.setToolTip("This clip contains detected mistakes")
            title_layout.addWidget(mistake_label)

        info_layout.addLayout(title_layout)

        # Player name
        player_label = QLabel(self.player_name)
        player_label.setStyleSheet("color: #D1D5DB;")
        info_layout.addWidget(player_label)

        # Tags
        if self.tags:
            tags_layout = QHBoxLayout()
            for tag in self.tags[:3]:  # Show up to 3 tags
                tag_label = QLabel(tag)
                tag_label.setStyleSheet(
                    """
                    background: #3B82F6;
                    color: #FFFFFF;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 10px;
                """
                )
                tags_layout.addWidget(tag_label)
            tags_layout.addStretch()
            info_layout.addLayout(tags_layout)

        layout.addWidget(info_widget)

        # Make card clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Apply styles
        self._apply_styles()

    def _apply_styles(self):
        """Apply styles to the card."""
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                border: none;
            }
            QFrame:hover {
                background: #3B3B3B;
            }
            QLabel {
                background: transparent;
            }
        """
        )

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
