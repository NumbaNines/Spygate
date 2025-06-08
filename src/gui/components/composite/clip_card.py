from datetime import datetime, timedelta

import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....services.clip_service import ClipService
from ....services.video_service import VideoService


class ClipCard(QWidget):
    previewStarted = pyqtSignal(str)  # Emits video path when preview starts
    previewStopped = pyqtSignal(str)  # Emits video path when preview stops
    clipSelected = pyqtSignal(int)  # Emits clip ID when clicked

    def __init__(self, clip_id: int, parent=None):
        super().__init__(parent)
        self.clip_id = clip_id
        self.clip_service = ClipService()
        self.video_service = VideoService()

        # Get clip data
        self.clip = self.clip_service.get_clip(clip_id)
        if not self.clip:
            raise ValueError(f"Clip with ID {clip_id} not found")

        # Preview state
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.start_preview)

        self.setup_ui()
        self.setup_preview_handling()

    def setup_ui(self):
        """Set up the card's UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thumbnail container (also holds preview)
        self.thumbnail_container = QWidget(self)
        self.thumbnail_container.setFixedSize(320, 180)  # 16:9 aspect ratio
        self.thumbnail_container.setStyleSheet("background-color: black;")

        # Thumbnail label
        self.thumbnail_label = QLabel(self.thumbnail_container)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setStyleSheet("background-color: transparent;")

        # Duration label
        self.duration_label = QLabel(self.thumbnail_container)
        self.duration_label.setStyleSheet(
            """
            QLabel {
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 2px 4px;
                border-radius: 2px;
            }
        """
        )
        self.duration_label.setText(self.format_duration(self.clip.duration))

        # Progress bar
        self.progress_bar = QProgressBar(self.thumbnail_container)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.3);
                border: none;
            }
            QProgressBar::chunk {
                background-color: #ff0000;
            }
        """
        )
        self.progress_bar.setValue(int(self.clip.watch_progress * 100))

        # Load thumbnail
        if self.clip.thumbnail_path:
            pixmap = QPixmap(self.clip.thumbnail_path)
            self.thumbnail_label.setPixmap(
                pixmap.scaled(
                    320,
                    180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        # Metadata section
        metadata_widget = QWidget(self)
        metadata_layout = QHBoxLayout(metadata_widget)
        metadata_layout.setContentsMargins(8, 8, 8, 8)

        # Title and details
        info_layout = QVBoxLayout()

        self.title_label = QLabel(self.clip.title)
        self.title_label.setStyleSheet("font-weight: bold; color: white;")
        info_layout.addWidget(self.title_label)

        details_text = f"{self.clip.player_name or 'Unknown Player'} • "
        details_text += self.format_upload_date(self.clip.created_at)
        self.details_label = QLabel(details_text)
        self.details_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        info_layout.addWidget(self.details_label)

        metadata_layout.addLayout(info_layout, stretch=1)

        # Menu button
        self.menu_button = QPushButton(self)
        self.menu_button.setIcon(QIcon.fromTheme("view-more"))
        self.menu_button.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                padding: 4px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
            }
        """
        )
        self.menu_button.clicked.connect(self.show_menu)
        metadata_layout.addWidget(self.menu_button)

        # Add all components to main layout
        layout.addWidget(self.thumbnail_container)
        layout.addWidget(metadata_widget)

        # Make the card clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            ClipCard {
                background-color: #1e1e1e;
                border-radius: 8px;
            }
            ClipCard:hover {
                background-color: #2d2d2d;
            }
        """
        )

    def setup_preview_handling(self):
        """Set up preview-related event handling"""
        # Mouse tracking for hover events
        self.thumbnail_container.setMouseTracking(True)
        self.thumbnail_container.enterEvent = self.handle_hover_enter
        self.thumbnail_container.leaveEvent = self.handle_hover_leave

        # Click handling
        self.mouseReleaseEvent = self.handle_click

    def handle_hover_enter(self, event):
        """Start preview timer when mouse enters"""
        self.preview_timer.start(500)  # 500ms delay before preview starts

    def handle_hover_leave(self, event):
        """Stop preview when mouse leaves"""
        self.preview_timer.stop()
        self.stop_preview()

    def handle_click(self, event):
        """Handle card click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clipSelected.emit(self.clip_id)

    def start_preview(self):
        """Start video preview"""
        self.previewStarted.emit(self.clip.filename)
        self.video_service.start_preview(self.clip.filename, self.update_preview_frame)

    def stop_preview(self):
        """Stop video preview"""
        self.previewStopped.emit(self.clip.filename)
        self.video_service.stop_preview(self.clip.filename)

        # Restore thumbnail
        if self.clip.thumbnail_path:
            pixmap = QPixmap(self.clip.thumbnail_path)
            self.thumbnail_label.setPixmap(
                pixmap.scaled(
                    320,
                    180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def update_preview_frame(self, frame: np.ndarray, position: float):
        """Update the preview frame"""
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width

        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.thumbnail_label.setPixmap(
            pixmap.scaled(
                320,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def show_menu(self):
        """Show the clip options menu"""
        menu = QMenu(self)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                color: white;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
        """
        )

        # Add menu actions
        menu.addAction("Add to Collection", lambda: self.handle_menu_action("add_to_collection"))
        menu.addAction("Share", lambda: self.handle_menu_action("share"))
        menu.addAction("Download", lambda: self.handle_menu_action("download"))
        menu.addSeparator()
        menu.addAction("Delete", lambda: self.handle_menu_action("delete"))

        # Show menu at button
        menu.exec(self.menu_button.mapToGlobal(QPoint(0, self.menu_button.height())))

    def handle_menu_action(self, action: str):
        """Handle menu actions"""
        if action == "add_to_collection":
            # TODO: Show collection selection dialog
            pass
        elif action == "share":
            # TODO: Show share dialog
            pass
        elif action == "download":
            # TODO: Handle clip download
            pass
        elif action == "delete":
            # TODO: Show delete confirmation dialog
            pass

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to MM:SS"""
        duration = timedelta(seconds=int(seconds))
        if duration.total_seconds() >= 3600:
            return f"{int(duration.total_seconds() // 3600)}:{duration.strftime('%M:%S')}"
        return duration.strftime("%M:%S")

    @staticmethod
    def format_upload_date(date: datetime) -> str:
        """Format upload date relative to now"""
        now = datetime.utcnow()
        diff = now - date

        if diff.days == 0:
            if diff.seconds < 3600:
                minutes = diff.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.days < 30:
            weeks = diff.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif diff.days < 365:
            months = diff.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = diff.days // 365
            return f"{years} year{'s' if years != 1 else ''} ago"
