from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..services.motion_service import MotionService
from ..video.frame_extractor import FrameExtractor
from .video_timeline_view import VideoTimelineView


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Spygate - Game Analysis")

        # Initialize database
        engine = create_engine("sqlite:///spygate.db")
        Session = sessionmaker(bind=engine)
        self.db_session = Session()

        # Initialize services
        self.frame_extractor = FrameExtractor()
        self.motion_service = MotionService(self.db_session)

        # Initialize UI
        self._init_ui()

        # Set window properties
        self.setMinimumSize(1024, 768)
        self.showMaximized()

    def _init_ui(self):
        """Initialize the UI layout and components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        layout = QVBoxLayout(central_widget)

        # Create video timeline view
        self.timeline_view = VideoTimelineView(
            self.frame_extractor, self.motion_service, self
        )
        layout.addWidget(self.timeline_view)

    def closeEvent(self, event):
        """Handle application close event."""
        # Clean up resources
        self.frame_extractor.cleanup()
        self.db_session.close()
        super().closeEvent(event)
