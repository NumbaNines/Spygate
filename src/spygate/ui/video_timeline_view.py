from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ..services.motion_service import MotionService
from ..video.frame_extractor import FrameExtractor
from .motion_analysis_view import MotionAnalysisView
from .timeline_view import TimelineView


class VideoTimelineView(QWidget):
    """Widget combining timeline and motion analysis views."""

    def __init__(
        self,
        frame_extractor: FrameExtractor,
        motion_service: MotionService,
        parent: QWidget = None,
    ):
        """Initialize the video timeline view."""
        super().__init__(parent)
        self.frame_extractor = frame_extractor
        self.motion_service = motion_service

        # Initialize UI
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI layout and components."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()

        # Timeline tab
        self.timeline_view = TimelineView(self.frame_extractor)
        self.tabs.addTab(self.timeline_view, "Timeline")

        # Motion analysis tab
        self.motion_view = MotionAnalysisView(self.motion_service)
        self.tabs.addTab(self.motion_view, "Motion Analysis")

        layout.addWidget(self.tabs)

        # Connect signals
        self.timeline_view.frame_changed.connect(self._on_frame_changed)

    def _on_frame_changed(self, frame_data):
        """Handle frame change events from timeline."""
        if self.motion_view and frame_data:
            self.motion_view.update_video(
                frame_data["video_id"],
                frame_data["frame"],
                frame_data["time"],
                frame_data["fps"],
            )
