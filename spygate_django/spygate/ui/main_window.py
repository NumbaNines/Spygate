from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QPushButton, QToolBar, QVBoxLayout, QWidget
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..services.motion_service import MotionService
from ..video.frame_extractor import FrameExtractor
from .components.tutorial_manager import TutorialManager
from .components.tutorial_system import TutorialSystem
from .video_timeline_view import VideoTimelineView


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("SpygateAI")
        self.setMinimumSize(1024, 768)

        # Initialize database
        engine = create_engine("sqlite:///spygate.db")
        Session = sessionmaker(bind=engine)
        self.db_session = Session()

        # Initialize services
        self.frame_extractor = FrameExtractor()
        self.motion_service = MotionService(self.db_session)

        # Initialize tutorial system
        self.tutorial_manager = TutorialManager()
        self.tutorial_system = TutorialSystem(self)
        self.tutorial_system.tutorial_completed.connect(self.on_tutorial_completed)

        # Set up the main UI
        self.setup_ui()

        # Start welcome tutorial if it hasn't been completed
        if not self.tutorial_manager.is_tutorial_completed("getting_started"):
            self.show_welcome_tutorial()

        # Set window properties
        self.showMaximized()

    def setup_ui(self):
        """Set up the main UI layout and components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar = QToolBar()
        toolbar.setObjectName("main_toolbar")
        self.addToolBar(toolbar)

        # Add toolbar buttons
        import_btn = QPushButton("Import")
        import_btn.setObjectName("import_button")
        toolbar.addWidget(import_btn)

        # Add help button
        help_btn = QPushButton("?")
        help_btn.setObjectName("help_button")
        help_btn.clicked.connect(self.show_help)
        toolbar.addWidget(help_btn)

        # Create video timeline view
        self.timeline_view = VideoTimelineView(self.frame_extractor, self.motion_service, self)
        layout.addWidget(self.timeline_view)

        # Create analysis panel
        analysis = QWidget()
        analysis.setObjectName("analysis_panel")
        layout.addWidget(analysis)

        # Set up keyboard shortcuts
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for the main window."""
        # Help shortcut
        help_action = QAction("Help", self)
        help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_action.triggered.connect(self.show_help)
        self.addAction(help_action)

    def show_welcome_tutorial(self):
        """Show the welcome tutorial for new users."""
        steps = self.tutorial_manager.create_tutorial_steps("getting_started", self)
        if steps:
            self.tutorial_system.start_tutorial("getting_started", steps)

    def show_help(self):
        """Show the help dialog with tutorial options."""
        tutorials = self.tutorial_manager.get_available_tutorials()
        if not tutorials:
            QMessageBox.information(self, "Help", "No tutorials available.")
            return

        # Show available tutorials
        msg = QMessageBox(self)
        msg.setWindowTitle("SpygateAI Help")
        msg.setText("Available Tutorials:")
        msg.setInformativeText(
            "\n".join(
                f"- {t.title}: {t.description}"
                + (" (Completed)" if self.tutorial_manager.is_tutorial_completed(t.id) else "")
                for t in tutorials
            )
        )

        # Add buttons for each tutorial
        for tutorial in tutorials:
            button = msg.addButton(f"Start {tutorial.title}", QMessageBox.ButtonRole.ActionRole)
            button.clicked.connect(lambda t=tutorial: self.start_tutorial(t.id))

        msg.addButton(QMessageBox.StandardButton.Close)
        msg.exec()

    def start_tutorial(self, tutorial_id):
        """Start a specific tutorial."""
        steps = self.tutorial_manager.create_tutorial_steps(tutorial_id, self)
        if steps:
            self.tutorial_system.start_tutorial(tutorial_id, steps)

    def on_tutorial_completed(self, tutorial_id):
        """Handle tutorial completion."""
        self.tutorial_manager.mark_tutorial_completed(tutorial_id)

        # Show completion message
        tutorial = self.tutorial_manager.get_tutorial(tutorial_id)
        if tutorial and tutorial.completion_reward:
            QMessageBox.information(
                self,
                "Tutorial Completed!",
                f"Congratulations! You've completed the {tutorial.title} tutorial.\n\n"
                f"Reward: {tutorial.completion_reward}",
            )

        # Check for next tutorial
        next_tutorial = self.tutorial_manager.get_next_tutorial()
        if next_tutorial:
            response = QMessageBox.question(
                self,
                "Continue Learning?",
                f"Would you like to start the next tutorial: {next_tutorial.title}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response == QMessageBox.StandardButton.Yes:
                self.start_tutorial(next_tutorial.id)

    def closeEvent(self, event):
        """Handle application close event."""
        # Clean up resources
        self.frame_extractor.cleanup()
        self.db_session.close()
        super().closeEvent(event)
