"""
Spygate - Main Window Component
"""

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..components.composite.community_view import CommunityView
from ..components.composite.dashboard import Dashboard
from ..components.composite.playbooks_view import PlaybooksView
from ..themes.theme_manager import ThemeManager
from ..video.video_import_widget import VideoImportWidget
from .analysis_panel import AnalysisPanel
from .menu_bar import create_menu_bar
from .sidebar import Sidebar
from .toolbar import create_main_toolbar
from .video_player import VideoPlayer


class ThemeDialog(QDialog):
    """Dialog for selecting and managing themes."""

    def __init__(self, theme_manager, parent=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.setWindowTitle("Theme Settings")
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Theme selector
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Select Theme:")
        self.theme_combo = QComboBox()
        themes = self.theme_manager.get_available_themes()
        self.theme_combo.addItems(themes.keys())
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        # Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_theme)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def apply_theme(self):
        """Apply the selected theme."""
        theme_name = self.theme_combo.currentText()
        self.theme_manager.apply_theme(theme_name)
        self.accept()


class MainWindow(QMainWindow):
    """Main window of the Spygate application."""

    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()

        # Initialize UI components
        self.dashboard = None
        self.video_player = None
        self.analysis_panel = None
        self.video_import = None
        self.analysis_dock = None
        self.sidebar = None
        self.playbooks_view = None
        self.community_view = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up the main window UI."""
        # Window properties
        self.setWindowTitle("Spygate - Football Analysis")
        self.setMinimumSize(1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setCentralWidget(main_widget)

        # Create and add sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Create stacked widget for main content
        self.central_widget = QStackedWidget()
        main_layout.addWidget(self.central_widget)

        # Create components
        self._create_dashboard()
        self._create_video_player()
        self._create_analysis_panel()
        self._create_video_import()
        self._create_playbooks_view()
        self._create_community_view()

        # Create toolbar and menu
        self.toolbar = create_main_toolbar(self)
        self.addToolBar(self.toolbar)
        self.menu_bar = create_menu_bar(self)
        self.setMenuBar(self.menu_bar)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect sidebar signals
        self._connect_sidebar_signals()

        # Show dashboard by default
        self.show_dashboard()

        # Apply theme
        self.theme_manager.apply_theme("dark_teal")

    def _create_dashboard(self):
        """Create the dashboard view."""
        self.dashboard = Dashboard()
        self.central_widget.addWidget(self.dashboard)

    def _create_video_player(self):
        """Create the video player view."""
        self.video_player = VideoPlayer()
        self.central_widget.addWidget(self.video_player)

    def _create_analysis_panel(self):
        """Create the analysis panel."""
        self.analysis_panel = AnalysisPanel()
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setWidget(self.analysis_panel)
        self.analysis_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)
        self.analysis_dock.hide()

    def _create_video_import(self):
        """Create the video import view."""
        self.video_import = VideoImportWidget()
        self.central_widget.addWidget(self.video_import)

    def _create_playbooks_view(self):
        """Create the playbooks view."""
        self.playbooks_view = PlaybooksView()
        self.central_widget.addWidget(self.playbooks_view)

    def _create_community_view(self):
        """Create the community view."""
        self.community_view = CommunityView()
        self.central_widget.addWidget(self.community_view)

    def _connect_sidebar_signals(self):
        """Connect sidebar navigation signals."""
        self.sidebar.dashboard_clicked.connect(self.show_dashboard)
        self.sidebar.video_clicked.connect(self.show_video_player)
        self.sidebar.import_clicked.connect(self.show_video_import)
        self.sidebar.playbooks_clicked.connect(self.show_playbooks)
        self.sidebar.community_clicked.connect(self.show_community)

    def show_dashboard(self):
        """Show the dashboard view."""
        self.central_widget.setCurrentWidget(self.dashboard)
        self.analysis_dock.hide()
        self.sidebar.set_active("dashboard")

    def show_video_player(self):
        """Show the video player view."""
        self.central_widget.setCurrentWidget(self.video_player)
        self.analysis_dock.show()
        self.sidebar.set_active("video")

    def show_video_import(self):
        """Show the video import view."""
        self.central_widget.setCurrentWidget(self.video_import)
        self.analysis_dock.hide()
        self.sidebar.set_active("import")

    def show_playbooks(self):
        """Show the playbooks view."""
        self.central_widget.setCurrentWidget(self.playbooks_view)
        self.analysis_dock.hide()
        self.sidebar.set_active("playbooks")

    def show_community(self):
        """Show the community view."""
        self.central_widget.setCurrentWidget(self.community_view)
        self.analysis_dock.hide()
        self.sidebar.set_active("community")
