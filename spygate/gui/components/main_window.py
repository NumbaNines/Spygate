"""
Spygate - Main Window Component
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QToolBar,
    QStatusBar,
    QDockWidget,
    QMenuBar,
    QMenu,
    QDialog,
    QComboBox,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from .video_player import VideoPlayer
from .analysis_panel import AnalysisPanel
from .toolbar import create_main_toolbar
from .menu_bar import create_menu_bar
from ..themes.theme_manager import ThemeManager


class ThemeDialog(QDialog):
    """Dialog for selecting and managing themes."""
    
    def __init__(self, theme_manager: ThemeManager, parent=None):
        """Initialize the theme dialog.
        
        Args:
            theme_manager: The application's theme manager
            parent: Parent widget
        """
        super().__init__(parent)
        self.theme_manager = theme_manager
        
        self.setWindowTitle("Theme Settings")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Select Theme:")
        self.theme_combo = QComboBox()
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)
        
        # Populate themes
        themes = theme_manager.get_available_themes()
        for name, type_ in themes.items():
            self.theme_combo.addItem(f"{name} ({type_})", (name, type_ == 'custom'))
        
        # Set current theme
        current_theme = theme_manager.get_current_theme()
        index = self.theme_combo.findText(current_theme, Qt.MatchFlag.MatchStartsWith)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        # Apply button
        apply_button = QPushButton("Apply Theme")
        apply_button.clicked.connect(self._apply_theme)
        layout.addWidget(apply_button)
    
    def _apply_theme(self):
        """Apply the selected theme."""
        current_data = self.theme_combo.currentData()
        if current_data:
            name, is_custom = current_data
            self.theme_manager.apply_theme(name, custom=is_custom)


class MainWindow(QMainWindow):
    """Main window of the application."""

    def __init__(self, app):
        """Initialize the main window.
        
        Args:
            app: The QApplication instance
        """
        super().__init__()
        self.setWindowTitle("Spygate")
        self.setMinimumSize(1024, 768)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager(app)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        
        # Create menu bar
        self.menu_bar = create_menu_bar(self)
        self.setMenuBar(self.menu_bar)
        
        # Add theme menu
        self._setup_theme_menu()
        
        # Create toolbar
        self.toolbar = create_main_toolbar(self)
        self.addToolBar(self.toolbar)
        
        # Create video player
        self.video_player = VideoPlayer()
        layout.addWidget(self.video_player)
        
        # Create analysis panel as dock widget
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_panel = AnalysisPanel()
        self.analysis_dock.setWidget(self.analysis_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analysis_dock)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Apply default theme
        self.theme_manager.apply_theme("spygate_dark", custom=True)
    
    def _setup_theme_menu(self):
        """Set up the theme menu in the menu bar."""
        view_menu = self.menu_bar.addMenu("&View")
        
        # Theme settings action
        theme_action = QAction("&Theme Settings...", self)
        theme_action.setStatusTip("Configure application theme")
        theme_action.triggered.connect(self._show_theme_dialog)
        view_menu.addAction(theme_action)
    
    def _show_theme_dialog(self):
        """Show the theme selection dialog."""
        dialog = ThemeDialog(self.theme_manager, self)
        dialog.exec()

    # Set dark theme style adjustments
    def _set_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QDockWidget {
                color: #ffffff;
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(float.png);
            }
            QStatusBar {
                color: #ffffff;
            }
        """) 