"""
Spygate - Toolbar Component
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QToolBar, QWidget


def create_main_toolbar(parent: QWidget) -> QToolBar:
    """Create the main toolbar with essential actions.

    Args:
        parent: Parent widget

    Returns:
        QToolBar: The configured toolbar
    """
    toolbar = QToolBar()
    toolbar.setMovable(False)
    toolbar.setStyleSheet(
        """
        QToolBar {
            spacing: 5px;
            background: #2A2A2A;
            border: none;
        }
        QToolButton {
            background: transparent;
            border: none;
            border-radius: 4px;
            padding: 5px;
            color: #D1D5DB;
        }
        QToolButton:hover {
            background: #3B82F6;
            color: #FFFFFF;
        }
    """
    )

    # Add actions
    home_action = QAction("Home", parent)
    home_action.setStatusTip("Return to dashboard")
    home_action.triggered.connect(parent.show_dashboard)
    toolbar.addAction(home_action)

    import_action = QAction("Import", parent)
    import_action.setStatusTip("Import video files")
    import_action.triggered.connect(parent.show_video_import)
    toolbar.addAction(import_action)

    player_action = QAction("Player", parent)
    player_action.setStatusTip("Open video player")
    player_action.triggered.connect(parent.show_video_player)
    toolbar.addAction(player_action)

    toolbar.addSeparator()

    analysis_action = QAction("Analysis", parent)
    analysis_action.setStatusTip("Toggle analysis panel")
    analysis_action.setCheckable(True)
    analysis_action.setChecked(False)
    analysis_action.triggered.connect(
        lambda checked: parent.analysis_panel.setVisible(checked)
    )
    toolbar.addAction(analysis_action)

    theme_action = QAction("Theme", parent)
    theme_action.setStatusTip("Change application theme")
    theme_action.triggered.connect(parent.show_theme_dialog)
    toolbar.addAction(theme_action)

    return toolbar
