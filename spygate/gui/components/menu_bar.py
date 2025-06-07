"""
Spygate - Menu Bar Component
"""

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QMenuBar


def create_menu_bar(parent):
    """Create the main menu bar.

    Args:
        parent: Parent window

    Returns:
        QMenuBar: The configured menu bar
    """
    menu_bar = QMenuBar(parent)
    menu_bar.setStyleSheet(
        """
        QMenuBar {
            background: #2A2A2A;
            color: #D1D5DB;
            border: none;
        }
        QMenuBar::item {
            padding: 8px 12px;
            background: transparent;
        }
        QMenuBar::item:selected {
            background: #3B82F6;
            color: #FFFFFF;
        }
        QMenu {
            background: #2A2A2A;
            color: #D1D5DB;
            border: 1px solid #3B3B3B;
            padding: 5px;
        }
        QMenu::item {
            padding: 8px 25px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background: #3B82F6;
            color: #FFFFFF;
        }
        QMenu::separator {
            height: 1px;
            background: #3B3B3B;
            margin: 5px 0;
        }
    """
    )

    # File menu
    file_menu = menu_bar.addMenu("File")

    import_action = QAction("Import Video...", parent)
    import_action.setStatusTip("Import a new video clip")
    import_action.triggered.connect(parent.show_video_import)
    file_menu.addAction(import_action)

    file_menu.addSeparator()

    exit_action = QAction("Exit", parent)
    exit_action.setStatusTip("Exit the application")
    exit_action.triggered.connect(parent.close)
    file_menu.addAction(exit_action)

    # View menu
    view_menu = menu_bar.addMenu("View")

    analysis_action = QAction("Analysis Panel", parent)
    analysis_action.setStatusTip("Toggle analysis panel visibility")
    analysis_action.setCheckable(True)
    analysis_action.triggered.connect(
        lambda checked: parent.analysis_dock.setVisible(checked)
    )
    view_menu.addAction(analysis_action)

    # Theme menu
    theme_menu = menu_bar.addMenu("Theme")

    theme_action = QAction("Theme Settings...", parent)
    theme_action.setStatusTip("Change application theme")
    theme_action.triggered.connect(
        lambda: ThemeDialog(parent.theme_manager, parent).exec()
    )
    theme_menu.addAction(theme_action)

    # Help menu
    help_menu = menu_bar.addMenu("Help")

    about_action = QAction("About Spygate", parent)
    about_action.setStatusTip("Show application information")
    about_action.triggered.connect(
        lambda: parent.status_bar.showMessage("Spygate - Football Analysis Tool", 5000)
    )
    help_menu.addAction(about_action)

    return menu_bar
