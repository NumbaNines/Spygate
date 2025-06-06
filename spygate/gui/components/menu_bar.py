"""
Spygate - Menu Bar Component
"""

from PyQt6.QtWidgets import QMenuBar, QStyle
from PyQt6.QtGui import QAction


def create_menu_bar(parent) -> QMenuBar:
    """Create the application menu bar.
    
    Args:
        parent: Parent window that will own the menu bar
        
    Returns:
        QMenuBar: The configured menu bar
    """
    menubar = QMenuBar()
    
    # File menu
    file_menu = menubar.addMenu("&File")
    
    # Open video action
    open_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
        "&Open Video...",
        parent
    )
    open_action.setShortcut("Ctrl+O")
    file_menu.addAction(open_action)
    
    file_menu.addSeparator()
    
    # Save analysis action
    save_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        "&Save Analysis...",
        parent
    )
    save_action.setShortcut("Ctrl+S")
    file_menu.addAction(save_action)
    
    # Export action
    export_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon),
        "&Export Data...",
        parent
    )
    export_action.setShortcut("Ctrl+E")
    file_menu.addAction(export_action)
    
    file_menu.addSeparator()
    
    # Exit action
    exit_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton),
        "E&xit",
        parent
    )
    exit_action.setShortcut("Alt+F4")
    file_menu.addAction(exit_action)
    
    # Analysis menu
    analysis_menu = menubar.addMenu("&Analysis")
    
    # Start analysis action
    start_analysis_action = QAction("&Start Analysis", parent)
    start_analysis_action.setShortcut("F5")
    analysis_menu.addAction(start_analysis_action)
    
    # Stop analysis action
    stop_analysis_action = QAction("St&op Analysis", parent)
    stop_analysis_action.setShortcut("F6")
    stop_analysis_action.setEnabled(False)
    analysis_menu.addAction(stop_analysis_action)
    
    analysis_menu.addSeparator()
    
    # Settings action
    settings_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
        "&Settings...",
        parent
    )
    settings_action.setShortcut("Ctrl+,")
    analysis_menu.addAction(settings_action)
    
    # Help menu
    help_menu = menubar.addMenu("&Help")
    
    # About action
    about_action = QAction("&About Spygate", parent)
    help_menu.addAction(about_action)
    
    return menubar 