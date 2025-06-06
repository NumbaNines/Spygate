"""
Spygate - Toolbar Component
"""

from PyQt6.QtWidgets import QToolBar, QStyle
from PyQt6.QtGui import QAction


def create_main_toolbar(parent) -> QToolBar:
    """Create the main toolbar with common actions.
    
    Args:
        parent: Parent window that will own the toolbar
        
    Returns:
        QToolBar: The configured toolbar
    """
    toolbar = QToolBar()
    toolbar.setMovable(False)
    
    # Open video action
    open_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
        "Open Video",
        parent
    )
    open_action.setStatusTip("Open a video file")
    toolbar.addAction(open_action)
    
    toolbar.addSeparator()
    
    # Save analysis action
    save_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        "Save Analysis",
        parent
    )
    save_action.setStatusTip("Save analysis results")
    toolbar.addAction(save_action)
    
    # Export action
    export_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_FileLinkIcon),
        "Export",
        parent
    )
    export_action.setStatusTip("Export analysis data")
    toolbar.addAction(export_action)
    
    toolbar.addSeparator()
    
    # Settings action
    settings_action = QAction(
        parent.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
        "Settings",
        parent
    )
    settings_action.setStatusTip("Application settings")
    toolbar.addAction(settings_action)
    
    return toolbar 