"""
Spygate - Playbooks View Component
Manages playbook creation, editing, and organization
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class PlaybookItem(QFrame):
    """Individual playbook item widget."""

    def __init__(self, name, description="", parent=None):
        super().__init__(parent)
        self.name = name
        self.description = description
        self._setup_ui()

    def _setup_ui(self):
        """Set up the playbook item UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
                padding: 12px;
            }
            QLabel {
                color: #D1D5DB;
            }
            QPushButton {
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )

        layout = QVBoxLayout(self)

        # Playbook name
        name_label = QLabel(self.name)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(name_label)

        # Description
        if self.description:
            desc_label = QLabel(self.description)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        # Action buttons
        button_layout = QHBoxLayout()

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_playbook)
        button_layout.addWidget(edit_btn)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_playbook)
        button_layout.addWidget(export_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet(
            """
            QPushButton {
                background: #DC2626;
            }
            QPushButton:hover {
                background: #B91C1C;
            }
        """
        )
        delete_btn.clicked.connect(self._delete_playbook)
        button_layout.addWidget(delete_btn)

        layout.addLayout(button_layout)

    def _edit_playbook(self):
        """Open playbook editor."""
        # TODO: Implement playbook editor
        pass

    def _export_playbook(self):
        """Export playbook to file."""
        # TODO: Implement playbook export
        pass

    def _delete_playbook(self):
        """Delete this playbook."""
        confirm = QMessageBox.question(
            self,
            "Delete Playbook",
            f"Are you sure you want to delete '{self.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            # TODO: Implement actual deletion
            self.deleteLater()


class PlaybooksView(QWidget):
    """Main playbooks management view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the playbooks view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header section
        header_layout = QHBoxLayout()

        title = QLabel("Playbooks")
        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: #D1D5DB;
        """
        )
        header_layout.addWidget(title)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search playbooks...")
        self.search_bar.setStyleSheet(
            """
            QLineEdit {
                background: #2A2A2A;
                color: #D1D5DB;
                border: 2px solid #3B82F6;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
        """
        )
        self.search_bar.textChanged.connect(self._filter_playbooks)
        header_layout.addWidget(self.search_bar)

        # New playbook button
        new_btn = QPushButton("New Playbook")
        new_btn.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )
        new_btn.clicked.connect(self._create_new_playbook)
        header_layout.addWidget(new_btn)

        layout.addLayout(header_layout)

        # Playbooks grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: transparent;
            }
        """
        )

        scroll_content = QWidget()
        self.grid_layout = QVBoxLayout(scroll_content)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(16)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Add some sample playbooks
        self._add_sample_playbooks()

    def _add_sample_playbooks(self):
        """Add sample playbooks for testing."""
        sample_playbooks = [
            ("Tournament Prep", "Strategies and formations for upcoming tournament"),
            ("Defense Analysis", "Breakdown of defensive plays and counters"),
            ("Offensive Schemes", "Core offensive plays and variations"),
        ]

        for name, desc in sample_playbooks:
            self._add_playbook(name, desc)

    def _add_playbook(self, name, description=""):
        """Add a new playbook item to the view."""
        playbook = PlaybookItem(name, description)
        self.grid_layout.addWidget(playbook)

    def _create_new_playbook(self):
        """Create a new playbook."""
        name, ok = QInputDialog.getText(
            self, "New Playbook", "Enter playbook name:", QLineEdit.EchoMode.Normal
        )

        if ok and name:
            self._add_playbook(name)

    def _filter_playbooks(self):
        """Filter playbooks based on search text."""
        search_text = self.search_bar.text().lower()

        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and isinstance(item.widget(), PlaybookItem):
                playbook = item.widget()
                matches = search_text in playbook.name.lower()
                playbook.setVisible(matches)
