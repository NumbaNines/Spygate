"""Form group component for organizing form fields."""

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..base import BaseWidget


class FormGroup(BaseWidget):
    """A form group component for organizing form fields."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the form group.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.fields: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self._required_fields: dict[str, bool] = {}
        self.setObjectName("")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the form group UI."""
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)

    def add_field(self, name: str, widget: QWidget, label_text: Optional[str] = None) -> None:
        """Add a field to the form.

        Args:
            name: Field name
            widget: Field widget
            label_text: Optional label text
        """
        row = self.layout.rowCount()

        if label_text:
            label = QLabel(label_text)
            self.labels[name] = label
            self.layout.addWidget(label, row, 0)
            self.layout.addWidget(widget, row, 1)
        else:
            self.layout.addWidget(widget, row, 0, 1, 2)

        self.fields[name] = widget

    def add_text_field(self, name: str, label: str, required: bool = False) -> QLineEdit:
        """Add a text field to the form.

        Args:
            name: Field name
            label: Label text
            required: Whether the field is required

        Returns:
            The created QLineEdit
        """
        field = QLineEdit()
        field.setPlaceholderText(label)
        self.add_field(name, field, label)
        self._required_fields[name] = required
        return field

    def add_password_field(self, name: str, label: str, required: bool = False) -> QLineEdit:
        """Add a password field to the form.

        Args:
            name: Field name
            label: Label text
            required: Whether the field is required

        Returns:
            The created QLineEdit
        """
        field = QLineEdit()
        field.setPlaceholderText(label)
        field.setEchoMode(QLineEdit.EchoMode.Password)
        self.add_field(name, field, label)
        self._required_fields[name] = required
        return field

    def add_checkbox_field(self, name: str, label: str) -> QCheckBox:
        """Add a checkbox field to the form.

        Args:
            name: Field name
            label: Label text

        Returns:
            The created QCheckBox
        """
        field = QCheckBox(label)
        self.add_field(name, field)
        return field

    def validate(self) -> bool:
        """Validate the form.

        Returns:
            True if all required fields are filled, False otherwise
        """
        for name, required in self._required_fields.items():
            if required:
                field = self.fields[name]
                if isinstance(field, QLineEdit) and not field.text():
                    return False
                elif isinstance(field, QCheckBox) and not field.isChecked():
                    return False
        return True

    def get_data(self) -> dict[str, Any]:
        """Get form data.

        Returns:
            Dictionary of field values
        """
        data = {}
        for name, field in self.fields.items():
            if isinstance(field, QLineEdit):
                data[name] = field.text()
            elif isinstance(field, QCheckBox):
                data[name] = field.isChecked()
        return data

    def reset(self) -> None:
        """Reset all form fields."""
        for field in self.fields.values():
            if isinstance(field, QLineEdit):
                field.clear()
            elif isinstance(field, QCheckBox):
                field.setChecked(False)

    def update_theme(self, theme: dict[str, Any]) -> None:
        """Update the form's theme.

        Args:
            theme: Theme dictionary with style properties
        """
        super().update_theme(theme)
        for widget in list(self.fields.values()) + list(self.labels.values()):
            if hasattr(widget, "update_theme"):
                widget.update_theme(theme)
