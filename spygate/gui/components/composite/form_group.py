from typing import Dict, List, Optional, Any, Callable
from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QComboBox, 
    QSpinBox, QCheckBox, QPushButton, QVBoxLayout
)
from PyQt6.QtCore import pyqtSignal

from ..base.base_widget import BaseWidget
from ...layouts.layout_manager import LayoutManager
from ...layouts.layout_config import LayoutConfig

class FormField:
    """Represents a single form field with its configuration and validation."""
    def __init__(
        self,
        name: str,
        field_type: str,
        label: str,
        required: bool = False,
        default_value: Any = None,
        options: List[str] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        error_message: str = ""
    ):
        self.name = name
        self.field_type = field_type
        self.label = label
        self.required = required
        self.default_value = default_value
        self.options = options or []
        self.validator = validator
        self.error_message = error_message
        self.widget = None
        self.error_label = None

class FormGroup(BaseWidget):
    """A composite component for creating and managing forms."""
    
    # Signals
    submitted = pyqtSignal(dict)  # Emits form data when submitted
    validated = pyqtSignal(bool)  # Emits validation status
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        fields: List[FormField] = None,
        layout_config: Optional[LayoutConfig] = None
    ):
        super().__init__(parent)
        self.fields = fields or []
        self.layout_config = layout_config or LayoutConfig(
            margins=(10, 10, 10, 10),
            spacing=10
        )
        self.field_widgets: Dict[str, QWidget] = {}
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """Initialize the form layout and create field widgets."""
        layout = LayoutManager.create_vertical(self, self.layout_config)
        
        for field in self.fields:
            # Create field container
            field_container = QWidget(self)
            field_layout = QVBoxLayout(field_container)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(5)
            
            # Add label
            label = QLabel(field.label, field_container)
            if field.required:
                label.setText(f"{field.label} *")
            field_layout.addWidget(label)
            
            # Create and add field widget
            widget = self._create_field_widget(field)
            field.widget = widget
            field_layout.addWidget(widget)
            
            # Add error label (hidden by default)
            error_label = QLabel(field_container)
            error_label.setStyleSheet("color: red;")
            error_label.hide()
            field.error_label = error_label
            field_layout.addWidget(error_label)
            
            layout.addWidget(field_container)
            self.field_widgets[field.name] = widget
        
        # Add submit button
        submit_btn = QPushButton("Submit", self)
        submit_btn.clicked.connect(self.submit)
        layout.addWidget(submit_btn)
        
    def _create_field_widget(self, field: FormField) -> QWidget:
        """Create the appropriate widget based on field type."""
        if field.field_type == "text":
            widget = QLineEdit(self)
            if field.default_value:
                widget.setText(str(field.default_value))
                
        elif field.field_type == "select":
            widget = QComboBox(self)
            widget.addItems(field.options)
            if field.default_value and field.default_value in field.options:
                widget.setCurrentText(field.default_value)
                
        elif field.field_type == "number":
            widget = QSpinBox(self)
            if field.default_value is not None:
                widget.setValue(int(field.default_value))
                
        elif field.field_type == "checkbox":
            widget = QCheckBox(self)
            if field.default_value:
                widget.setChecked(bool(field.default_value))
                
        else:
            widget = QLineEdit(self)  # Default to text input
            
        return widget
    
    def get_values(self) -> Dict[str, Any]:
        """Get the current values of all form fields."""
        values = {}
        for field in self.fields:
            widget = self.field_widgets[field.name]
            
            if isinstance(widget, QLineEdit):
                values[field.name] = widget.text()
            elif isinstance(widget, QComboBox):
                values[field.name] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                values[field.name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[field.name] = widget.isChecked()
                
        return values
    
    def validate(self) -> bool:
        """Validate all form fields."""
        is_valid = True
        
        for field in self.fields:
            value = self.field_widgets[field.name].text()
            field.error_label.hide()
            
            # Check required fields
            if field.required and not value:
                is_valid = False
                field.error_label.setText("This field is required")
                field.error_label.show()
                continue
            
            # Run custom validation if provided
            if field.validator and value:
                try:
                    if not field.validator(value):
                        is_valid = False
                        field.error_label.setText(field.error_message or "Invalid value")
                        field.error_label.show()
                except Exception as e:
                    is_valid = False
                    field.error_label.setText(str(e))
                    field.error_label.show()
        
        self.validated.emit(is_valid)
        return is_valid
    
    def submit(self) -> None:
        """Validate and submit form data."""
        if self.validate():
            self.submitted.emit(self.get_values())
    
    def reset(self) -> None:
        """Reset all form fields to their default values."""
        for field in self.fields:
            widget = self.field_widgets[field.name]
            
            if isinstance(widget, QLineEdit):
                widget.setText(str(field.default_value or ""))
            elif isinstance(widget, QComboBox):
                if field.default_value in field.options:
                    widget.setCurrentText(field.default_value)
                else:
                    widget.setCurrentIndex(0)
            elif isinstance(widget, QSpinBox):
                widget.setValue(field.default_value or 0)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(field.default_value))
            
            field.error_label.hide() 