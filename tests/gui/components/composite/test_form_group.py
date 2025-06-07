"""Test form group component."""

import pytest
from PyQt6.QtWidgets import QLineEdit

from src.gui.components.composite import FormGroup


@pytest.mark.gui
def test_form_group_creation():
    """Test creating a form group."""
    form = FormGroup()
    assert form is not None
    assert form.objectName() == ""


@pytest.mark.gui
def test_add_text_field():
    """Test adding a text field."""
    form = FormGroup()
    field = form.add_text_field("name", "Name")
    assert isinstance(field, QLineEdit)
    assert field.placeholderText() == "Name"


@pytest.mark.gui
def test_add_password_field():
    """Test adding a password field."""
    form = FormGroup()
    field = form.add_password_field("password", "Password")
    assert isinstance(field, QLineEdit)
    assert field.echoMode() == QLineEdit.Password


@pytest.mark.gui
def test_add_checkbox_field():
    """Test adding a checkbox field."""
    form = FormGroup()
    field = form.add_checkbox_field("remember", "Remember me")
    assert not field.isChecked()


@pytest.mark.gui
def test_field_validation():
    """Test field validation."""
    form = FormGroup()
    form.add_text_field("name", "Name", required=True)
    assert not form.validate()
    form.fields["name"].setText("John")
    assert form.validate()


@pytest.mark.gui
def test_form_submission():
    """Test form submission."""
    form = FormGroup()
    form.add_text_field("name", "Name")
    form.fields["name"].setText("John")
    data = form.get_data()
    assert data["name"] == "John"


@pytest.mark.gui
def test_form_reset():
    """Test form reset."""
    form = FormGroup()
    form.add_text_field("name", "Name")
    form.fields["name"].setText("John")
    form.reset()
    assert form.fields["name"].text() == ""


@pytest.mark.gui
def test_form_group_add_field():
    """Test adding a field to the form group."""
    form = FormGroup()
    field = QLineEdit()
    form.add_field("test", field)
    assert field in form.fields
