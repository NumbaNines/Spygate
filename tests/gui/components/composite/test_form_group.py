import pytest
from PyQt6.QtWidgets import QApplication
from spygate.gui.components.composite import FormGroup

@pytest.fixture
def app(qtbot):
    """Create a Qt Application."""
    return QApplication.instance() or QApplication([])

@pytest.fixture
def form_group(app, qtbot):
    """Create a FormGroup instance."""
    form = FormGroup()
    qtbot.addWidget(form)
    return form

def test_form_group_creation(form_group):
    """Test that FormGroup is created properly."""
    assert form_group is not None
    assert isinstance(form_group, FormGroup)

def test_add_text_field(form_group):
    """Test adding a text field to the form."""
    form_group.add_field("username", "text", label="Username")
    assert "username" in form_group.fields
    assert form_group.fields["username"].field_type == "text"

def test_add_password_field(form_group):
    """Test adding a password field to the form."""
    form_group.add_field("password", "password", label="Password")
    assert "password" in form_group.fields
    assert form_group.fields["password"].field_type == "password"

def test_add_checkbox_field(form_group):
    """Test adding a checkbox field to the form."""
    form_group.add_field("remember", "checkbox", label="Remember me")
    assert "remember" in form_group.fields
    assert form_group.fields["remember"].field_type == "checkbox"

def test_field_validation(form_group):
    """Test field validation."""
    form_group.add_field("username", "text", label="Username")
    form_group.set_validation("username", lambda x: len(x) >= 3, "Username must be at least 3 characters")
    
    # Test invalid input
    form_group.fields["username"].setText("ab")
    assert not form_group.validate()
    
    # Test valid input
    form_group.fields["username"].setText("user123")
    assert form_group.validate()

def test_form_submission(form_group, qtbot):
    """Test form submission."""
    # Setup form
    form_group.add_field("username", "text", label="Username")
    form_group.add_field("password", "password", label="Password")
    
    # Set values
    form_group.fields["username"].setText("testuser")
    form_group.fields["password"].setText("password123")
    
    # Create submission handler
    submitted_data = {}
    def handle_submit(data):
        submitted_data.update(data)
    
    form_group.on_submit = handle_submit
    
    # Trigger submission
    form_group.submit()
    
    # Verify submission
    assert submitted_data["username"] == "testuser"
    assert submitted_data["password"] == "password123"

def test_form_reset(form_group):
    """Test form reset functionality."""
    # Setup form
    form_group.add_field("username", "text", label="Username")
    form_group.add_field("remember", "checkbox", label="Remember me")
    
    # Set values
    form_group.fields["username"].setText("testuser")
    form_group.fields["remember"].setChecked(True)
    
    # Reset form
    form_group.reset()
    
    # Verify reset
    assert form_group.fields["username"].text() == ""
    assert not form_group.fields["remember"].isChecked() 