import pytest
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope="session")
def app(qapp):
    """Create a Qt Application that persists for the entire test session."""
    return qapp

@pytest.fixture
def theme():
    """Provide a default theme configuration for testing."""
    return {
        # General
        "primary": "#007bff",
        "secondary": "#6c757d",
        "success": "#28a745",
        "danger": "#dc3545",
        "warning": "#ffc107",
        "info": "#17a2b8",
        
        # Text
        "text_primary": "#212529",
        "text_secondary": "#6c757d",
        
        # Background
        "bg_primary": "#ffffff",
        "bg_secondary": "#f8f9fa",
        
        # Components
        "card_bg": "#ffffff",
        "card_border": "#dee2e6",
        "dialog_bg": "#ffffff",
        "dialog_border": "#dee2e6",
        "form_bg": "#ffffff",
        "form_border": "#ced4da",
        "nav_bg": "#ffffff",
        "nav_border": "#dee2e6",
        
        # States
        "hover": "#e9ecef",
        "active": "#007bff",
        "disabled": "#6c757d",
        
        # Elevation shadows
        "elevation_1": "0 2px 4px rgba(0,0,0,0.1)",
        "elevation_2": "0 4px 8px rgba(0,0,0,0.1)",
        "elevation_3": "0 8px 16px rgba(0,0,0,0.1)",
        "elevation_4": "0 16px 32px rgba(0,0,0,0.1)",
        "elevation_5": "0 32px 64px rgba(0,0,0,0.1)"
    } 