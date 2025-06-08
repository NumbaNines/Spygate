import os
from pathlib import Path

import pytest


def test_project_structure():
    """Test that the project structure is set up correctly."""
    # Define expected directories
    expected_dirs = [
        "spygate/gui",
        "spygate/gui/components",
        "spygate/gui/components/base",
        "spygate/gui/components/composite",
        "spygate/gui/layouts",
        "spygate/gui/themes",
        "spygate/utils",
        "spygate/video",
        "spygate/ml",
        "spygate/core",
    ]

    # Check each directory exists
    for dir_path in expected_dirs:
        assert os.path.isdir(dir_path), f"Directory {dir_path} not found"
        assert os.path.isfile(
            os.path.join(dir_path, "__init__.py")
        ), f"__init__.py missing in {dir_path}"


def test_pyqt6_installation():
    """Test that PyQt6 is properly installed."""
    try:
        import PyQt6
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication
    except ImportError as e:
        pytest.fail(f"PyQt6 import failed: {e}")


def test_project_metadata():
    """Test that project metadata is properly configured."""
    import spygate

    assert hasattr(spygate, "__version__"), "Version not defined in package"

    # Check pyproject.toml exists
    assert os.path.isfile("pyproject.toml"), "pyproject.toml not found"

    # Check README exists
    assert os.path.isfile("README.md"), "README.md not found"
