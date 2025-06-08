import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow


def test_hello_world(qtbot):
    """Test basic PyQt6 functionality with a hello world window."""
    # Create the application and window
    window = QMainWindow()
    window.setWindowTitle("Hello World Test")

    # Create and set up the label
    label = QLabel("Hello, World!")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.setCentralWidget(label)

    # Show the window
    window.show()

    # Use qtbot to verify the window
    assert window.windowTitle() == "Hello World Test"
    assert label.text() == "Hello, World!"
    assert label.alignment() == Qt.AlignmentFlag.AlignCenter
