import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spygate Layout Demo")
        self.setMinimumSize(400, 300)

        # Set dark theme styles
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                min-width: 200px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton#register-btn {
                background-color: #2d2d2d;
                border: 1px solid #0078d4;
            }
            QPushButton#register-btn:hover {
                background-color: #353535;
            }
        """
        )

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Welcome label
        welcome_label = QLabel("Welcome to Spygate")
        welcome_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create login form using grid layout
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setSpacing(10)

        # Add form widgets
        username_label = QLabel("Username:")
        username_input = QLineEdit()
        username_input.setPlaceholderText("Enter your username")
        password_label = QLabel("Password:")
        password_input = QLineEdit()
        password_input.setPlaceholderText("Enter your password")
        password_input.setEchoMode(QLineEdit.EchoMode.Password)

        form_layout.addWidget(username_label, 0, 0)
        form_layout.addWidget(username_input, 0, 1)
        form_layout.addWidget(password_label, 1, 0)
        form_layout.addWidget(password_input, 1, 1)

        # Create button row using horizontal layout
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setSpacing(10)

        login_button = QPushButton("Login")
        register_button = QPushButton("Register")
        register_button.setObjectName("register-btn")

        button_layout.addWidget(login_button)
        button_layout.addWidget(register_button)

        # Add widgets to main layout
        main_layout.addWidget(welcome_label)
        main_layout.addWidget(form_widget)
        main_layout.addWidget(button_widget)
        main_layout.addStretch()


def main():
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
