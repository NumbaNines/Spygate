from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TutorialStep:
    def __init__(self, title, description, target_widget=None, highlight_rect=None):
        self.title = title
        self.description = description
        self.target_widget = target_widget
        self.highlight_rect = highlight_rect


class TutorialOverlay(QWidget):
    """Semi-transparent overlay widget that highlights tutorial elements"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.highlight_rect = None

    def set_highlight(self, rect):
        self.highlight_rect = rect
        self.update()

    def paintEvent(self, event):
        if not self.highlight_rect:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw semi-transparent dark overlay
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.PenStyle.NoPen)

        # Create path for the entire widget
        path = QPainterPath()
        path.addRect(self.rect())

        # Create path for the highlighted area
        highlight = QPainterPath()
        highlight.addRoundedRect(self.highlight_rect, 8, 8)

        # Subtract highlight from overlay
        path = path.subtracted(highlight)
        painter.drawPath(path)

        # Draw highlight border
        painter.setPen(QPen(QColor("#3B82F6"), 2, Qt.PenStyle.SolidLine))
        painter.drawRoundedRect(self.highlight_rect, 8, 8)


class TutorialPopup(QFrame):
    """Popup widget displaying tutorial step information"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet(
            """
            TutorialPopup {
                background-color: #2D3748;
                border: 1px solid #4A5568;
                border-radius: 8px;
                padding: 16px;
            }
            QLabel {
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #6B7280;
            }
        """
        )

        self.layout = QVBoxLayout(self)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.layout.addWidget(self.description_label)

        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.skip_button = QPushButton("Skip Tutorial")

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.skip_button)

        self.layout.addLayout(button_layout)


class TutorialSystem(QObject):
    """Main tutorial system managing the tutorial flow"""

    tutorial_completed = pyqtSignal(str)  # Emitted when a tutorial is completed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.current_step = 0
        self.steps = []
        self.overlay = None
        self.popup = None
        self.current_tutorial = None

    def start_tutorial(self, tutorial_id, steps):
        """Start a tutorial with the given steps"""
        self.current_tutorial = tutorial_id
        self.steps = steps
        self.current_step = 0

        if not self.overlay:
            self.overlay = TutorialOverlay(self.main_window)
            self.overlay.resize(self.main_window.size())

        if not self.popup:
            self.popup = TutorialPopup(self.main_window)
            self.popup.prev_button.clicked.connect(self.previous_step)
            self.popup.next_button.clicked.connect(self.next_step)
            self.popup.skip_button.clicked.connect(self.skip_tutorial)

        self.show_current_step()
        self.overlay.show()
        self.popup.show()

    def show_current_step(self):
        """Display the current tutorial step"""
        if not self.steps or self.current_step >= len(self.steps):
            self.complete_tutorial()
            return

        step = self.steps[self.current_step]
        self.popup.title_label.setText(step.title)
        self.popup.description_label.setText(step.description)

        # Update button states
        self.popup.prev_button.setEnabled(self.current_step > 0)
        self.popup.next_button.setText(
            "Finish" if self.current_step == len(self.steps) - 1 else "Next"
        )

        # Position popup and highlight target
        if step.target_widget:
            target_rect = step.target_widget.geometry()
            global_pos = step.target_widget.mapToGlobal(QPoint(0, 0))
            target_rect.moveTopLeft(self.main_window.mapFromGlobal(global_pos))

            self.overlay.set_highlight(target_rect)

            # Position popup near the target widget
            popup_pos = self.calculate_popup_position(target_rect)
            self.popup.move(popup_pos)
        else:
            self.overlay.set_highlight(None)
            # Center popup if no target
            self.center_popup()

    def calculate_popup_position(self, target_rect):
        """Calculate the best position for the popup relative to the target"""
        popup_size = self.popup.sizeHint()
        window_rect = self.main_window.rect()

        # Try to position below the target
        pos = QPoint(
            target_rect.center().x() - popup_size.width() // 2,
            target_rect.bottom() + 10,
        )

        # Adjust if popup would go outside window bounds
        pos.setX(max(10, min(pos.x(), window_rect.width() - popup_size.width() - 10)))
        if pos.y() + popup_size.height() > window_rect.height() - 10:
            # Position above target if not enough space below
            pos.setY(target_rect.top() - popup_size.height() - 10)

        return pos

    def center_popup(self):
        """Center the popup in the main window"""
        popup_size = self.popup.sizeHint()
        window_rect = self.main_window.rect()

        pos = QPoint(
            (window_rect.width() - popup_size.width()) // 2,
            (window_rect.height() - popup_size.height()) // 2,
        )
        self.popup.move(pos)

    def next_step(self):
        """Advance to the next tutorial step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.show_current_step()
        else:
            self.complete_tutorial()

    def previous_step(self):
        """Go back to the previous tutorial step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_current_step()

    def skip_tutorial(self):
        """Skip the current tutorial"""
        self.complete_tutorial()

    def complete_tutorial(self):
        """Complete the current tutorial"""
        if self.overlay:
            self.overlay.hide()
        if self.popup:
            self.popup.hide()

        if self.current_tutorial:
            self.tutorial_completed.emit(self.current_tutorial)
            self.current_tutorial = None

    def cleanup(self):
        """Clean up tutorial system resources"""
        if self.overlay:
            self.overlay.deleteLater()
            self.overlay = None
        if self.popup:
            self.popup.deleteLater()
            self.popup = None
