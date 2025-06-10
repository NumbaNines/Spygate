#!/usr/bin/env python3
"""
Formation Editor - Visual tool for creating football formation presets
Allows dragging players to desired positions and exporting as formation presets
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass
class PlayerPosition:
    """Represents a player position with coordinates"""

    x: float
    y: float
    position: str


class ZoomableGraphicsView(QGraphicsView):
    """Custom QGraphicsView with zoom functionality"""

    def __init__(self):
        super().__init__()
        self.zoom_factor = 1.15
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.current_zoom = 1.0

        # Enable zoom and pan functionality
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # self.setRenderHint(QGraphicsView.RenderHint.Antialiasing)  # Skip render hint for now
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        # Check if Ctrl is pressed for zoom, otherwise normal scroll
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle = event.angleDelta().y()
            factor = self.zoom_factor if angle > 0 else 1 / self.zoom_factor

            # Calculate new zoom level
            new_zoom = self.current_zoom * factor

            # Clamp zoom level
            if self.min_zoom <= new_zoom <= self.max_zoom:
                self.scale(factor, factor)
                self.current_zoom = new_zoom
                # Notify parent window to update zoom label
                parent = self.parent()
                while parent and not hasattr(parent, "update_zoom_label"):
                    parent = parent.parent()
                if parent and hasattr(parent, "update_zoom_label"):
                    parent.update_zoom_label()
        else:
            # Normal scroll behavior
            super().wheelEvent(event)

    def zoom_in(self):
        """Zoom in"""
        if self.current_zoom < self.max_zoom:
            self.scale(self.zoom_factor, self.zoom_factor)
            self.current_zoom *= self.zoom_factor

    def zoom_out(self):
        """Zoom out"""
        if self.current_zoom > self.min_zoom:
            factor = 1 / self.zoom_factor
            self.scale(factor, factor)
            self.current_zoom *= factor

    def reset_zoom(self):
        """Reset zoom to 1:1"""
        factor = 1 / self.current_zoom
        self.scale(factor, factor)
        self.current_zoom = 1.0

    def fit_in_view_custom(self):
        """Fit the field in view"""
        self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Update current zoom based on transform
        self.current_zoom = self.transform().m11()


class DraggablePlayer(QGraphicsEllipseItem):
    """Draggable player icon"""

    def __init__(self, position: str, x: float, y: float, color: QColor, parent_widget):
        super().__init__(0, 0, 30, 30)
        self.position = position
        self.parent_widget = parent_widget
        self.setPos(x - 15, y - 15)  # Center the icon

        # Set appearance
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor("#ffffff"), 2))

        # Make draggable
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Add text label
        self.label = QGraphicsTextItem(position, self)
        self.label.setDefaultTextColor(QColor("#ffffff"))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        self.label.setFont(font)
        self.label.setPos(5, 8)  # Center text in circle

    def itemChange(self, change, value):
        """Called when item is moved - update coordinates display"""
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            # Update coordinates in real-time
            new_pos = value
            x = new_pos.x() + 15  # Add offset back to get center
            y = new_pos.y() + 15
            self.parent_widget.update_coordinates(self.position, x, y)
        return super().itemChange(change, value)


class FormationEditor(QMainWindow):
    """Main formation editor application"""

    def __init__(self):
        super().__init__()
        self.players: dict[str, DraggablePlayer] = {}
        self.current_formation_name = ""
        self.init_ui()
        self.create_field()
        self.add_players()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Formation Editor - SpygateAI")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel - field view with zoom controls
        field_panel = QWidget()
        field_layout = QVBoxLayout(field_panel)

        # Zoom control buttons
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton("🔍+ Zoom In")
        zoom_out_btn = QPushButton("🔍- Zoom Out")
        reset_zoom_btn = QPushButton("↺ Reset Zoom")
        fit_view_btn = QPushButton("⌂ Fit View")

        zoom_in_btn.setToolTip("Zoom in (or Ctrl + Mouse Wheel)")
        zoom_out_btn.setToolTip("Zoom out (or Ctrl + Mouse Wheel)")
        reset_zoom_btn.setToolTip("Reset to 1:1 zoom")
        fit_view_btn.setToolTip("Fit entire field in view")

        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(reset_zoom_btn)
        zoom_layout.addWidget(fit_view_btn)
        zoom_layout.addStretch()

        # Zoom level display
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setStyleSheet("font-weight: bold; color: #333;")
        zoom_layout.addWidget(self.zoom_label)

        field_layout.addLayout(zoom_layout)

        # Graphics view
        self.field_view = ZoomableGraphicsView()
        self.field_scene = QGraphicsScene()
        self.field_view.setScene(self.field_scene)
        self.field_view.setFixedSize(800, 550)  # Slightly smaller to make room for controls

        # Connect zoom buttons
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        fit_view_btn.clicked.connect(self.fit_view)

        field_layout.addWidget(self.field_view)
        layout.addWidget(field_panel)

        # Right panel - controls
        controls_panel = self.create_controls_panel()
        layout.addWidget(controls_panel)

    def create_controls_panel(self):
        """Create the controls panel"""
        panel = QWidget()
        panel.setFixedWidth(350)
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Formation Editor")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #ffffff; background-color: #2a2a2a; padding: 10px;"
        )
        layout.addWidget(title)

        # Formation name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Formation Name:"))
        self.formation_name_input = QLineEdit()
        self.formation_name_input.setPlaceholderText("e.g., Gun Bunch")
        name_layout.addWidget(self.formation_name_input)
        layout.addLayout(name_layout)

        # Load existing formation
        load_layout = QHBoxLayout()
        load_layout.addWidget(QLabel("Load Formation:"))
        self.formation_combo = QComboBox()
        self.load_existing_formations()
        load_layout.addWidget(self.formation_combo)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_formation)
        load_layout.addWidget(load_btn)
        layout.addLayout(load_layout)

        # Coordinates display
        coords_label = QLabel("Player Coordinates:")
        coords_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(coords_label)

        self.coordinates_display = QLabel()
        self.coordinates_display.setStyleSheet(
            "font-family: monospace; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;"
        )
        self.coordinates_display.setWordWrap(True)
        layout.addWidget(self.coordinates_display)

        # Buttons
        button_layout = QVBoxLayout()

        save_btn = QPushButton("Save Formation")
        save_btn.clicked.connect(self.save_formation)
        save_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        button_layout.addWidget(save_btn)

        export_btn = QPushButton("Export to Clipboard")
        export_btn.clicked.connect(self.export_to_clipboard)
        export_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 10px;"
        )
        button_layout.addWidget(export_btn)

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_to_default)
        reset_btn.setStyleSheet(
            "background-color: #ff9800; color: white; font-weight: bold; padding: 10px;"
        )
        button_layout.addWidget(reset_btn)

        layout.addLayout(button_layout)
        layout.addStretch()

        # Instructions
        instructions = QLabel(
            """
Instructions:
• Drag players to desired positions
• ZOOM: Ctrl + Mouse Wheel or zoom buttons
• Enter formation name above
• Click 'Save Formation' to save
• Click 'Export to Clipboard' to copy code
• Use exported code in main app

Tips:
• Line of scrimmage is at y=300 (25-yard line)
• Field numbered like real football: 10-20-30-40-50-40-30-20-10
• QB typically 5-7 yards behind LOS
• RB positioned behind QB
• Zoom in for precise positioning!
        """
        )
        instructions.setStyleSheet(
            "background-color: #e8e8e8; padding: 10px; border: 1px solid #ccc; font-size: 11px;"
        )
        layout.addWidget(instructions)

        return panel

    def create_field(self):
        """Create the football field graphics"""
        # Field background
        field = QGraphicsRectItem(0, 0, 600, 1000)
        field.setBrush(QBrush(QColor("#228B22")))  # Green
        field.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(field)

        # End zones
        end_zone_1 = QGraphicsRectItem(0, 0, 600, 100)
        end_zone_1.setBrush(QBrush(QColor("#1e7e1e")))  # Darker green
        end_zone_1.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(end_zone_1)

        end_zone_2 = QGraphicsRectItem(0, 900, 600, 100)
        end_zone_2.setBrush(QBrush(QColor("#1e7e1e")))
        end_zone_2.setPen(QPen(QColor("#ffffff"), 2))
        self.field_scene.addItem(end_zone_2)

        # Goal lines
        goal_line_1 = QGraphicsLineItem(0, 100, 600, 100)
        goal_line_1.setPen(QPen(QColor("#ffffff"), 3))
        self.field_scene.addItem(goal_line_1)

        goal_line_2 = QGraphicsLineItem(0, 900, 600, 900)
        goal_line_2.setPen(QPen(QColor("#ffffff"), 3))
        self.field_scene.addItem(goal_line_2)

        # Real football field layout: 0-10-20-30-40-50-40-30-20-10-0
        # Each yard = 8 pixels, total field = 100 yards = 800 pixels + 2 x 100 pixel end zones

        # Major yard lines (every 10 yards)
        for yard in range(10, 100, 10):
            y_pos = 100 + (yard * 8)  # Scale to field
            line = QGraphicsLineItem(0, y_pos, 600, y_pos)
            line.setPen(QPen(QColor("#ffffff"), 2))  # Thicker for major yard lines
            self.field_scene.addItem(line)

        # 50-yard line (midfield) - special thicker line
        midfield_line = QGraphicsLineItem(0, 500, 600, 500)  # y=500 is midfield
        midfield_line.setPen(QPen(QColor("#ffffff"), 3))
        self.field_scene.addItem(midfield_line)

        # 1-yard increment lines - Minor lines
        for yard in range(1, 100):
            if yard % 10 != 0:  # Skip the major 10-yard lines we already drew
                y_pos = 100 + (yard * 8)  # Scale to field
                line = QGraphicsLineItem(0, y_pos, 600, y_pos)
                line.setPen(QPen(QColor("#ffffff"), 0.5))  # Thinner for 1-yard increments
                self.field_scene.addItem(line)

        # 5-yard lines (slightly thicker than 1-yard)
        for yard in range(5, 100, 5):
            if yard % 10 != 0:  # Skip the major 10-yard lines
                y_pos = 100 + (yard * 8)  # Scale to field
                line = QGraphicsLineItem(0, y_pos, 600, y_pos)
                line.setPen(QPen(QColor("#ffffff"), 1))  # Medium thickness for 5-yard lines
                self.field_scene.addItem(line)

        # Line of scrimmage (highlight) - positioned at 25-yard line
        los_line = QGraphicsLineItem(0, 300, 600, 300)  # 25-yard line position
        los_line.setPen(QPen(QColor("#ff6b35"), 4))  # Orange highlight
        self.field_scene.addItem(los_line)

        # Add LOS label
        los_label = QGraphicsTextItem("Line of Scrimmage (25-yard line)")
        los_label.setDefaultTextColor(QColor("#ff6b35"))
        los_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        los_label.setPos(10, 270)
        self.field_scene.addItem(los_label)

        # Add yard number labels for real football field numbering
        # Real field: 0, 10, 20, 30, 40, 50, 40, 30, 20, 10, 0
        yard_numbers = [10, 20, 30, 40, 50, 40, 30, 20, 10]

        for i, display_yard in enumerate(yard_numbers):
            actual_yard = (i + 1) * 10  # Position on field (10, 20, 30, ..., 90)
            y_pos = 100 + (actual_yard * 8)  # Scale to field

            # Left side yard numbers
            yard_label_left = QGraphicsTextItem(str(display_yard))
            yard_label_left.setDefaultTextColor(QColor("#ffffff"))
            yard_label_left.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            yard_label_left.setPos(5, y_pos - 15)
            self.field_scene.addItem(yard_label_left)

            # Right side yard numbers
            yard_label_right = QGraphicsTextItem(str(display_yard))
            yard_label_right.setDefaultTextColor(QColor("#ffffff"))
            yard_label_right.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            yard_label_right.setPos(570, y_pos - 15)
            self.field_scene.addItem(yard_label_right)

        # Special label for 50-yard line
        midfield_label = QGraphicsTextItem("50")
        midfield_label.setDefaultTextColor(QColor("#ffff00"))  # Yellow for midfield
        midfield_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        midfield_label.setPos(290, 485)  # Center of field
        self.field_scene.addItem(midfield_label)

    def add_players(self):
        """Add draggable player icons to the field"""
        # Default Gun Bunch formation positions (on 25-yard line)
        default_positions = {
            "QB": (300, 250, QColor("#0066cc")),  # Blue for QB (7 yards behind LOS)
            "RB": (300, 230, QColor("#cc6600")),  # Orange for RB (behind QB)
            "WR1": (100, 300, QColor("#cc0066")),  # Pink for WRs (on LOS)
            "WR2": (200, 300, QColor("#cc0066")),
            "WR3": (410, 300, QColor("#cc0066")),
            "TE": (430, 300, QColor("#9900cc")),  # Purple for TE (on LOS)
            "LT": (250, 300, QColor("#666666")),  # Gray for O-line (on LOS)
            "LG": (275, 300, QColor("#666666")),
            "C": (300, 300, QColor("#666666")),
            "RG": (325, 300, QColor("#666666")),
            "RT": (350, 300, QColor("#666666")),
        }

        for position, (x, y, color) in default_positions.items():
            player = DraggablePlayer(position, x, y, color, self)
            self.players[position] = player
            self.field_scene.addItem(player)

        self.update_all_coordinates()

    def update_coordinates(self, position: str, x: float, y: float):
        """Update coordinates display for a specific player"""
        # This will be called as players are dragged
        self.update_all_coordinates()

    def update_all_coordinates(self):
        """Update the coordinates display with all player positions"""
        coords_text = ""
        for position in ["QB", "RB", "WR1", "WR2", "WR3", "TE", "LT", "LG", "C", "RG", "RT"]:
            if position in self.players:
                player = self.players[position]
                x = player.pos().x() + 15  # Add offset to get center
                y = player.pos().y() + 15
                coords_text += f'"{position}": ({int(x)}, {int(y)}),\n'

        self.coordinates_display.setText(coords_text)

    def get_current_formation(self) -> dict[str, tuple[int, int]]:
        """Get current formation as a dictionary"""
        formation = {}
        for position, player in self.players.items():
            x = int(player.pos().x() + 15)
            y = int(player.pos().y() + 15)
            formation[position] = (x, y)
        return formation

    def save_formation(self):
        """Save the current formation to a JSON file"""
        name = self.formation_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a formation name")
            return

        formation = self.get_current_formation()

        # Load existing formations
        try:
            with open("formations.json") as f:
                formations = json.load(f)
        except FileNotFoundError:
            formations = {}

        formations[name] = formation

        # Save back to file
        with open("formations.json", "w") as f:
            json.dump(formations, f, indent=2)

        QMessageBox.information(self, "Success", f"Formation '{name}' saved successfully!")
        self.load_existing_formations()

    def export_to_clipboard(self):
        """Export current formation to clipboard as code"""
        name = self.formation_name_input.text().strip() or "Custom Formation"
        formation = self.get_current_formation()

        code = f'"{name}": {{\n'
        for position, (x, y) in formation.items():
            code += f'    "{position}": ({x}, {y}),\n'
        code += "},"

        clipboard = QApplication.clipboard()
        clipboard.setText(code)

        QMessageBox.information(
            self,
            "Exported",
            "Formation code copied to clipboard!\nPaste it into your main app's formation dictionary.",
        )

    def load_existing_formations(self):
        """Load existing formations into the combo box"""
        self.formation_combo.clear()
        try:
            with open("formations.json") as f:
                formations = json.load(f)
                self.formation_combo.addItems(formations.keys())
        except FileNotFoundError:
            pass

    def load_formation(self):
        """Load selected formation"""
        formation_name = self.formation_combo.currentText()
        if not formation_name:
            return

        try:
            with open("formations.json") as f:
                formations = json.load(f)

            if formation_name in formations:
                formation = formations[formation_name]

                # Move players to loaded positions
                for position, (x, y) in formation.items():
                    if position in self.players:
                        self.players[position].setPos(x - 15, y - 15)

                self.formation_name_input.setText(formation_name)
                self.update_all_coordinates()

        except FileNotFoundError:
            QMessageBox.warning(self, "Error", "No formations file found")

    def reset_to_default(self):
        """Reset to default Gun Bunch formation"""
        default_positions = {
            "QB": (300, 730),
            "RB": (300, 710),
            "WR1": (100, 780),
            "WR2": (200, 780),
            "WR3": (410, 780),
            "TE": (430, 780),
            "LT": (250, 780),
            "LG": (275, 780),
            "C": (300, 780),
            "RG": (325, 780),
            "RT": (350, 780),
        }

        for position, (x, y) in default_positions.items():
            if position in self.players:
                self.players[position].setPos(x - 15, y - 15)

        self.formation_name_input.setText("Shotgun Gun Bunch")
        self.update_all_coordinates()

    def zoom_in(self):
        """Zoom in and update zoom label"""
        self.field_view.zoom_in()
        self.update_zoom_label()

    def zoom_out(self):
        """Zoom out and update zoom label"""
        self.field_view.zoom_out()
        self.update_zoom_label()

    def reset_zoom(self):
        """Reset zoom to 1:1 and update zoom label"""
        self.field_view.reset_zoom()
        self.update_zoom_label()

    def fit_view(self):
        """Fit the field in view and update zoom label"""
        self.field_view.fit_in_view_custom()
        self.update_zoom_label()

    def update_zoom_label(self):
        """Update the zoom level display"""
        zoom_percent = int(self.field_view.current_zoom * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")


def main():
    app = QApplication(sys.argv)
    editor = FormationEditor()
    editor.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
