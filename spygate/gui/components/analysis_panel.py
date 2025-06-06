"""
Spygate - Analysis Panel Component
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QFormLayout,
)
from PyQt6.QtCore import Qt


class AnalysisPanel(QWidget):
    """Panel containing analysis tools and results."""

    def __init__(self):
        """Initialize the analysis panel."""
        super().__init__()
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Create tabs for different analysis tools
        self.tabs = QTabWidget()
        
        # YOLO Detection Settings tab
        self.detection_tab = QWidget()
        detection_layout = QFormLayout(self.detection_tab)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLO v8n", "YOLO v8s", "YOLO v8m", "YOLO v8l", "YOLO v8x"])
        detection_layout.addRow("Model:", self.model_combo)
        
        # Confidence threshold
        self.conf_threshold = QSpinBox()
        self.conf_threshold.setRange(1, 100)
        self.conf_threshold.setValue(50)
        self.conf_threshold.setSuffix("%")
        detection_layout.addRow("Confidence:", self.conf_threshold)
        
        # Start analysis button
        self.analyze_btn = QPushButton("Start Analysis")
        detection_layout.addRow(self.analyze_btn)
        
        # Results tab
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        results_layout.addWidget(QLabel("Analysis results will appear here"))
        
        # Add tabs
        self.tabs.addTab(self.detection_tab, "Detection Settings")
        self.tabs.addTab(self.results_tab, "Results")
        
        layout.addWidget(self.tabs) 