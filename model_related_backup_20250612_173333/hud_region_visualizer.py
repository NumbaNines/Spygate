import cv2
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                           QComboBox, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class HUDRegionVisualizer(QMainWindow):
    def __init__(self, model_path="hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"):
        super().__init__()
        self.setWindowTitle("YOLOv8 HUD Region Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Load YOLO model
        print(f"🚀 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class definitions from training
        self.classes = [
            "hud",                      # Main HUD bar
            "possession_triangle_area",  # Left triangle area (between team abbrev/scores, shows possession)
            "territory_triangle_area",   # Right triangle area (next to field pos, ▲=opp ▼=own territory)
            "preplay_indicator",        # Pre-play state indicator
            "play_call_screen"          # Play call screen overlay
        ]
        
        # Colors for visualization
        self.colors = {
            0: (255, 255, 0),    # Cyan - Main HUD
            1: (0, 255, 0),      # Green - Possession triangle area
            2: (0, 0, 255),      # Red - Territory triangle area  
            3: (255, 0, 255),    # Magenta - Pre-play
            4: (0, 165, 255)     # Orange - Play call
        }

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Controls
        controls_layout = QHBoxLayout()
        
        # Confidence threshold slider
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(30)  # Default 0.3
        self.conf_slider.valueChanged.connect(self.update_display)
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(QLabel("0.3"))
        controls_layout.addLayout(conf_layout)

        # Region toggles
        self.region_toggles = {}
        for i, region in enumerate(self.classes):
            cb = QCheckBox(region)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_display)
            self.region_toggles[i] = cb
            controls_layout.addWidget(cb)

        layout.addLayout(controls_layout)

        # Image display
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Create a test image
        self.test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.update_display()

    def update_display(self):
        """Update display with YOLO detections."""
        try:
            # Run YOLO detection
            conf_threshold = self.conf_slider.value() / 100
            results = self.model(self.test_image, conf=conf_threshold, verbose=False)
            
            # Create visualization
            vis_frame = self.test_image.copy()
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if class_id < len(self.classes) and self.region_toggles[class_id].isChecked():
                            class_name = self.classes[class_id]
                            x1, y1, x2, y2 = map(int, box)
                            color = self.colors.get(class_id, (255, 255, 255))
                            
                            # Draw bounding box
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"{class_name}: {conf:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(vis_frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
                            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Convert to Qt format
            height, width = vis_frame.shape[:2]
            bytes_per_line = 3 * width
            qt_image = QImage(vis_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error updating display: {e}")

    def load_video_frame(self, frame):
        """Load a video frame for visualization."""
        if frame is not None:
            self.test_image = frame.copy()
            self.update_display()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = HUDRegionVisualizer()
    window.show()
    sys.exit(app.exec()) 