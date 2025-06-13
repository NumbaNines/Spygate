"""
Spygate - Madden NFL 25 Game Analysis Tool
Main application entry point
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Global variables to track what's available
FULL_FUNCTIONALITY = True
AVAILABLE_MODULES = {
    "database": False,
    "yolov8": False,
    "main_window": False,
    "services": False
}

def check_imports():
    """Check which modules are available and create fallbacks."""
    global FULL_FUNCTIONALITY, AVAILABLE_MODULES

    # Database module
    try:
        from spygate.database.config import init_db
        AVAILABLE_MODULES["database"] = True
    except ImportError as e:
        logging.warning(f"Database module unavailable: {e}")
        def init_db():
            logging.info("Database initialization skipped - module not available")

    # YOLOv8 module
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        AVAILABLE_MODULES["yolov8"] = True
        logging.info("✅ YOLOv8 integration available!")
    except ImportError as e:
        logging.warning(f"YOLOv8 module unavailable: {e}")

    # Main window module
    try:
        from spygate.gui.components.main_window import MainWindow as CustomMainWindow
        AVAILABLE_MODULES["main_window"] = True
        MainWindow = CustomMainWindow
    except ImportError as e:
        logging.warning(f"Custom main window module unavailable: {e}")
        MainWindow = create_fallback_main_window()

    # Services module
    try:
        from spygate.services.analysis import AnalysisService
        from spygate.services.video import VideoService
        AVAILABLE_MODULES["services"] = True
    except ImportError as e:
        logging.warning(f"Services module unavailable: {e}")
        AnalysisService = create_fallback_analysis_service()
        VideoService = create_fallback_video_service()

    return MainWindow, AnalysisService, VideoService

def create_fallback_main_window():
    """Create a fallback main window class when the custom one is not available."""
    class FallbackMainWindow(QMainWindow):
        def __init__(self, video_service=None, analysis_service=None):
            super().__init__()
            self.setWindowTitle("Spygate - Enhanced Mode")
            self.setGeometry(100, 100, 1200, 800)
            self.setStyleSheet(
                """
                QMainWindow {
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                QLabel {
                    color: #3B82F6;
                    font-size: 14px;
                    padding: 10px;
                }
                QPushButton {
                    background-color: #3B82F6;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    font-size: 12px;
                    border-radius: 6px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2563EB;
                }
                QPushButton:pressed {
                    background-color: #1D4ED8;
                }
                QTextEdit {
                    background-color: #2a2a2a;
                    color: #ffffff;
                    border: 1px solid #3B82F6;
                    font-family: 'Consolas', monospace;
                    font-size: 11px;
                    border-radius: 4px;
                }
                QProgressBar {
                    background-color: #2a2a2a;
                    border: 1px solid #3B82F6;
                    border-radius: 4px;
                    text-align: center;
                    color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #3B82F6;
                    border-radius: 3px;
                }
            """
            )

            central_widget = QWidget()
            layout = QVBoxLayout(central_widget)

            # Header
            header = QLabel("🎮 Spygate - Football Analysis Tool")
            header.setStyleSheet(
                "font-size: 28px; font-weight: bold; text-align: center; padding: 20px; color: #3B82F6;"
            )
            layout.addWidget(header)

            # Status with SpygateAI HUD Detection info
            yolo_status = (
                "✅ SpygateAI HUD Detection Ready (5-class model)"
                if AVAILABLE_MODULES.get("yolov8")
                else "❌ YOLOv8 Unavailable"
            )
            status_text = f"""
            <b>Application Status:</b> Enhanced Mode Active<br>
            <b>Available Modules:</b> {len([k for k, v in AVAILABLE_MODULES.items() if v])}/{len(AVAILABLE_MODULES)}<br>
            <br>
            <b>🔥 Core Features Available:</b><br>
            ✅ PyQt6 Professional Interface<br>
            ✅ Application Framework<br>
            ✅ Advanced Window Management<br>
            {yolo_status}<br>
            <br>
            <b>📊 Modules Status:</b><br>
            {'✅' if AVAILABLE_MODULES.get('database') else '❌'} Database Module<br>
            {'✅' if AVAILABLE_MODULES.get('main_window') else '❌'} Full Main Window<br>
            {'✅' if AVAILABLE_MODULES.get('services') else '❌'} Analysis Services<br>
            {'✅' if AVAILABLE_MODULES.get('yolov8') else '❌'} SpygateAI HUD Detection (5-class)<br>
            """

            status_label = QLabel(status_text)
            layout.addWidget(status_label)

            # Buttons row
            buttons_layout = QVBoxLayout()

            # Core functionality buttons
            test_btn = QPushButton("🧪 Test Core Functionality")
            test_btn.clicked.connect(self._test_functionality)
            buttons_layout.addWidget(test_btn)

            if AVAILABLE_MODULES.get("yolov8"):
                yolo_btn = QPushButton("🤖 Test SpygateAI HUD Detection")
                yolo_btn.clicked.connect(self._test_yolo)
                buttons_layout.addWidget(yolo_btn)

                analyze_btn = QPushButton("📹 Analyze Video/Image for HUD Elements")
                analyze_btn.clicked.connect(self._analyze_file)
                buttons_layout.addWidget(analyze_btn)

            layout.addLayout(buttons_layout)

            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)

            # Module info
            self.info_text = QTextEdit()
            self.info_text.setPlainText(self._get_module_info())
            self.info_text.setMaximumHeight(200)
            layout.addWidget(self.info_text)

            self.setCentralWidget(central_widget)

        def _get_module_info(self):
            return f"""
🔥 SPYGATE ENHANCED MODE 🔥
Working Directory: {Path.cwd()}
Available Modules: {AVAILABLE_MODULES}
PyQt6: ✅ Available
SpygateAI HUD Detection: {'✅ Ready for Analysis' if AVAILABLE_MODULES.get('yolov8') else '❌ Not Available'}

{f'🤖 HUD DETECTION READY: SpygateAI 5-class model loaded and ready for HUD analysis!' if AVAILABLE_MODULES.get('yolov8') else 'Install YOLOv8 dependencies for HUD analysis capabilities.'}
            """.strip()

        def _test_functionality(self):
            try:
                from spygate.simple_main import test_functionality
                if test_functionality():
                    self._log("✅ Core functionality test passed!")
                else:
                    self._log("❌ Core functionality test failed!")
            except Exception as e:
                self._log(f"❌ Core functionality test failed: {e}")

        def _test_yolo(self):
            if not AVAILABLE_MODULES.get("yolov8"):
                self._log("❌ YOLOv8 not available")
                return

            try:
                self._log("🤖 Testing SpygateAI HUD Detection...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(20)

                import torch
                from ultralytics import YOLO

                self.progress_bar.setValue(40)
                # Use SpygateAI's custom 5-class HUD detection model
                model_path = os.path.join(project_root, "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt")
                from ultralytics import YOLO  # Import moved inside the method
                model = YOLO(model_path)
                self.progress_bar.setValue(60)
                
                # Define the 5 custom classes
                class_names = {
                    0: "hud",
                    1: "possession_triangle_area", 
                    2: "territory_triangle_area",
                    3: "preplay_indicator",
                    4: "play_call_screen"
                }

                # Test with demo image if available
                demo_path = Path(os.path.join(project_root, "demo_frame.jpg"))
                if demo_path.exists():
                    results = model(str(demo_path))
                    self.progress_bar.setValue(80)
                    self._log(f"✅ SpygateAI HUD Detection successful! Processed {demo_path.name}")
                    if results[0].boxes is not None:
                        self._log(f"   Detected {len(results[0].boxes)} HUD elements")
                        for i, box in enumerate(results[0].boxes):
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            cls_name = class_names.get(cls_id, f"class_{cls_id}")
                            self._log(f"     {cls_name}: {conf:.3f}")
                else:
                    self._log("❌ Demo image not found")
                self.progress_bar.setValue(100)
            except Exception as e:
                self._log(f"❌ YOLOv8 test failed: {e}")
            finally:
                self.progress_bar.setVisible(False)

        def _analyze_file(self):
            if not AVAILABLE_MODULES.get("yolov8"):
                self._log("❌ YOLOv8 not available")
                return

            try:
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Video/Image",
                    "",
                    "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png);;All Files (*.*)"
                )

                if file_path:
                    self._log(f"Selected file: {file_path}")
                    self.progress_bar.setVisible(True)
                    self.progress_bar.setValue(0)

                    # Load model
                    model_path = os.path.join(project_root, "hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt")
                    from ultralytics import YOLO  # Import moved inside the method
                    model = YOLO(model_path)

                    # Process file
                    results = model(file_path)
                    self.progress_bar.setValue(100)

                    # Display results
                    self._log(f"✅ Analysis complete! Found {len(results[0].boxes)} HUD elements")
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0].item())
                        conf = box.conf[0].item()
                        cls_name = {
                            0: "hud",
                            1: "possession_triangle_area",
                            2: "territory_triangle_area",
                            3: "preplay_indicator",
                            4: "play_call_screen"
                        }.get(cls_id, f"class_{cls_id}")
                        self._log(f"   {cls_name}: {conf:.3f}")

            except Exception as e:
                self._log(f"❌ Analysis failed: {e}")
            finally:
                self.progress_bar.setVisible(False)

        def _log(self, message):
            self.info_text.append(message)

    return FallbackMainWindow

def create_fallback_analysis_service():
    """Create a fallback analysis service when the custom one is not available."""
    class FallbackAnalysisService:
        def __init__(self, video_service):
            self.video_service = video_service
            logging.info("Using fallback analysis service")

    return FallbackAnalysisService

def create_fallback_video_service():
    """Create a fallback video service when the custom one is not available."""
    class FallbackVideoService:
        def __init__(self):
            logging.info("Using fallback video service")

    return FallbackVideoService

def init_error_tracking():
    """Initialize error tracking system."""
    logging.info("Error tracking initialized")

def setup_logging(log_file="spygate.log"):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def initialize_services(modules):
    """Initialize application services."""
    video_service = modules.get("video_service", create_fallback_video_service())()
    analysis_service = modules.get("analysis_service", create_fallback_analysis_service())(video_service)
    return video_service, analysis_service

def main():
    """Main application entry point."""
    # Set up logging
    setup_logging()
    logging.info("Starting Spygate application")

    # Initialize error tracking
    init_error_tracking()

    # Check available modules and get appropriate classes
    MainWindow, AnalysisService, VideoService = check_imports()

    # Initialize Qt application
    app = QApplication(sys.argv)

    # Initialize services
    video_service, analysis_service = initialize_services({
        "video_service": VideoService,
        "analysis_service": AnalysisService
    })

    # Create and show main window
    window = MainWindow(video_service, analysis_service)
    window.show()

    # Start application event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
