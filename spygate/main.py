"""
Spygate - Madden NFL 25 Game Analysis Tool
Main application entry point
"""

import logging
import sys
from pathlib import Path

# Fix import path issues
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from PyQt6.QtWidgets import QApplication

# Global variables to track what's available
FULL_FUNCTIONALITY = True
AVAILABLE_MODULES = {}


def check_imports():
    """Check which modules are available and create fallbacks."""
    global FULL_FUNCTIONALITY, AVAILABLE_MODULES

    try:
        from database.config import init_db

        AVAILABLE_MODULES["database"] = True
    except ImportError as e:
        print(f"Database module unavailable: {e}")
        AVAILABLE_MODULES["database"] = False

        def init_db():
            print("Database initialization skipped - module not available")

    # Check YOLOv8 availability
    try:
        import cv2
        import torch
        from ultralytics import YOLO

        AVAILABLE_MODULES["yolov8"] = True
        print("✅ YOLOv8 integration available!")
    except ImportError as e:
        print(f"YOLOv8 module unavailable: {e}")
        AVAILABLE_MODULES["yolov8"] = False

    try:
        from gui.components.main_window import MainWindow

        AVAILABLE_MODULES["main_window"] = True
    except ImportError as e:
        print(f"Main window module unavailable: {e}")
        AVAILABLE_MODULES["main_window"] = False
        from PyQt6.QtWidgets import (
            QFileDialog,
            QLabel,
            QMainWindow,
            QProgressBar,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        class MainWindow(QMainWindow):
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
                    from simple_main import test_functionality

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
                    model_path = "../hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
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
                    demo_path = Path("demo_frame.jpg")
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
                            self._log("   No HUD elements detected")
                    else:
                        self._log("✅ SpygateAI HUD Detection model loaded successfully!")

                    self.progress_bar.setValue(100)
                    self._log(f"   PyTorch: {torch.__version__}")
                    self._log(f"   CUDA Available: {torch.cuda.is_available()}")

                except Exception as e:
                    self._log(f"❌ SpygateAI HUD Detection test failed: {e}")
                finally:
                    self.progress_bar.setVisible(False)

            def _analyze_file(self):
                if not AVAILABLE_MODULES.get("yolov8"):
                    self._log("❌ SpygateAI HUD Detection not available for analysis")
                    return

                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Video or Image",
                    "",
                    "Media Files (*.mp4 *.avi *.mov *.jpg *.jpeg *.png);;All Files (*)",
                )

                if file_path:
                    try:
                        self._log(f"🔄 Analyzing: {Path(file_path).name}")
                        self.progress_bar.setVisible(True)
                        self.progress_bar.setValue(10)

                        from ultralytics import YOLO

                        # Use SpygateAI's custom 5-class HUD detection model
                        model_path = "../hud_region_training/runs/hud_regions_fresh_1749629437/weights/best.pt"
                        model = YOLO(model_path)
                        self.progress_bar.setValue(30)
                        
                        # Define the 5 custom classes
                        class_names = {
                            0: "hud",
                            1: "possession_triangle_area", 
                            2: "territory_triangle_area",
                            3: "preplay_indicator",
                            4: "play_call_screen"
                        }

                        results = model(file_path)
                        self.progress_bar.setValue(70)

                        if results and len(results) > 0:
                            boxes = results[0].boxes
                            if boxes is not None:
                                self._log(f"✅ HUD Analysis complete! Found {len(boxes)} HUD elements")
                                for i, box in enumerate(boxes):
                                    conf = box.conf[0].item()
                                    cls_id = int(box.cls[0].item())
                                    cls_name = class_names.get(cls_id, f"class_{cls_id}")
                                    coords = box.xyxy[0].tolist()
                                    self._log(f"   {cls_name}: {conf:.3f} at [{coords[0]:.0f},{coords[1]:.0f},{coords[2]:.0f},{coords[3]:.0f}]")
                            else:
                                self._log("✅ Analysis complete! No HUD elements detected")

                        self.progress_bar.setValue(100)

                    except Exception as e:
                        self._log(f"❌ Analysis failed: {e}")
                    finally:
                        self.progress_bar.setVisible(False)

            def _log(self, message):
                print(message)
                current_text = self.info_text.toPlainText()
                new_text = f"{current_text}\n{message}"
                self.info_text.setPlainText(new_text)
                # Scroll to bottom
                cursor = self.info_text.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.info_text.setTextCursor(cursor)

    try:
        from services.analysis_service import AnalysisService
        from services.video_service import VideoService

        AVAILABLE_MODULES["services"] = True
    except ImportError as e:
        print(f"Services modules unavailable: {e}")
        AVAILABLE_MODULES["services"] = False

        class AnalysisService:
            def __init__(self, video_service):
                pass

        class VideoService:
            pass

    try:
        from utils.error_tracking import init_error_tracking
        from utils.logging import setup_logging

        AVAILABLE_MODULES["utils"] = True
    except ImportError as e:
        print(f"Utils modules unavailable: {e}")
        AVAILABLE_MODULES["utils"] = False

        def init_error_tracking():
            print("Error tracking initialization skipped - module not available")

        def setup_logging(log_file):
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )

    # Check if we have full functionality
    FULL_FUNCTIONALITY = all(AVAILABLE_MODULES.values())

    # Return the modules
    return {
        "init_db": locals()["init_db"],
        "MainWindow": locals()["MainWindow"],
        "AnalysisService": locals()["AnalysisService"],
        "VideoService": locals()["VideoService"],
        "init_error_tracking": locals()["init_error_tracking"],
        "setup_logging": locals()["setup_logging"],
    }


def initialize_services(modules):
    """Initialize application services."""
    video_service = modules["VideoService"]()
    analysis_service = modules["AnalysisService"](video_service)
    return video_service, analysis_service


def main():
    """Application entry point."""
    # Check available modules
    modules = check_imports()

    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    modules["setup_logging"](log_dir / "spygate.log")
    logger = logging.getLogger(__name__)

    mode = "Full Functionality" if FULL_FUNCTIONALITY else "Enhanced Mode"
    logger.info(f"Starting Spygate application in {mode}")
    print(f"🚀 Starting Spygate in {mode}")
    if AVAILABLE_MODULES.get("yolov8"):
        print("🤖 YOLOv8 AI Analysis Ready!")

    try:
        # Initialize error tracking
        modules["init_error_tracking"]()
        logger.info("Error tracking initialized")

        # Initialize the database
        modules["init_db"]()
        logger.info("Database initialized successfully")

        # Initialize services
        video_service, analysis_service = initialize_services(modules)
        logger.info("Services initialized successfully")

        # Create Qt application
        app = QApplication(sys.argv)

        # Create and show main window
        window = modules["MainWindow"](
            video_service=video_service, analysis_service=analysis_service
        )
        window.show()
        logger.info(f"Main window displayed - {mode}")

        return app.exec()

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        print(f"❌ Startup error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
