#!/usr/bin/env python3
"""
SpygateAI - Main Application Entry Point
=======================================

This is the consolidated main entry point for SpygateAI that handles:
- Proper import path configuration
- System initialization
- Error handling and logging setup
- Resource management
- Application launch with fallback options

This file should be used as the primary way to launch SpygateAI.
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Configure Python path first, before any other imports
def setup_python_path():
    """Set up Python path to handle all the different project structures."""
    current_dir = Path(__file__).parent.resolve()
    
    # Add all possible source directories to Python path
    possible_paths = [
        current_dir,
        current_dir / "src",
        current_dir / "src" / "spygate",
        current_dir / "spygate_django",
        current_dir / "spygate_django" / "spygate",
    ]
    
    for path in possible_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    
    return current_dir

# Set up paths before any imports
PROJECT_ROOT = setup_python_path()

# Now we can safely import our modules
try:
    from src.spygate.utils.logging_config import setup_logging, get_logger
    from src.spygate.utils.error_handling import get_error_handler, handle_errors, error_boundary
    from src.spygate.utils.resource_manager import get_resource_manager
    from src.spygate.utils.validation import get_security_validator
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SpygateAI utilities not available: {e}")
    print("Running in basic mode...")
    UTILS_AVAILABLE = False

# Try to import GUI frameworks
GUI_AVAILABLE = False
try:
    from PyQt6.QtWidgets import QApplication
    GUI_AVAILABLE = True
    GUI_FRAMEWORK = "PyQt6"
except ImportError:
    try:
        from PyQt5.QtWidgets import QApplication
        GUI_AVAILABLE = True
        GUI_FRAMEWORK = "PyQt5"
    except ImportError:
        GUI_FRAMEWORK = None

# Try to import main application modules
APP_MODULES = {}

# Try production app
try:
    if Path(PROJECT_ROOT / "spygate_production_app.py").exists():
        sys.path.insert(0, str(PROJECT_ROOT))
        from spygate_production_app import SpygateProductionApp
        APP_MODULES["production"] = SpygateProductionApp
except ImportError as e:
    pass

# Try desktop app
try:
    if Path(PROJECT_ROOT / "spygate_desktop_app_faceit_style.py").exists():
        from spygate_desktop_app_faceit_style import SpygateDesktop
        APP_MODULES["desktop"] = SpygateDesktop
except ImportError as e:
    pass

# Try main from src
try:
    from src.spygate.main import main as src_main
    APP_MODULES["src_main"] = src_main
except ImportError as e:
    pass

# Try spygate engine
try:
    from src.spygate.core.spygate_engine import SpygateAI
    APP_MODULES["engine"] = SpygateAI
except ImportError as e:
    pass


class SpygateMainApplication:
    """Main application coordinator with fallback options."""
    
    def __init__(self):
        self.logger = None
        self.error_handler = None
        self.resource_manager = None
        self.app_config = {}
        
        # Initialize utilities if available
        if UTILS_AVAILABLE:
            self.logger = setup_logging("INFO", "logs")
            self.error_handler = get_error_handler()
            self.resource_manager = get_resource_manager()
            self.logger.info("SpygateAI utilities initialized successfully")
        else:
            # Fallback logging
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('spygate_basic.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            self.logger = logging.getLogger("spygate")
            self.logger.info("SpygateAI running in basic mode (utilities not available)")
    
    def detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect what the system can run."""
        capabilities = {
            "gui_available": GUI_AVAILABLE,
            "gui_framework": GUI_FRAMEWORK,
            "utilities_available": UTILS_AVAILABLE,
            "available_apps": list(APP_MODULES.keys()),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_root": str(PROJECT_ROOT)
        }
        
        # Check for specific dependencies
        capabilities["dependencies"] = {}
        
        for module_name in ["torch", "cv2", "numpy", "ultralytics", "sqlite3", "bcrypt"]:
            try:
                __import__(module_name)
                capabilities["dependencies"][module_name] = True
            except ImportError:
                capabilities["dependencies"][module_name] = False
        
        # Check GPU availability
        try:
            import torch
            capabilities["gpu_available"] = torch.cuda.is_available()
            if capabilities["gpu_available"]:
                capabilities["gpu_count"] = torch.cuda.device_count()
                capabilities["gpu_name"] = torch.cuda.get_device_name(0)
        except:
            capabilities["gpu_available"] = False
        
        return capabilities
    
    def print_startup_info(self, capabilities: Dict[str, Any]):
        """Print startup information."""
        print("🏈 SpygateAI - Football Analysis System")
        print("=" * 50)
        print(f"Project Root: {capabilities['project_root']}")
        print(f"Python Version: {capabilities['python_version']}")
        print(f"GUI Available: {capabilities['gui_available']} ({capabilities['gui_framework']})")
        print(f"Utilities Available: {capabilities['utilities_available']}")
        print(f"GPU Available: {capabilities.get('gpu_available', False)}")
        
        if capabilities.get('gpu_available'):
            print(f"GPU: {capabilities.get('gpu_name', 'Unknown')} (Count: {capabilities.get('gpu_count', 0)})")
        
        print(f"Available Apps: {', '.join(capabilities['available_apps']) if capabilities['available_apps'] else 'None'}")
        
        print("\nDependency Status:")
        for dep, available in capabilities["dependencies"].items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
        
        print("=" * 50)
    
    def choose_best_app(self, capabilities: Dict[str, Any]) -> Optional[str]:
        """Choose the best available application to run."""
        available_apps = capabilities["available_apps"]
        
        if not available_apps:
            return None
        
        # Priority order (best to fallback) - prioritize desktop (faceit style) as requested
        priority_order = ["desktop", "production", "src_main", "engine"]
        
        for app_name in priority_order:
            if app_name in available_apps:
                # Check if this app can actually run
                if app_name in ["production", "desktop"] and not capabilities["gui_available"]:
                    continue
                return app_name
        
        # Return any available app as last resort
        return available_apps[0] if available_apps else None
    
    @handle_errors(reraise=False) if UTILS_AVAILABLE else lambda f: f
    def run_gui_app(self, app_class) -> int:
        """Run a GUI application."""
        if not GUI_AVAILABLE:
            self.logger.error("GUI not available, cannot run GUI application")
            return 1
        
        try:
            # Create QApplication
            app = QApplication(sys.argv)
            app.setStyle("Fusion")
            
            # Create main window
            if app_class == "production":
                window = APP_MODULES["production"]()
            elif app_class == "desktop":
                window = APP_MODULES["desktop"]()
            else:
                self.logger.error(f"Unknown GUI app class: {app_class}")
                return 1
            
            window.show()
            
            self.logger.info(f"Starting {app_class} GUI application")
            
            # Run event loop
            return app.exec()
            
        except Exception as e:
            self.logger.error(f"Failed to run GUI application: {e}")
            if UTILS_AVAILABLE:
                traceback.print_exc()
            return 1
    
    @handle_errors(reraise=False) if UTILS_AVAILABLE else lambda f: f
    def run_console_app(self, app_name: str) -> int:
        """Run a console application."""
        try:
            if app_name == "src_main":
                self.logger.info("Running src main application")
                return APP_MODULES["src_main"]()
            elif app_name == "engine":
                self.logger.info("Running SpygateAI engine demo")
                engine = APP_MODULES["engine"](str(PROJECT_ROOT))
                
                # Run a simple demo
                print("\n🤖 SpygateAI Engine Demo")
                print("-" * 30)
                
                # Show system status
                status = engine.get_system_status()
                print(f"Engine Status: {status['status']}")
                print(f"Version: {status['engine_version']}")
                print(f"Systems: {status['systems_count']}")
                
                # Show hardware optimization
                hardware_info = engine.optimize_for_hardware()
                print(f"Hardware Tier: {hardware_info['performance_tier']}")
                print(f"GPU Memory: {hardware_info['gpu_memory_gb']:.1f}GB")
                
                print("\n✅ Engine demo completed successfully")
                return 0
            else:
                self.logger.error(f"Unknown console app: {app_name}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Failed to run console application: {e}")
            if UTILS_AVAILABLE:
                traceback.print_exc()
            return 1
    
    def run_fallback_mode(self) -> int:
        """Run in fallback mode when no apps are available."""
        print("\n⚠️ SpygateAI Fallback Mode")
        print("=" * 40)
        print("No complete applications are available to run.")
        print("This usually means some dependencies are missing.")
        print("\nTo fix this, try:")
        print("1. pip install -r requirements.txt")
        print("2. pip install PyQt6")
        print("3. pip install torch torchvision ultralytics")
        print("4. Ensure all SpygateAI files are present")
        
        # Try to show some basic functionality
        try:
            if UTILS_AVAILABLE:
                print("\n📊 System Information:")
                resource_manager = get_resource_manager()
                summary = resource_manager.get_resource_summary()
                
                current = summary.get("current", {})
                print(f"  CPU: {current.get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {current.get('memory_mb', 0):.1f}MB ({current.get('memory_percent', 0):.1f}%)")
                print(f"  Active Threads: {current.get('active_threads', 0)}")
                
                print("\n✅ SpygateAI utilities are working correctly")
            else:
                print("\n❌ SpygateAI utilities are not available")
        except Exception as e:
            print(f"\n❌ Error in fallback mode: {e}")
        
        print("\nExiting...")
        return 1
    
    def main(self) -> int:
        """Main application entry point."""
        try:
            # Detect system capabilities
            capabilities = self.detect_system_capabilities()
            
            # Print startup information
            self.print_startup_info(capabilities)
            
            # Choose best application
            chosen_app = self.choose_best_app(capabilities)
            
            if not chosen_app:
                return self.run_fallback_mode()
            
            self.logger.info(f"Launching application: {chosen_app}")
            
            # Run the chosen application
            if chosen_app in ["production", "desktop"]:
                return self.run_gui_app(chosen_app)
            else:
                return self.run_console_app(chosen_app)
                
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return 0
        except Exception as e:
            if self.logger:
                self.logger.critical(f"Critical error in main application: {e}")
            else:
                print(f"Critical error: {e}")
                traceback.print_exc()
            return 1
        finally:
            # Cleanup resources
            if UTILS_AVAILABLE and self.resource_manager:
                try:
                    self.resource_manager.cleanup_all()
                    if self.logger:
                        self.logger.info("Resource cleanup completed")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during cleanup: {e}")


def main():
    """Entry point for SpygateAI."""
    app = SpygateMainApplication()
    return app.main()


if __name__ == "__main__":
    sys.exit(main())