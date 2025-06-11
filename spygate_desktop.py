#!/usr/bin/env python3
"""
SpygateAI Desktop Application
============================

Main entry point for the SpygateAI desktop application.
This is the file users should run to start the application.

Features:
- FaceIt-style modern UI with custom window controls
- Frameless window with drag-to-move functionality
- User profile management with premium subscriptions
- Video analysis upload interface
- Formation builder and game planning tools
- Dashboard with analytics and quick actions

Usage:
    python spygate_desktop.py

Requirements:
    - Python 3.8+
    - PyQt6
    - All dependencies listed in requirements.txt
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path to ensure imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import PyQt6
    except ImportError:
        missing_deps.append("PyQt6")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install them using:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def main():
    """Main entry point for SpygateAI Desktop"""
    print("🏈 Starting SpygateAI Desktop Application...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if required files exist
    required_files = [
        "spygate_desktop_app_faceit_style.py",
        "user_database.py",
        "profile_picture_manager.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure you have all SpygateAI files in the current directory.")
        sys.exit(1)
    
    try:
        # Import and run the main application
        from spygate_desktop_app_faceit_style import SpygateDesktop
        from PyQt6.QtWidgets import QApplication
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # Modern style
        
        # Create main window
        print("✅ Dependencies loaded successfully")
        print("🚀 Launching SpygateAI Desktop...")
        
        window = SpygateDesktop()
        window.show()
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"❌ Failed to start SpygateAI Desktop: {e}")
        print("\nIf this error persists, please check:")
        print("1. All dependencies are properly installed")
        print("2. You're running Python 3.8 or later")
        print("3. All SpygateAI files are present")
        sys.exit(1)

if __name__ == "__main__":
    main() 