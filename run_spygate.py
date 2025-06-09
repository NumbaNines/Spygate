#!/usr/bin/env python3
"""
Spygate Application Launcher
============================

This script serves as the main entry point for the Spygate application.
Run this from the root directory: python run_spygate.py

Expert Developer Note: This solves the relative import issues by:
1. Adding the spygate directory to Python path
2. Running as a proper module
3. Ensuring all imports work correctly
"""

import os
import sys
from pathlib import Path

# Add the spygate directory to Python path
SPYGATE_ROOT = Path(__file__).parent
SPYGATE_DIR = SPYGATE_ROOT / "spygate"

if str(SPYGATE_DIR) not in sys.path:
    sys.path.insert(0, str(SPYGATE_DIR))

# Now we can import and run the main application
if __name__ == "__main__":
    try:
        # Change to spygate directory for relative file paths
        os.chdir(SPYGATE_DIR)
        
        # Import and run main
        from main import main
        main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Attempting to diagnose...")
        
        # Diagnostic information
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"🐍 Python path: {sys.path[:3]}...")
        print(f"📦 Spygate directory exists: {SPYGATE_DIR.exists()}")
        
        if SPYGATE_DIR.exists():
            print(f"📋 Spygate contents: {[f.name for f in SPYGATE_DIR.iterdir() if f.is_file()][:5]}")
        
        sys.exit(1)
    except Exception as e:
        print(f"❌ Application Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 