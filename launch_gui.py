#!/usr/bin/env python3
"""
Quick launcher for SpygateAI Live Detection GUI
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the GUI."""
    gui_script = Path(__file__).parent / "gui_live_detection.py"
    
    if not gui_script.exists():
        print("❌ GUI script not found!")
        return
    
    print("🚀 Launching SpygateAI Live Detection GUI...")
    try:
        subprocess.run([sys.executable, str(gui_script)], check=True)
    except KeyboardInterrupt:
        print("\n👋 GUI closed")
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")

if __name__ == "__main__":
    main() 