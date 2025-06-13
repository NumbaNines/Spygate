#!/usr/bin/env python3
"""
Application launcher for Spygate
This script properly sets up the Python path to resolve import issues
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so absolute imports work
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Change to the spygate directory for relative file access
os.chdir(current_dir)

# Now import and run the application
if __name__ == "__main__":
    try:
        # Import the main application
        from spygate.main import main

        # Run the application
        sys.exit(main())

    except ImportError as e:
        print(f"Import error: {e}")
        print("Falling back to simple main...")

        # Fallback to simple main if full app has issues
        from spygate.simple_main import main as simple_main

        sys.exit(simple_main())

    except Exception as e:
        print(f"Application error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
