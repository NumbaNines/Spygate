"""
SpygateAI Django Project
"""

from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root) 