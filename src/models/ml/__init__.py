"""
Machine Learning models and utilities for SpygateAI.
This module contains implementations of various ML models used for game analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

# Type aliases
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Confidence = float
Label = str
Detection = Tuple[BoundingBox, Confidence, Label]

__all__ = [
    'BoundingBox',
    'Confidence',
    'Label',
    'Detection',
] 