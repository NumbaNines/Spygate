"""
Layout Configuration System
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QSizePolicy


@dataclass
class LayoutConfig:
    """Configuration for layout settings."""

    margins: tuple[int, int, int, int] = (0, 0, 0, 0)
    spacing: int = 0
    min_size: Optional[QSize] = None
    max_size: Optional[QSize] = None
    horizontal_policy: QSizePolicy.Policy = QSizePolicy.Policy.Preferred
    vertical_policy: QSizePolicy.Policy = QSizePolicy.Policy.Preferred
    alignment: Optional[Qt.AlignmentFlag] = None


class LayoutConfigs:
    """Predefined layout configurations."""

    # Common margins
    NO_MARGINS = LayoutConfig()
    SMALL_MARGINS = LayoutConfig(margins=(4, 4, 4, 4))
    MEDIUM_MARGINS = LayoutConfig(margins=(8, 8, 8, 8))
    LARGE_MARGINS = LayoutConfig(margins=(16, 16, 16, 16))

    # Common spacings
    TIGHT_SPACING = LayoutConfig(spacing=2)
    NORMAL_SPACING = LayoutConfig(spacing=4)
    LOOSE_SPACING = LayoutConfig(spacing=8)

    # Panel configurations
    PANEL_CONFIG = LayoutConfig(
        margins=(8, 8, 8, 8),
        spacing=4,
        min_size=QSize(200, 100),
    )

    # Dialog configurations
    DIALOG_CONFIG = LayoutConfig(
        margins=(16, 16, 16, 16),
        spacing=16,
    )

    # Toolbar configurations
    TOOLBAR_CONFIG = LayoutConfig(
        margins=(4, 0, 4, 0),
        spacing=2,
        horizontal_policy=QSizePolicy.Policy.Expanding,
        vertical_policy=QSizePolicy.Policy.Fixed,
    )

    # Status bar configurations
    STATUSBAR_CONFIG = LayoutConfig(
        margins=(4, 2, 4, 2),
        spacing=8,
        vertical_policy=QSizePolicy.Policy.Fixed,
    )

    # Video player configurations
    VIDEO_PLAYER_CONFIG = LayoutConfig(
        margins=(0, 0, 0, 0),
        spacing=0,
        min_size=QSize(640, 360),
        horizontal_policy=QSizePolicy.Policy.Expanding,
        vertical_policy=QSizePolicy.Policy.Expanding,
    )

    # Analysis panel configurations
    ANALYSIS_PANEL_CONFIG = LayoutConfig(
        margins=(8, 8, 8, 8),
        spacing=4,
        min_size=QSize(250, 300),
        max_size=QSize(400, 16777215),  # Max height
        horizontal_policy=QSizePolicy.Policy.Fixed,
        vertical_policy=QSizePolicy.Policy.Expanding,
    )
