"""
SpygateAI HUD Region Detection - 8-Class Expanded System
Backward compatible with existing 5-class model
"""

# EXPANDED 8-Class System for Enhanced HUD Detection
HUD_REGION_CLASSES_8 = [
    # Existing 5 classes (preserve order for backward compatibility)
    "hud",                      # 0: Main HUD bar region
    "possession_triangle_area", # 1: Left triangle area (between team abbreviations/scores, shows ball possession)
    "territory_triangle_area",  # 2: Right triangle area (next to field position, shows territory ▲=opponent's ▼=own)
    "preplay_indicator",       # 3: Bottom left pre-play indicator
    "play_call_screen",        # 4: Play call screen overlay
    
    # NEW classes for enhanced detection
    "down_distance_area",      # 5: Down and distance text region (e.g., "3rd & 8")
    "game_clock_area",         # 6: Game clock region (quarter time)
    "play_clock_area"          # 7: Play clock region (40-second countdown)
]

# Legacy 5-class system for backward compatibility
HUD_REGION_CLASSES_5 = HUD_REGION_CLASSES_8[:5]

# Class mapping for YOLO (8-class)
CLASS_MAPPING_8 = {name: idx for idx, name in enumerate(HUD_REGION_CLASSES_8)}

# Class mapping for legacy 5-class
CLASS_MAPPING_5 = {name: idx for idx, name in enumerate(HUD_REGION_CLASSES_5)}

# Colors for visualization (BGR format) - 8-class
CLASS_COLORS_8 = {
    # Existing colors
    "hud": (255, 255, 0),                    # Cyan - Main HUD
    "possession_triangle_area": (0, 255, 0), # Green - Possession area
    "territory_triangle_area": (0, 0, 255),  # Red - Territory area  
    "preplay_indicator": (255, 0, 255),      # Magenta - Pre-play
    "play_call_screen": (0, 165, 255),      # Orange - Play call
    
    # New colors for new classes
    "down_distance_area": (255, 255, 255),   # White - Down/Distance
    "game_clock_area": (128, 128, 128),      # Gray - Game Clock
    "play_clock_area": (255, 128, 0)         # Blue - Play Clock
}

# Legacy colors for 5-class
CLASS_COLORS_5 = {k: v for k, v in CLASS_COLORS_8.items() if k in HUD_REGION_CLASSES_5}

def get_class_info_8():
    """Return complete 8-class information."""
    return {
        "classes": HUD_REGION_CLASSES_8,
        "mapping": CLASS_MAPPING_8,
        "colors": CLASS_COLORS_8,
        "count": len(HUD_REGION_CLASSES_8)
    }

def get_class_info_5():
    """Return legacy 5-class information for backward compatibility."""
    return {
        "classes": HUD_REGION_CLASSES_5,
        "mapping": CLASS_MAPPING_5,
        "colors": CLASS_COLORS_5,
        "count": len(HUD_REGION_CLASSES_5)
    }

def get_class_info(use_8_class=True):
    """Return class information based on model type."""
    return get_class_info_8() if use_8_class else get_class_info_5()

# Migration mapping for converting 5-class to 8-class annotations
MIGRATION_MAPPING = {
    # Existing classes remain the same
    0: 0,  # hud -> hud
    1: 1,  # possession_triangle_area -> possession_triangle_area
    2: 2,  # territory_triangle_area -> territory_triangle_area
    3: 3,  # preplay_indicator -> preplay_indicator
    4: 4,  # play_call_screen -> play_call_screen
    # New classes will be added through annotation
} 