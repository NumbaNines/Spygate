"""
SpygateAI HUD Region Detection - Class Definitions
"""

# NEW 5-Class System for HUD Region Detection
HUD_REGION_CLASSES = [
    "hud",                      # 0: Main HUD bar region
    "possession_triangle_area", # 1: Left triangle area (between team abbreviations/scores, shows ball possession)
    "territory_triangle_area",  # 2: Right triangle area (next to field position, shows territory ▲=opponent's ▼=own)
    "preplay_indicator",       # 3: Bottom left pre-play indicator
    "play_call_screen"         # 4: Play call screen overlay
]

# Class mapping for YOLO
CLASS_MAPPING = {name: idx for idx, name in enumerate(HUD_REGION_CLASSES)}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    "hud": (255, 255, 0),                    # Cyan - Main HUD
    "possession_triangle_area": (0, 255, 0), # Green - Possession area
    "territory_triangle_area": (0, 0, 255),  # Red - Territory area  
    "preplay_indicator": (255, 0, 255),      # Magenta - Pre-play
    "play_call_screen": (0, 165, 255)       # Orange - Play call
}

def get_class_info():
    """Return complete class information."""
    return {
        "classes": HUD_REGION_CLASSES,
        "mapping": CLASS_MAPPING,
        "colors": CLASS_COLORS,
        "count": len(HUD_REGION_CLASSES)
    }
