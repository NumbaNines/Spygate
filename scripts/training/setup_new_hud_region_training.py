#!/usr/bin/env python3
"""
Setup script for NEW 5-Class HUD Region Detection Training
REPLACES all previous triangle detection attempts.

NEW CLASSES:
0: hud                      - Main HUD bar region
1: possession_triangle_area - Left triangle area (possession indicator)  
2: territory_triangle_area  - Right triangle area (territory indicator)
3: preplay_indicator       - Bottom left pre-play indicator
4: play_call_screen        - Play call screen overlay

This approach uses YOLO for robust region detection, then applies
computer vision techniques within those regions for precise analysis.
"""

import logging
import shutil
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# New class definitions
NEW_CLASSES = [
    "hud",                      # 0: Main HUD bar region
    "possession_triangle_area", # 1: Left triangle area (possession indicator)  
    "territory_triangle_area",  # 2: Right triangle area (territory indicator)
    "preplay_indicator",       # 3: Bottom left pre-play indicator
    "play_call_screen"         # 4: Play call screen overlay
]

def create_directory_structure():
    """Create clean directory structure for new training approach."""
    
    # Create main training directory
    training_dir = Path("hud_region_training")
    if training_dir.exists():
        logger.warning(f"Directory {training_dir} already exists. Removing...")
        shutil.rmtree(training_dir)
    
    # Create directory structure
    directories = [
        training_dir / "images",
        training_dir / "annotations_labelme", 
        training_dir / "annotations_yolo",
        training_dir / "datasets" / "train" / "images",
        training_dir / "datasets" / "train" / "labels", 
        training_dir / "datasets" / "val" / "images",
        training_dir / "datasets" / "val" / "labels",
        training_dir / "models",
        training_dir / "results"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {directory}")
    
    return training_dir

def create_dataset_yaml(training_dir: Path):
    """Create dataset.yaml file for YOLO training."""
    
    yaml_content = f"""# SpygateAI HUD Region Detection Dataset
# 5-Class approach for robust region detection + computer vision analysis

# Dataset paths
path: {training_dir.absolute() / 'datasets'}
train: train/images
val: val/images

# Number of classes
nc: {len(NEW_CLASSES)}

# Class names
names:
"""
    
    for i, class_name in enumerate(NEW_CLASSES):
        yaml_content += f"  {i}: {class_name}\n"
    
    yaml_content += """
# Class descriptions:
# 0: hud - Main HUD bar containing all game information
# 1: possession_triangle_area - Left triangle area showing ball possession
# 2: territory_triangle_area - Right triangle area showing field territory  
# 3: preplay_indicator - Bottom left indicator (pre-play state)
# 4: play_call_screen - Play call screen (post-play state)
"""
    
    yaml_path = training_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created dataset configuration: {yaml_path}")
    return yaml_path

def create_class_definitions_file(training_dir: Path):
    """Create a Python file with class definitions for easy import."""
    
    py_content = '''"""
SpygateAI HUD Region Detection - Class Definitions
"""

# NEW 5-Class System for HUD Region Detection
HUD_REGION_CLASSES = [
    "hud",                      # 0: Main HUD bar region
    "possession_triangle_area", # 1: Left triangle area (possession indicator)  
    "territory_triangle_area",  # 2: Right triangle area (territory indicator)
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
'''
    
    py_path = training_dir / "class_definitions.py"
    with open(py_path, 'w') as f:
        f.write(py_content)
    
    logger.info(f"Created class definitions: {py_path}")
    return py_path

def create_annotation_instructions(training_dir: Path):
    """Create detailed annotation instructions."""
    
    instructions = """# HUD Region Annotation Instructions

## NEW 5-Class System Overview:

This approach focuses on detecting **REGIONS** rather than tiny elements,
then using computer vision within those regions for precise analysis.

## Classes to Annotate:

### 1. hud (Class 0)
- **What**: The entire main HUD bar at the bottom of the screen
- **How**: Draw a rectangle covering the complete dark HUD bar
- **Size**: Should capture all HUD elements (scores, clock, down/distance, etc.)
- **Note**: This is the largest and most important region

### 2. possession_triangle_area (Class 1) 
- **What**: Left triangle area between team abbreviations
- **How**: Draw a small rectangle around where the possession triangle appears
- **Location**: Between the away/home team names on the left side
- **Size**: Make box generous enough to always capture the triangle
- **Note**: Triangle direction indicates which team has possession

### 3. territory_triangle_area (Class 2)
- **What**: Right triangle area next to yard line indicator  
- **How**: Draw a small rectangle around where the territory triangle appears
- **Location**: Far right side of HUD, next to the yard number
- **Size**: Make box generous enough to always capture the triangle
- **Note**: Triangle direction (▲▼) indicates field territory

### 4. preplay_indicator (Class 3)
- **What**: Bottom left indicator that appears only before plays
- **How**: Draw rectangle around the pre-play indicator when visible
- **Location**: Bottom left corner of screen
- **Note**: Only annotate when this indicator is actually visible

### 5. play_call_screen (Class 4)
- **What**: Play call screen overlay that appears after plays
- **How**: Draw rectangle around play call interface when visible  
- **Location**: Usually center/overlay on the main game view
- **Note**: Only annotate when play call screen is actually shown

## Annotation Tips:

1. **Be Generous with Box Sizes**: Better to have slightly larger boxes than miss content
2. **Triangle Areas**: Focus on the AREA, not the triangle itself
3. **Consistency**: Try to keep box sizes consistent across similar images
4. **Visibility**: Only annotate elements that are actually visible and clear
5. **HUD Priority**: The 'hud' class is most important - make sure it's always accurate

## Processing Pipeline:

1. YOLO detects these 5 regions reliably
2. Computer vision processes content within each region:
   - OCR for text in HUD region
   - Triangle detection/analysis in triangle areas
   - State detection for preplay/playcall indicators

This approach gives us the best of both worlds: robust detection + precise analysis!
"""
    
    instructions_path = training_dir / "ANNOTATION_INSTRUCTIONS.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    logger.info(f"Created annotation instructions: {instructions_path}")
    return instructions_path

def clean_old_conflicts():
    """Remove or rename old files that might conflict."""
    
    logger.info("Checking for potential conflicts with old triangle detection...")
    
    # Files to check/warn about
    potential_conflicts = [
        "test_triangle_model.py",
        "gui_live_detection.py", 
        "train_triangle_model.py"
    ]
    
    conflicts_found = []
    for file_name in potential_conflicts:
        if Path(file_name).exists():
            conflicts_found.append(file_name)
    
    if conflicts_found:
        logger.warning("Found files that may reference old triangle classes:")
        for conflict in conflicts_found:
            logger.warning(f"  - {conflict}")
        logger.warning("These files may need updating to use the new 5-class system")
    
    return conflicts_found

def main():
    """Set up the new HUD region training system."""
    
    logger.info("🚀 Setting up NEW 5-Class HUD Region Detection Training")
    logger.info("This REPLACES all previous triangle detection attempts")
    
    try:
        # Clean up potential conflicts
        conflicts = clean_old_conflicts()
        
        # Create directory structure
        training_dir = create_directory_structure()
        
        # Create configuration files
        dataset_yaml = create_dataset_yaml(training_dir)
        class_defs = create_class_definitions_file(training_dir)
        instructions = create_annotation_instructions(training_dir)
        
        # Success summary
        logger.info("✅ Setup completed successfully!")
        logger.info(f"📁 Training directory: {training_dir}")
        logger.info(f"📄 Dataset config: {dataset_yaml}")
        logger.info(f"🐍 Class definitions: {class_defs}")
        logger.info(f"📋 Instructions: {instructions}")
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Collect screenshots with varied HUD states")
        logger.info("2. Use labelme to annotate the 5 region classes")
        logger.info("3. Convert labelme annotations to YOLO format")
        logger.info("4. Train YOLOv8 model with new classes")
        logger.info("5. Implement computer vision analysis within detected regions")
        
        if conflicts:
            logger.info(f"\n⚠️  NOTE: {len(conflicts)} files may need updating for new classes")
        
        logger.info(f"\n📚 NEW CLASSES: {NEW_CLASSES}")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 