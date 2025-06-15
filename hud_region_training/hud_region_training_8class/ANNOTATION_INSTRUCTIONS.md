
# 8-Class HUD Annotation Instructions

## New Classes to Annotate (in addition to existing 5):

### 5. down_distance_area
- **Location**: Usually center-left of HUD
- **Contains**: Down and distance text (e.g., "3rd & 8", "1st & 10")
- **Shape**: Rectangle around the down/distance text

### 6. game_clock_area
- **Location**: Usually center of HUD
- **Contains**: Quarter and game time (e.g., "1st 12:34", "4th 2:00")
- **Shape**: Rectangle around the game clock

### 7. play_clock_area
- **Location**: Usually near game clock or separate area
- **Contains**: Play clock countdown (e.g., "25", "40")
- **Shape**: Rectangle around the play clock number

## Annotation Workflow:

1. Start labelme: `labelme hud_region_training_8class/datasets_8class/train/images --config hud_region_training_8class/labelme_config.json`

2. For each image:
   - Draw rectangles around ALL 8 HUD elements
   - Use the exact class names from the config
   - Save annotations in JSON format

3. Convert to YOLO format when done:
   - Use the conversion script (will be created)

## Tips:
- Focus on the NEW classes (5-7) since existing classes (0-4) can be migrated
- Make rectangles tight around text but with small padding
- Be consistent with rectangle sizes across similar elements
- If an element is not visible, skip that class for that image
