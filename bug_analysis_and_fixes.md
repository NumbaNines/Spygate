# SpygateAI Clip Detection Bug Analysis & Fixes

## Executive Summary

The SpygateAI system has several critical mismatches between situation detection and clip creation that prevent clips from being created for the right situations. The main issues are:

1. **Situation Type Mismatch**: The enhanced game analyzer returns situation types that don't match what the desktop app expects
2. **Missing Situation Mappings**: Many detected situations have no corresponding clip preference check
3. **Logic Flow Issues**: The clip detection logic has several bugs preventing proper clip creation

## Critical Bugs Found

### Bug #1: Situation Type Mismatch

**Location**: `src/spygate/ml/enhanced_game_analyzer.py` line 2781 vs `spygate_desktop_app_faceit_style.py` line 349

**Problem**: The `_classify_situation_type` method returns situation types like:

- `"third_and_long_red_zone"`
- `"third_and_short_goal_line"`
- `"fourth_down_goal_line"`
- `"goal_line_offense"`
- `"red_zone_offense"`
- `"two_minute_drill"`
- `"normal_play"`

But the desktop app checks for different keys:

- `"3rd_long"` (not `"third_and_long"`)
- `"goal_line"` (not `"goal_line_offense"`)
- `"red_zone"` (not `"red_zone_offense"`)

**Impact**: Clips are never created for these situations because the keys don't match.

### Bug #2: Missing Situation Context Attributes

**Location**: `spygate_desktop_app_faceit_style.py` lines 349-370

**Problem**: The code checks for `situation_context.situation_type` but many important situations are detected through game_state attributes that aren't being checked:

- Penalties are detected but not set as situation_type
- Turnovers need possession change detection
- Sacks need play result analysis
- PAT/Field Goal/Touchdown detection missing

**Impact**: Game situations category clips are rarely created.

### Bug #3: Field Position Logic Error

**Location**: `spygate_desktop_app_faceit_style.py` lines 321-329

**Problem**: The field position checks use raw `yard_line` without considering territory context:

```python
if analysis_game_state.yard_line <= 10 and self.situation_preferences.get("goal_line", False):
```

This doesn't account for which side of the field (own vs opponent territory).

**Impact**: Goal line and red zone clips created at wrong field positions.

### Bug #4: Down/Distance Detection Timing

**Location**: `spygate_desktop_app_faceit_style.py` lines 303-317

**Problem**: The code creates clips for EVERY occurrence of a down, not just when plays happen:

- Creates clips during timeouts
- Creates clips during replays
- Creates clips when OCR re-detects same down

**Impact**: Too many duplicate/unnecessary clips.

## Recommended Fixes

### Fix #1: Create Situation Type Mapping

Add a mapping function to translate between analyzer and desktop app situation types:

```python
def map_situation_type_to_preference(situation_type: str) -> list[str]:
    """Map analyzer situation types to preference keys."""
    mapping = {
        # Third down mappings
        "third_and_long": ["3rd_long", "3rd_down"],
        "third_and_long_red_zone": ["3rd_long", "red_zone"],
        "third_and_short": ["3rd_down"],
        "third_and_short_goal_line": ["3rd_down", "goal_line"],
        "third_and_short_red_zone": ["3rd_down", "red_zone"],
        "third_down": ["3rd_down"],
        "third_down_red_zone": ["3rd_down", "red_zone"],

        # Fourth down mappings
        "fourth_down": ["4th_down"],
        "fourth_down_goal_line": ["4th_down", "goal_line"],
        "fourth_down_red_zone": ["4th_down", "red_zone"],

        # Field position mappings
        "goal_line_offense": ["goal_line"],
        "goal_line_defense": ["goal_line"],
        "red_zone_offense": ["red_zone"],
        "red_zone_defense": ["red_zone"],
        "scoring_position": ["red_zone"],

        # Time-based mappings
        "two_minute_drill": ["two_minute_drill"],
        "fourth_quarter": [],  # No direct mapping

        # Default
        "normal_play": [],
    }

    return mapping.get(situation_type, [])
```

### Fix #2: Enhanced Situation Detection

Add detection for missing situation types in the analyzer:

```python
def _detect_special_situations(self, game_state: GameState, frame_data: dict) -> list[str]:
    """Detect special situations like penalties, turnovers, etc."""
    special_situations = []

    # Penalty detection (yellow flag or "FLAG" text)
    if self._detect_penalty_indicators(frame_data):
        special_situations.append("penalty")

    # PAT detection
    if game_state.down_text and "PAT" in game_state.down_text.upper():
        special_situations.append("pat")

    # Turnover detection (possession change)
    if self._detect_possession_change(game_state):
        special_situations.append("turnover")

    # Touchdown detection (score change by 6)
    if self._detect_touchdown(game_state):
        special_situations.append("touchdown")

    return special_situations
```

### Fix #3: Proper Field Position Context

Update field position checks to use territory context:

```python
# Goal line check with territory
if (hasattr(game_state, 'territory') and
    game_state.territory == "opponent" and
    game_state.yard_line <= 10 and
    self.situation_preferences.get("goal_line", False)):
    clip_matches.append("goal_line")

# Red zone check with territory
elif (hasattr(game_state, 'territory') and
      game_state.territory == "opponent" and
      game_state.yard_line <= 25 and
      self.situation_preferences.get("red_zone", False)):
    clip_matches.append("red_zone")
```

### Fix #4: Play-Based Clip Creation

Only create clips when actual plays occur, not just when down/distance is visible:

```python
def _is_valid_play_occurrence(self, game_state, previous_state, current_frame):
    """Check if this is an actual play, not just OCR re-detection."""

    # Must have valid game state
    if not game_state or not game_state.down:
        return False

    # Check for play progression indicators
    play_indicators = []

    # 1. Down changed
    if previous_state and previous_state.down != game_state.down:
        play_indicators.append("down_change")

    # 2. Distance changed significantly (>2 yards)
    if (previous_state and previous_state.distance and game_state.distance and
        abs(previous_state.distance - game_state.distance) > 2):
        play_indicators.append("distance_change")

    # 3. Clock is running (not timeout)
    if self._is_clock_running(game_state):
        play_indicators.append("clock_running")

    # 4. Not in menu/replay
    if not self._is_in_replay_or_menu(game_state):
        play_indicators.append("live_play")

    # Need at least 2 indicators to confirm real play
    return len(play_indicators) >= 2
```

### Fix #5: Complete \_should_create_clip Rewrite

Here's the complete fixed method:

```python
def _should_create_clip(self, game_state, situation_context) -> bool:
    """Determine if we should create a clip based on selected preferences."""
    if not game_state:
        return False

    # Only process if this is a valid play occurrence
    if not self._is_valid_play_occurrence(game_state, self.previous_game_state,
                                          getattr(situation_context, 'frame_number', 0)):
        return False

    clip_matches = []

    # === DOWNS CATEGORY (Fixed) ===
    if game_state.down is not None:
        if game_state.down == 1 and self.situation_preferences.get("1st_down", False):
            clip_matches.append("1st_down")
        elif game_state.down == 2 and self.situation_preferences.get("2nd_down", False):
            clip_matches.append("2nd_down")
        elif game_state.down == 3:
            if self.situation_preferences.get("3rd_down", False):
                clip_matches.append("3rd_down")
            if game_state.distance and game_state.distance >= 7 and self.situation_preferences.get("3rd_long", False):
                clip_matches.append("3rd_long")
        elif game_state.down == 4 and self.situation_preferences.get("4th_down", False):
            clip_matches.append("4th_down")

    # === MAPPED SITUATION TYPES (New) ===
    if hasattr(situation_context, 'situation_type'):
        mapped_preferences = map_situation_type_to_preference(situation_context.situation_type)
        for pref_key in mapped_preferences:
            if self.situation_preferences.get(pref_key, False):
                clip_matches.append(pref_key)

    # === SPECIAL SITUATIONS (New) ===
    if hasattr(situation_context, 'special_situations'):
        for special_sit in situation_context.special_situations:
            if self.situation_preferences.get(special_sit, False):
                clip_matches.append(special_sit)

    # === FIELD POSITION WITH CONTEXT (Fixed) ===
    if (hasattr(game_state, 'yard_line') and game_state.yard_line and
        hasattr(game_state, 'territory')):

        if (game_state.territory == "opponent" and game_state.yard_line <= 10 and
            self.situation_preferences.get("goal_line", False)):
            clip_matches.append("goal_line")
        elif (game_state.territory == "opponent" and game_state.yard_line <= 25 and
              self.situation_preferences.get("red_zone", False)):
            clip_matches.append("red_zone")

    # Update state and return
    self.previous_game_state = game_state

    if clip_matches:
        print(f"✅ CLIP MATCHES: {clip_matches} for {situation_context.situation_type}")
        return True

    return False
```

## Implementation Priority

1. **High Priority**: Fix situation type mapping (Fix #1) - This will immediately improve clip detection
2. **High Priority**: Fix field position context (Fix #3) - Prevents wrong clips
3. **Medium Priority**: Add special situation detection (Fix #2) - Enables more clip types
4. **Medium Priority**: Add play occurrence validation (Fix #4) - Reduces duplicate clips
5. **Low Priority**: Complete rewrite of \_should_create_clip (Fix #5) - Clean implementation

## Testing Recommendations

1. Create test video with known situations (3rd & long, red zone, etc.)
2. Enable debug logging to verify situation detection
3. Test each clip preference individually
4. Verify no duplicate clips for same play
5. Check clip boundaries include full play

## Configuration File Recommendation

Create a `clip_mapping_config.json`:

```json
{
  "situation_mappings": {
    "third_and_long": ["3rd_long", "3rd_down"],
    "third_and_long_red_zone": ["3rd_long", "red_zone"],
    "goal_line_offense": ["goal_line"],
    "red_zone_offense": ["red_zone"]
  },
  "special_detectors": {
    "penalty": {
      "indicators": ["FLAG", "PENALTY"],
      "color_threshold": 0.3
    },
    "pat": {
      "text_patterns": ["PAT", "P.A.T", "POINT AFTER"]
    }
  },
  "clip_boundaries": {
    "pre_play_buffer": 3.0,
    "post_play_buffer": 2.0,
    "min_clip_duration": 2.0,
    "max_clip_duration": 30.0
  }
}
```

This configuration-driven approach makes the system more maintainable and testable.
