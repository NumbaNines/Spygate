# Performance Tier Implementation Checklist

## 🎯 **1. CLUTCH PLAY (Situational Success)**

### **Detection Requirements:**

- [ ] **4th Down Conversion**:

  - Track: `4th & X` → `1st & 10` + SAME possession indicator
  - Verify: Possession indicator stays with same team
  - Calculate: Yards gained (must be 1+)

- [ ] **Game-Winning Score in Final 2:00**:
  - Track: Game clock ≤ 2:00
  - Detect: Score change that breaks tie OR takes lead
  - Verify: Score differential before/after

### **Implementation Challenges:**

- [ ] Parse game clock format (2:00, 1:30, etc.)
- [ ] Track score differential context
- [ ] Distinguish 4th down success vs failure by possession

---

## 🚀 **2. BIG PLAY (Explosive Gain)**

### **Detection Requirements:**

- [ ] **20+ Yard Gains**:

  - Calculate: Field position change ≥ 20 yards
  - Example: OPP 35 → OPP 10 = 25 yards

- [ ] **10+ Yards on 3rd/4th Down with First Down**:
  - Track: `3rd/4th & X` → `1st & 10`
  - Calculate: Field position change ≥ 10 yards
  - Verify: First down achieved

### **Implementation Challenges:**

- [ ] Accurate field position math (OPP vs OWN territory)
- [ ] Handle territory crossovers - EASY: Monitor territory triangle (▼→▲)
- [ ] Parse yards_to_goal + territory_indicator correctly

---

## ⭐ **3. GOOD PLAY (Solid Gain)**

### **Detection Requirements:**

- [ ] **10-19 Yard Gains**:

  - Calculate: Field position change 10-19 yards
  - No first down requirement

- [ ] **5-9 Yards on 3rd/4th Down with First Down**:
  - Track: `3rd/4th & X` → `1st & 10`
  - Calculate: Field position change 5-9 yards
  - Verify: First down achieved

### **Implementation Challenges:**

- [ ] Distinguish from Big Play (different yardage thresholds)
- [ ] Handle edge cases at territory boundaries

---

## ✅ **4. AVERAGE PLAY (Moderate Gain)**

### **Detection Requirements:**

- [ ] **0-9 Yards, No First Down**:

  - Calculate: Field position change 0-9 yards
  - Track: Normal down progression (1st→2nd→3rd)
  - Verify: NO first down achieved

- [ ] **Exception - Critical Short Yardage**:
  - Track: `4th & 1` → `1st & 10` (still Average despite conversion)

### **Implementation Challenges:**

- [ ] Ensure no overlap with Good Play detection
- [ ] Handle 4th & 1 special case properly

---

## 📉 **5. POOR PLAY (Loss or No Gain)**

### **Detection Requirements:**

- [ ] **Backward Movement (-5 to -1 yards)**:

  - Calculate: Field position change = negative 1-5 yards
  - Example: OPP 25 → OPP 30 = -5 yards

- [ ] **Zero Gains with Pressure**:
  - Calculate: Field position change = 0 yards
  - Track: Down progression with same/increased distance
  - Example: `1st & 10` → `2nd & 10` (no gain)

### **Implementation Challenges:**

- [ ] Distinguish zero gains from measurement/penalty
- [ ] Handle sacks that cross territory lines
- [ ] Detect "pressure" indicators vs normal zero gains

---

## 🔴 **6. TURNOVER PLAY (Critical Loss)**

### **Detection Requirements:**

- [ ] **Interception Detection**:

  - Track: Possession indicator CHANGES (only moving HUD element)
  - Verify: All other HUD elements STATIC for ~2 seconds
  - Result: Interception detected (regardless of outcome)

- [ ] **Fumble Detection**:

  - Detect: RED BOX with "FUMBLE" text above score_bug
  - Track: Possession indicator change (if fumble lost)
  - Result: Fumble detected (regardless of outcome)

- [ ] **Safety Detection**:
  - Track: Opponent score +2 points
  - Verify: Possession change (unique: scoring team gets ball back)

### **Implementation Challenges:**

- [ ] OCR detection of "FUMBLE" text in red box
- [ ] Distinguish fumble vs "No Huddle" in same red box location
- [ ] Handle rapid score changes

---

## 🛡️ **7. DEFENSIVE STAND PLAY (Strong Defense)**

### **Detection Requirements:**

- [ ] **Major Loss (-10 to -6 yards)**:

  - Calculate: Field position change ≥ 6 yards backward
  - Example: OPP 25 → OWN 37 = -12 yard sack

- [ ] **Turnover on Downs**:
  - Track: `4th & X` + possession indicator CHANGES
  - Verify: 0 yards gained (or minimal)
  - Result: Defense gets 1st & 10

### **Implementation Challenges:**

- [ ] Distinguish from regular Poor Play (larger loss threshold)
- [ ] Handle major sacks that cross territory boundaries
- [ ] Verify defensive perspective vs offensive failure

---

## 🔧 **CORE IMPLEMENTATION REQUIREMENTS**

### **User Team Identification (CRITICAL):**

- [ ] **Manual Team Selection**: User specifies "I am HOME" or "I am AWAY"
- [ ] **Possession Pattern Analysis**: Track initial possession direction
- [ ] **Play Clock Behavior**: Detect when play clock is active (user's turn)
- [ ] **Team Abbreviation Tracking**: Monitor which team user controls
- [ ] **Verification System**: Confirm user team assignment is correct

### **Implementation Priority:**

1. **PRIMARY: Manual Setup**: User selects their team at start of analysis ✅
2. **FUTURE: Auto-Detection**: Monitor play clock timing (user gets time to make decisions)
3. **FUTURE: Possession Tracking**: User typically starts with possession in practice
4. **FUTURE: Behavioral Analysis**: Detect "human" decision patterns vs AI patterns

### **Why This Matters:**

- **Performance Analysis**: Need to know which plays are user's vs opponent's
- **Turnover Context**: User interception = bad, User causing interception = good
- **Score Context**: User touchdown = positive, Opponent touchdown = negative
- **Decision Quality**: Only analyze user's team decisions, not AI opponent

### **Field Position Calculation:**

- [ ] Parse `yards_to_goal` + `territory_indicator` (▲▼)
- [ ] Convert to standardized format (OPP 25, OWN 35)
- [ ] Handle special cases ("GL", "2-PT", etc.)
- [ ] Calculate yardage differences accurately
- [ ] **Territory Crossover Detection**: Monitor triangle change (▼→▲ or ▲→▼)

### **Down/Distance Tracking:**

- [ ] Parse `down_distance` text (e.g., "3rd & 7")
- [ ] Track down progression (1st→2nd→3rd→4th)
- [ ] Detect first down conversions (ANY→1st & 10)
- [ ] Handle special cases ("Goal", short yardage)

### **Possession Tracking:**

- [ ] Monitor `possession_indicator` changes
- [ ] Detect static vs moving HUD elements
- [ ] Handle mid-play vs between-play changes
- [ ] Verify possession with down/distance context

### **Time Context:**

- [ ] Parse `game_clock` format variations
- [ ] Detect final 2:00 scenarios
- [ ] Handle quarter changes
- [ ] Track timeout usage context

### **Score Tracking:**

- [ ] Monitor `score_home` and `score_away` changes
- [ ] Calculate score differentials
- [ ] Detect tie-breaking vs lead-extending scores
- [ ] Handle rapid scoring sequences

---

## ⚠️ **EDGE CASES TO HANDLE**

- [ ] **Territory Crossovers**: OWN 5 → OPP 45 = 50 yard gain
  - SOLUTION: Monitor territory indicator triangle (▼→▲ or ▲→▼)
  - Same possession indicator + territory change = 50-yard line crossed
- [ ] **Measurement/Replay**: Same down repeated
- [ ] **Penalty Scenarios**: Down same, field position changes
- [ ] **Commercial Breaks**: HUD disappears/reappears
- [ ] **Rapid Sequences**: TD → Extra Point → Kickoff

- [ ] **No Huddle**: Red box "No Huddle" vs "FUMBLE"

---

**Which performance tier should we deep-dive first for implementation?**
