#!/usr/bin/env python3
"""Use the real cropped templates from raw_gameplay."""

import shutil
from pathlib import Path

# Copy the real cropped templates
mappings = {
    "templates/raw_gameplay/1st_10.png": "down_templates_real/1ST.png",
    "templates/raw_gameplay/2nd_7.png": "down_templates_real/2ND.png", 
    "templates/raw_gameplay/3rd_3.png": "down_templates_real/3RD.png",
    "templates/raw_gameplay/4th_1.png": "down_templates_real/4TH.png",
    "templates/raw_gameplay/3rd_goal.png": "down_templates_real/3RD_GOAL.png",
    "templates/raw_gameplay/4th_goal.png": "down_templates_real/4TH_GOAL.png"
}

for src, dst in mappings.items():
    if Path(src).exists():
        shutil.copy2(src, dst)
        print(f"✅ {src} → {dst}")
    else:
        print(f"❌ {src} not found")

print("Done!") 