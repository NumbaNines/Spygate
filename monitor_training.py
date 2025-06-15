#!/usr/bin/env python3
"""Monitor training progress"""

import os
import time
from pathlib import Path

model_dir = Path("models/expert_madden_ocr")

print("🔍 Monitoring Expert OCR Training Progress")
print("=" * 50)

while True:
    try:
        # Check if model directory exists and has files
        if model_dir.exists():
            files = list(model_dir.glob("*"))
            if files:
                print(f"📁 Model directory contents ({len(files)} files):")
                for file in files:
                    size = file.stat().st_size if file.is_file() else "DIR"
                    print(
                        f"  - {file.name}: {size} bytes"
                        if isinstance(size, int)
                        else f"  - {file.name}: {size}"
                    )
            else:
                print("📁 Model directory exists but is empty")
        else:
            print("📁 Model directory not created yet")

        # Check for any .h5 files in current directory
        h5_files = list(Path(".").glob("*.h5"))
        if h5_files:
            print(f"🎯 Found .h5 files in current directory: {[f.name for f in h5_files]}")

        print(f"⏰ {time.strftime('%H:%M:%S')} - Checking again in 30 seconds...")
        print("-" * 30)
        time.sleep(30)

    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(10)
