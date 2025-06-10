#!/usr/bin/env python3

try:
    from spygate.video.formation_analyzer import FormationType

    print("FormationType imported successfully!")
    print("Available formations:")
    for formation in FormationType:
        print(f"  {formation.name}: {formation.value}")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback

    traceback.print_exc()
