#!/usr/bin/env python3
"""Test script for SpygateAI engine initialization."""

try:
    print("Testing SpygateAI engine...")
    from spygate.core.spygate_engine import SpygateAI

    print("✅ Import successful")

    engine = SpygateAI()
    print("✅ Engine initialized successfully")

    # Test basic functionality
    status = engine.get_system_status()
    print(f"✅ System status: {status['status']}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
