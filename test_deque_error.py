#!/usr/bin/env python3
"""Test script to reproduce the deque initialization error in EnhancedGameAnalyzer."""

import traceback


def test_analyzer_init():
    """Test EnhancedGameAnalyzer initialization to reproduce deque error."""
    try:
        print("📦 Importing EnhancedGameAnalyzer...")
        from src.spygate.ml.enhanced_game_analyzer import EnhancedGameAnalyzer

        print("✅ Import successful")

        print("🚀 Initializing EnhancedGameAnalyzer...")
        analyzer = EnhancedGameAnalyzer()
        print("✅ Initialization successful")

        return analyzer

    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_analyzer_init()
