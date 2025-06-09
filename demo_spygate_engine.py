#!/usr/bin/env python3
"""
SpygateAI Engine Demo
Comprehensive demonstration of all major PRD features.
"""

import json
from pathlib import Path

from spygate.core.spygate_engine import SpygateAI


def main():
    print("🎮 SpygateAI Engine - Complete PRD Demo")
    print("=" * 50)

    # Initialize the engine
    engine = SpygateAI()

    # Show system status
    print("\n1️⃣ System Status Check")
    print("-" * 25)
    status = engine.get_system_status()
    print(f"Engine Status: {status['status']}")
    print(f"Engine Version: {status['engine_version']}")
    print(f"Systems Ready: {status['systems_count']}")

    # Demo Feature 1: Hardware Optimization
    print("\n2️⃣ Hardware-Adaptive Performance")
    print("-" * 35)
    optimization = engine.optimize_for_hardware()
    print(f"Performance Tier: {optimization['performance_tier']}")
    print(f"GPU Memory: {optimization['gpu_memory_gb']:.1f} GB")
    print(f"Optimizations: {len(optimization['optimizations_applied'])}")

    # Demo Feature 2: Quick Analysis
    print("\n3️⃣ Quick Analysis Demo")
    print("-" * 25)
    # Use a test video file if it exists, otherwise simulate
    test_video = "video-720.mp4" if Path("video-720.mp4").exists() else "demo_video.mp4"
    quick_result = engine.quick_analysis(test_video)
    print(f"Video: {Path(quick_result['video_file']).name}")
    print(f"Detected Situations: {', '.join(quick_result['detected_situations'])}")
    print(f"Analysis Time: {quick_result['analysis_time']}")
    print(f"Confidence: {quick_result['confidence']}")

    # Demo Feature 3: Cross-Game Strategy Migration
    print("\n4️⃣ Cross-Game Strategy Migration")
    print("-" * 35)
    day_one_plan = engine.migrate_strategies_for_new_game("madden_26")
    print(f"Target Game: Madden 26")
    print(f"Ready Concepts: {len(day_one_plan['actionable_concepts'])}")
    print(f"Needs Adaptation: {len(day_one_plan['manual_adaptation_needed'])}")

    # Demo Feature 4: Tournament Preparation
    print("\n5️⃣ Tournament Preparation Mode")
    print("-" * 33)
    tournament_prep = engine.prepare_for_tournament_match(
        opponent_username="DemoOpponent",
        opponent_footage_files=[test_video],
        tournament_type="mcs_qualifier",
    )
    print(f"Opponent: {tournament_prep['opponent_profile']['username']}")
    print(f"Confidence Score: {tournament_prep['opponent_profile']['confidence_score']:.1f}%")
    print(f"Counter Strategies: {len(tournament_prep['gameplan']['counter_strategies'])}")
    print(f"Estimated Prep Time: {tournament_prep['preparation_summary']['estimated_prep_time']}")

    # Demo Feature 5: Situational Library
    print("\n6️⃣ Situational Library Building")
    print("-" * 35)
    library = engine.build_situational_library("3rd_long")
    print(f"Situation: 3rd & Long")
    print(f"Total Clips: {library['total_clips']}")
    print(f"Games Analyzed: {len(library['clips_by_game'])}")
    print(f"Strategies: {', '.join(library['recommended_strategies'])}")

    # Demo Feature 6: Analyze Any Footage (Core Feature)
    print("\n7️⃣ Analyze Any Footage (Core Feature)")
    print("-" * 40)
    print("🎬 Analyzing demo footage...")

    # Simulate the core analysis workflow
    analysis = engine.analyze_any_footage(
        video_file=test_video, context="my_gameplay", auto_export=True
    )

    print(f"Video: {Path(analysis['video_file']).name}")
    print(f"Game Version: {analysis['game_version']}")
    print(f"Key Clips Found: {len(analysis['detected_clips'])}")
    print(f"Exported Clips: {len(analysis['export_results'])}")
    print(f"Actionable Insights: {len(analysis['actionable_insights'])}")

    if analysis["actionable_insights"]:
        print("\n📋 Key Insights:")
        for insight in analysis["actionable_insights"][:3]:
            print(f"  • {insight}")

    # Summary
    print("\n" + "=" * 50)
    print("🎯 SpygateAI Engine Demo Complete!")
    print("All PRD core features demonstrated successfully.")
    print("=" * 50)

    return {
        "demo_status": "completed",
        "features_tested": [
            "System Status",
            "Hardware Optimization",
            "Quick Analysis",
            "Cross-Game Migration",
            "Tournament Prep",
            "Situational Library",
            "Core Analysis Workflow",
        ],
        "engine_status": status["status"],
        "total_clips_processed": len(analysis["detected_clips"]),
    }


if __name__ == "__main__":
    try:
        demo_results = main()
        print(f"\n✅ Demo completed successfully!")
        print(f"Features tested: {len(demo_results['features_tested'])}")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
