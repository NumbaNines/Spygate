#!/usr/bin/env python3
"""
Demo: Advanced Situation Tracking with Hidden MMR System

This demonstrates how SpygateAI can now track sophisticated game situations
using our perfect triangle detection for possession/territory awareness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from spygate.ml.enhanced_game_analyzer import (
    EnhancedGameAnalyzer, GameState, SituationContext, 
    PerformanceTier, HiddenMMRMetrics
)

def demo_advanced_situation_tracking():
    """Demonstrate advanced situation tracking capabilities."""
    
    print("🎯 SpygateAI Advanced Situation Tracking Demo")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = EnhancedGameAnalyzer()
    
    # Set user context (user is home team)
    analyzer.set_user_context(user_team="home", analysis_type="self")
    print("✅ User context set: Home team, Self-analysis")
    print()
    
    # Demo scenarios with different game situations
    scenarios = [
        {
            "name": "Red Zone Offense",
            "game_state": GameState(
                possession_team="home",  # User has ball
                territory="opponent",    # In opponent territory
                down=2,
                distance=8,
                yard_line=15,  # 15 yards from goal
                quarter=3,
                time="8:45",
                score_home=14,
                score_away=10
            )
        },
        {
            "name": "Third and Long Defense", 
            "game_state": GameState(
                possession_team="away",  # Opponent has ball
                territory="own",         # In own territory
                down=3,
                distance=12,
                yard_line=35,
                quarter=4,
                time="3:22",
                score_home=21,
                score_away=17
            )
        },
        {
            "name": "Two-Minute Drill",
            "game_state": GameState(
                possession_team="home",  # User has ball
                territory="own",         # In own territory
                down=1,
                distance=10,
                yard_line=25,  # Backed up
                quarter=4,
                time="1:45",
                score_home=17,
                score_away=21  # User trailing
            )
        },
        {
            "name": "Goal Line Defense",
            "game_state": GameState(
                possession_team="away",  # Opponent has ball
                territory="own",         # In own territory
                down=3,
                distance=3,
                yard_line=2,   # 2 yards from goal
                quarter=4,
                time="0:45",
                score_home=24,
                score_away=21  # User leading by 3
            )
        }
    ]
    
    print("🏈 ANALYZING GAME SITUATIONS")
    print("-" * 40)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("   " + "="*30)
        
        # Analyze the situation
        context = analyzer.analyze_advanced_situation(scenario['game_state'])
        
        # Display analysis
        print(f"   📊 Possession: {context.possession_team.upper()}")
        print(f"   🏟️  Territory: {context.territory}")
        print(f"   🎯 Situation: {context.situation_type}")
        print(f"   ⚡ Pressure: {context.pressure_level}")
        print(f"   📈 Leverage: {context.leverage_index:.2f}")
        
        # Show what this means strategically
        strategic_meaning = get_strategic_meaning(context, scenario['game_state'])
        print(f"   💡 Strategy: {strategic_meaning}")
        
        # Simulate some outcomes for MMR tracking
        if context.possession_team == "user":
            simulate_offensive_outcome(analyzer, scenario['game_state'], context)
        else:
            simulate_defensive_outcome(analyzer, scenario['game_state'], context)
    
    print("\n" + "="*60)
    print("🎮 HIDDEN MMR PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Get hidden performance summary
    performance = analyzer.get_hidden_performance_summary()
    
    print(f"🏆 Performance Tier: {performance['tier_name']}")
    print(f"📊 Hidden MMR Score: {performance['hidden_mmr_score']:.1f}/100")
    print(f"🎯 Analysis Context: {performance['analysis_context']}")
    
    print("\n📈 Situational Breakdown:")
    breakdown = performance['situational_breakdown']
    for metric, value in breakdown.items():
        print(f"   • {metric.replace('_', ' ').title()}: {value:.2f}")
    
    print("\n📋 Situation Counts:")
    counts = performance['situation_counts']
    for category, count in counts.items():
        print(f"   • {category.replace('_', ' ').title()}: {count}")
    
    print("\n" + "="*60)
    print("✨ KEY INSIGHTS")
    print("-" * 40)
    
    insights = generate_insights(performance, analyzer)
    for insight in insights:
        print(f"💡 {insight}")
    
    print(f"\n🎯 This demonstrates how SpygateAI can now track:")
    print("   • Sophisticated offensive/defensive situations")
    print("   • Hidden performance metrics (MMR system)")
    print("   • Strategic context awareness")
    print("   • Pressure and leverage analysis")
    print("   • Performance tier classification")

def get_strategic_meaning(context: SituationContext, game_state: GameState) -> str:
    """Get strategic meaning of the situation."""
    
    if context.situation_type == "red_zone_offense":
        return "High scoring probability - focus on execution"
    elif context.situation_type == "third_and_long":
        if context.possession_team == "opponent":
            return "Force punt opportunity - aggressive pass rush"
        else:
            return "Conversion challenge - consider checkdown"
    elif context.situation_type == "two_minute_drill":
        return "Clock management critical - balance speed/accuracy"
    elif context.situation_type == "goal_line_defense":
        return "Prevent touchdown - stack the box"
    elif context.pressure_level == "critical":
        return "High-stakes moment - avoid mistakes"
    else:
        return "Standard situation - execute fundamentals"

def simulate_offensive_outcome(analyzer, game_state: GameState, context: SituationContext):
    """Simulate an offensive play outcome for MMR tracking."""
    
    # Simulate different outcomes based on situation
    if context.situation_type == "red_zone_offense":
        # Simulate touchdown
        analyzer.hidden_mmr.red_zone_efficiency += 0.2
        analyzer.hidden_mmr.consistency += 0.1
    elif context.situation_type == "two_minute_drill":
        # Simulate good clock management
        analyzer.hidden_mmr.clock_management += 0.15
        analyzer.hidden_mmr.pressure_performance += 0.1

def simulate_defensive_outcome(analyzer, game_state: GameState, context: SituationContext):
    """Simulate a defensive play outcome for MMR tracking."""
    
    if context.situation_type == "third_and_long":
        # Simulate forcing punt
        analyzer.hidden_mmr.pressure_performance += 0.15
        analyzer.hidden_mmr.situational_play_calling += 0.1
    elif context.situation_type == "goal_line_defense":
        # Simulate preventing touchdown
        analyzer.hidden_mmr.clutch_factor += 0.2
        analyzer.hidden_mmr.momentum_management += 0.1

def generate_insights(performance: dict, analyzer) -> list:
    """Generate strategic insights based on performance."""
    
    insights = []
    
    mmr_score = performance['hidden_mmr_score']
    tier = performance['tier_name']
    
    if mmr_score >= 85:
        insights.append("Elite-level situational awareness detected")
    elif mmr_score >= 70:
        insights.append("Strong strategic fundamentals with room for improvement")
    else:
        insights.append("Focus on situational decision-making improvement")
    
    # Analyze specific strengths/weaknesses
    breakdown = performance['situational_breakdown']
    
    if breakdown['red_zone_efficiency'] > 0.15:
        insights.append("Excellent red zone execution - maintain this strength")
    elif breakdown['red_zone_efficiency'] < -0.1:
        insights.append("Red zone struggles detected - practice goal line situations")
    
    if breakdown['pressure_performance'] > 0.1:
        insights.append("Performs well under pressure - clutch player")
    elif breakdown['pressure_performance'] < -0.1:
        insights.append("Pressure situations need work - practice high-leverage moments")
    
    if breakdown['turnover_avoidance'] < -0.15:
        insights.append("Ball security is a concern - focus on protecting possession")
    
    return insights

if __name__ == "__main__":
    demo_advanced_situation_tracking() 