"""
Django models for SpygateAI Web Collaboration Hub.

Implements cross-game data models, strategy sharing, and community features
as specified in the PRD Phase 2 requirements.
"""

import uuid
from enum import TextChoices

from django.contrib.auth.models import AbstractUser
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


class SupportedGame(TextChoices):
    """Supported EA football games for cross-game intelligence."""

    MADDEN_25 = "madden_25", "Madden NFL 25"
    CFB_25 = "cfb_25", "College Football 25"
    MADDEN_26 = "madden_26", "Madden NFL 26"
    CFB_26 = "cfb_26", "College Football 26"


class PerformanceTier(TextChoices):
    """7-Tier Performance Analysis System from PRD."""

    CLUTCH_PLAY = "clutch_play", "Clutch Play (95-100 pts)"
    BIG_PLAY = "big_play", "Big Play (85-94 pts)"
    GOOD_PLAY = "good_play", "Good Play (75-84 pts)"
    AVERAGE_PLAY = "average_play", "Average Play (60-74 pts)"
    POOR_PLAY = "poor_play", "Poor Play (40-59 pts)"
    TURNOVER_PLAY = "turnover_play", "Turnover Play (0-39 pts)"
    DEFENSIVE_STAND = "defensive_stand", "Defensive Stand (0-20 pts)"


class HardwareTier(TextChoices):
    """Hardware tiers for adaptive performance."""

    ULTRA_LOW = "ultra_low", "Ultra Low"
    LOW = "low", "Low"
    MEDIUM = "medium", "Medium"
    HIGH = "high", "High"
    ULTRA = "ultra", "Ultra"


class SpygateUser(AbstractUser):
    """Extended user model for SpygateAI with competitive gaming focus."""

    # Competitive Profile
    mcs_username = models.CharField(
        max_length=50, blank=True, null=True, help_text="MCS Competitor Username"
    )
    players_lounge_username = models.CharField(
        max_length=50, blank=True, null=True, help_text="Players Lounge Username"
    )
    skill_level = models.CharField(
        max_length=20,
        choices=[
            ("casual", "Casual Player"),
            ("competitive", "Competitive Player"),
            ("mcs_grinder", "MCS Grinder"),
            ("mcs_pro", "MCS Professional"),
        ],
        default="casual",
    )

    # Preferences
    primary_game = models.CharField(
        max_length=20, choices=SupportedGame.choices, default=SupportedGame.MADDEN_25
    )
    hardware_tier = models.CharField(
        max_length=20, choices=HardwareTier.choices, default=HardwareTier.MEDIUM
    )
    enable_cross_game_intelligence = models.BooleanField(default=True)
    enable_professional_benchmarking = models.BooleanField(default=True)

    # Statistics
    total_videos_analyzed = models.PositiveIntegerField(default=0)
    total_strategies_created = models.PositiveIntegerField(default=0)
    total_gameplans_shared = models.PositiveIntegerField(default=0)

    # Subscription
    subscription_tier = models.CharField(
        max_length=20,
        choices=[
            ("free", "Free Tier"),
            ("premium", "Premium ($19.99/month)"),
            ("professional", "Professional ($39.99/month)"),
        ],
        default="free",
    )
    subscription_expires = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.username} ({self.get_skill_level_display()})"


class UniversalConcept(models.Model):
    """Universal football concepts that transfer across EA Sports titles."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    category = models.CharField(
        max_length=30,
        choices=[
            ("formation_families", "Formation Families"),
            ("strategic_concepts", "Strategic Concepts"),
            ("defensive_schemes", "Defensive Schemes"),
            ("situational_contexts", "Situational Contexts"),
        ],
    )
    core_principle = models.TextField(help_text="Game-agnostic core principle")
    effectiveness_rating = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Cross-game effectiveness (0.0-1.0)",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(SpygateUser, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering = ["category", "name"]

    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"


class GameSpecificImplementation(models.Model):
    """Game-specific implementations of universal concepts."""

    universal_concept = models.ForeignKey(
        UniversalConcept, on_delete=models.CASCADE, related_name="implementations"
    )
    game = models.CharField(max_length=20, choices=SupportedGame.choices)
    implementation_name = models.CharField(max_length=100)
    formation_code = models.CharField(max_length=50, blank=True, null=True)
    playbook_location = models.CharField(max_length=200, blank=True, null=True)
    success_rate = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)], null=True, blank=True
    )

    # Professional benchmarking data
    pro_usage_frequency = models.FloatField(
        null=True, blank=True, help_text="Professional usage frequency"
    )
    pro_success_rate = models.FloatField(
        null=True, blank=True, help_text="Professional success rate"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ["universal_concept", "game"]
        ordering = ["game", "implementation_name"]

    def __str__(self):
        return f"{self.implementation_name} ({self.get_game_display()})"


class VideoAnalysis(models.Model):
    """Analysis results for uploaded videos with cross-game intelligence."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(SpygateUser, on_delete=models.CASCADE, related_name="video_analyses")

    # Video Information
    video_file = models.FileField(upload_to="videos/%Y/%m/%d/")
    video_filename = models.CharField(max_length=255)
    video_duration = models.FloatField(help_text="Duration in seconds")
    game_detected = models.CharField(max_length=20, choices=SupportedGame.choices)

    # Analysis Context
    analysis_context = models.CharField(
        max_length=30,
        choices=[
            ("my_gameplay", "My Gameplay"),
            ("studying_opponent", "Studying Opponent"),
            ("learning_from_pros", "Learning from Pros"),
            ("tournament_prep", "Tournament Preparation"),
        ],
        default="my_gameplay",
    )
    opponent_username = models.CharField(max_length=100, blank=True, null=True)

    # Analysis Results
    total_situations_detected = models.PositiveIntegerField(default=0)
    key_moments_count = models.PositiveIntegerField(default=0)
    analysis_confidence = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )

    # Performance Analysis
    overall_performance_score = models.FloatField(
        null=True, blank=True, help_text="Average performance score"
    )
    performance_tier_distribution = models.JSONField(
        default=dict, help_text="Distribution across 7 tiers"
    )

    # Hardware Performance
    hardware_tier_used = models.CharField(max_length=20, choices=HardwareTier.choices)
    processing_time = models.FloatField(help_text="Processing time in seconds")
    memory_usage_mb = models.FloatField(null=True, blank=True)

    # Cross-Game Intelligence
    universal_concepts_detected = models.ManyToManyField(UniversalConcept, blank=True)
    cross_game_insights = models.JSONField(default=dict, help_text="Cross-game strategy insights")

    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
            ("queued", "Queued"),
        ],
        default="queued",
    )
    error_message = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.video_filename} - {self.get_game_detected_display()} ({self.get_status_display()})"


class SituationalClip(models.Model):
    """Individual situational clips extracted from video analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_analysis = models.ForeignKey(
        VideoAnalysis, on_delete=models.CASCADE, related_name="clips"
    )

    # Clip Information
    start_time = models.FloatField(help_text="Start time in seconds")
    end_time = models.FloatField(help_text="End time in seconds")
    clip_file = models.FileField(upload_to="clips/%Y/%m/%d/", null=True, blank=True)

    # Game Situation
    down = models.PositiveIntegerField(
        null=True, blank=True, validators=[MinValueValidator(1), MaxValueValidator(4)]
    )
    distance = models.PositiveIntegerField(null=True, blank=True)
    yard_line = models.CharField(max_length=20, blank=True, null=True)
    score_differential = models.IntegerField(null=True, blank=True)
    time_remaining = models.CharField(max_length=10, blank=True, null=True)

    # Strategic Context
    situation_type = models.CharField(
        max_length=50,
        choices=[
            ("third_and_long", "Third & Long"),
            ("third_and_short", "Third & Short"),
            ("fourth_down", "Fourth Down"),
            ("red_zone", "Red Zone"),
            ("goal_line", "Goal Line"),
            ("two_minute_warning", "Two-Minute Warning"),
            ("close_game", "Close Game"),
            ("no_huddle_offense", "No Huddle"),
            ("commercial_break", "Commercial Break"),
            ("turnover_interception", "Interception"),
            ("turnover_fumble", "Fumble"),
        ],
    )

    # Performance Analysis
    performance_tier = models.CharField(
        max_length=20, choices=PerformanceTier.choices, null=True, blank=True
    )
    performance_score = models.FloatField(
        null=True, blank=True, validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    yards_gained = models.IntegerField(null=True, blank=True)

    # Professional Benchmarking
    professional_success_rate = models.FloatField(
        null=True, blank=True, help_text="Pro success rate for this situation"
    )
    user_vs_pro_comparison = models.JSONField(
        default=dict, help_text="User performance vs professional benchmarks"
    )

    # User Actions
    user_tagged = models.BooleanField(default=False)
    user_tags = models.JSONField(default=list, help_text="User-applied tags")
    bookmarked = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)

    # Cross-Game Intelligence
    universal_concepts = models.ManyToManyField(UniversalConcept, blank=True)
    transferable_to_games = models.JSONField(
        default=list, help_text="Games this strategy transfers to"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["video_analysis", "start_time"]

    def __str__(self):
        return f"{self.situation_type} at {self.start_time}s"


class GamePlan(models.Model):
    """Strategic gameplans with cross-game intelligence support."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(SpygateUser, on_delete=models.CASCADE, related_name="gameplans")

    # Gameplan Information
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    primary_game = models.CharField(max_length=20, choices=SupportedGame.choices)
    compatible_games = models.JSONField(
        default=list, help_text="Games this gameplan is compatible with"
    )

    # Opponent Information (for tournament prep)
    opponent_username = models.CharField(max_length=100, blank=True, null=True)
    opponent_analysis = models.JSONField(default=dict, help_text="Opponent tendencies and patterns")

    # Strategy Content
    situations_covered = models.ManyToManyField(SituationalClip, blank=True)
    universal_concepts = models.ManyToManyField(UniversalConcept, blank=True)
    custom_strategies = models.JSONField(default=dict, help_text="Custom strategy definitions")

    # Cross-Game Intelligence
    source_analyses = models.ManyToManyField(
        VideoAnalysis, blank=True, help_text="Source video analyses"
    )
    cross_game_adaptations = models.JSONField(default=dict, help_text="Game-specific adaptations")

    # Effectiveness Tracking
    times_used = models.PositiveIntegerField(default=0)
    success_rate = models.FloatField(null=True, blank=True)
    last_used = models.DateTimeField(null=True, blank=True)

    # Sharing
    is_public = models.BooleanField(default=False)
    shared_count = models.PositiveIntegerField(default=0)
    likes_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        opponent_part = f" vs {self.opponent_username}" if self.opponent_username else ""
        return f"{self.name}{opponent_part} ({self.get_primary_game_display()})"


class CommunityStrategy(models.Model):
    """Community-shared strategies with professional analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    creator = models.ForeignKey(
        SpygateUser, on_delete=models.CASCADE, related_name="shared_strategies"
    )

    # Strategy Information
    title = models.CharField(max_length=200)
    description = models.TextField()
    primary_game = models.CharField(max_length=20, choices=SupportedGame.choices)
    universal_concept = models.ForeignKey(
        UniversalConcept, on_delete=models.CASCADE, null=True, blank=True
    )

    # Professional Context
    source_pro_player = models.CharField(
        max_length=100, blank=True, null=True, help_text="Professional player who used this"
    )
    pro_tournament_context = models.CharField(max_length=200, blank=True, null=True)
    effectiveness_rating = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(5.0)],
        help_text="Community effectiveness rating (0-5 stars)",
    )

    # Content
    strategy_data = models.JSONField(help_text="Detailed strategy implementation")
    supporting_clips = models.ManyToManyField(SituationalClip, blank=True)
    cross_game_implementations = models.JSONField(
        default=dict, help_text="Implementations across games"
    )

    # Community Engagement
    views_count = models.PositiveIntegerField(default=0)
    likes_count = models.PositiveIntegerField(default=0)
    downloads_count = models.PositiveIntegerField(default=0)
    comments_count = models.PositiveIntegerField(default=0)

    # Moderation
    is_verified = models.BooleanField(default=False, help_text="Verified by professional players")
    is_featured = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} by {self.creator.username}"


class StrategyComment(models.Model):
    """Comments on community strategies."""

    strategy = models.ForeignKey(
        CommunityStrategy, on_delete=models.CASCADE, related_name="comments"
    )
    user = models.ForeignKey(SpygateUser, on_delete=models.CASCADE)
    content = models.TextField()

    # Engagement
    likes_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Comment by {self.user.username} on {self.strategy.title}"


class PerformanceTracking(models.Model):
    """Long-term performance tracking across sessions and games."""

    user = models.ForeignKey(
        SpygateUser, on_delete=models.CASCADE, related_name="performance_tracking"
    )

    # Session Information
    session_date = models.DateField()
    game = models.CharField(max_length=20, choices=SupportedGame.choices)
    session_type = models.CharField(
        max_length=30,
        choices=[
            ("practice", "Practice"),
            ("ranked", "Ranked Match"),
            ("tournament", "Tournament"),
            ("casual", "Casual"),
        ],
    )

    # Performance Metrics
    total_plays_analyzed = models.PositiveIntegerField()
    average_performance_score = models.FloatField()
    tier_distribution = models.JSONField(
        default=dict, help_text="Distribution across 7 performance tiers"
    )

    # Situational Performance
    third_down_conversion_rate = models.FloatField(null=True, blank=True)
    red_zone_efficiency = models.FloatField(null=True, blank=True)
    two_minute_drill_success = models.FloatField(null=True, blank=True)

    # Professional Comparison
    vs_pro_benchmarks = models.JSONField(
        default=dict, help_text="Performance vs professional benchmarks"
    )
    improvement_areas = models.JSONField(default=list, help_text="Identified improvement areas")

    # Cross-Game Intelligence
    transferable_skills = models.JSONField(
        default=dict, help_text="Skills that transfer between games"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-session_date"]
        unique_together = ["user", "session_date", "game", "session_type"]

    def __str__(self):
        return f"{self.user.username} - {self.session_date} ({self.get_game_display()})"


class TournamentPrep(models.Model):
    """Tournament preparation with multi-opponent analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(SpygateUser, on_delete=models.CASCADE, related_name="tournament_preps")

    # Tournament Information
    tournament_name = models.CharField(max_length=200)
    tournament_date = models.DateField()
    game = models.CharField(max_length=20, choices=SupportedGame.choices)

    # Opponents
    primary_opponent = models.CharField(max_length=100, blank=True, null=True)
    potential_opponents = models.JSONField(default=list, help_text="List of potential opponents")
    opponent_analyses = models.JSONField(default=dict, help_text="Analysis for each opponent")

    # Preparation Strategy
    gameplans = models.ManyToManyField(GamePlan, blank=True)
    practice_sessions = models.JSONField(default=list, help_text="Planned practice sessions")
    key_strategies = models.JSONField(default=dict, help_text="Key strategies to practice")

    # Cross-Game Preparation
    strategy_sources = models.JSONField(
        default=dict, help_text="Strategies adapted from other games"
    )
    universal_concepts_to_practice = models.ManyToManyField(UniversalConcept, blank=True)

    # Progress Tracking
    preparation_progress = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    confidence_level = models.FloatField(
        default=0.0, validators=[MinValueValidator(0.0), MaxValueValidator(10.0)]
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["tournament_date"]

    def __str__(self):
        return f"{self.tournament_name} - {self.tournament_date} ({self.get_game_display()})"


class APIUsageLog(models.Model):
    """Log API usage for performance monitoring and billing."""

    user = models.ForeignKey(
        SpygateUser, on_delete=models.CASCADE, related_name="api_usage", null=True, blank=True
    )

    # Request Information
    endpoint = models.CharField(max_length=200)
    method = models.CharField(max_length=10)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True, null=True)

    # Processing Information
    processing_time = models.FloatField(help_text="Processing time in seconds")
    memory_usage_mb = models.FloatField(null=True, blank=True)
    video_duration = models.FloatField(
        null=True, blank=True, help_text="Duration of processed video"
    )

    # Response Information
    status_code = models.PositiveIntegerField()
    response_size_bytes = models.PositiveIntegerField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        user_part = f"{self.user.username} - " if self.user else ""
        return f"{user_part}{self.method} {self.endpoint} ({self.status_code})"
