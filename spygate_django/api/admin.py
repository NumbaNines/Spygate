"""
Django admin configuration for SpygateAI models.

Provides comprehensive admin interfaces for managing users, video analyses,
strategies, and community features.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html

from .models import (
    APIUsageLog,
    CommunityStrategy,
    GamePlan,
    GameSpecificImplementation,
    PerformanceTracking,
    SituationalClip,
    SpygateUser,
    StrategyComment,
    TournamentPrep,
    UniversalConcept,
    VideoAnalysis,
)


@admin.register(SpygateUser)
class SpygateUserAdmin(UserAdmin):
    """Enhanced admin for SpygateAI users."""

    list_display = (
        "username",
        "email",
        "skill_level",
        "primary_game",
        "subscription_tier",
        "total_videos_analyzed",
        "is_active",
        "date_joined",
    )
    list_filter = (
        "skill_level",
        "primary_game",
        "subscription_tier",
        "hardware_tier",
        "enable_cross_game_intelligence",
        "enable_professional_benchmarking",
        "is_active",
        "is_staff",
    )
    search_fields = ("username", "email", "mcs_username", "players_lounge_username")
    readonly_fields = ("date_joined", "last_login", "created_at", "updated_at")

    fieldsets = UserAdmin.fieldsets + (
        (
            "SpygateAI Profile",
            {
                "fields": (
                    "mcs_username",
                    "players_lounge_username",
                    "skill_level",
                    "primary_game",
                    "hardware_tier",
                )
            },
        ),
        (
            "Preferences",
            {"fields": ("enable_cross_game_intelligence", "enable_professional_benchmarking")},
        ),
        (
            "Statistics",
            {
                "fields": (
                    "total_videos_analyzed",
                    "total_strategies_created",
                    "total_gameplans_shared",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Subscription",
            {"fields": ("subscription_tier", "subscription_expires"), "classes": ("collapse",)},
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related()


@admin.register(UniversalConcept)
class UniversalConceptAdmin(admin.ModelAdmin):
    """Admin for universal football concepts."""

    list_display = ("name", "category", "effectiveness_rating", "created_by", "created_at")
    list_filter = ("category", "effectiveness_rating", "created_at")
    search_fields = ("name", "core_principle")
    readonly_fields = ("created_at",)

    fieldsets = (
        (None, {"fields": ("name", "category", "core_principle", "effectiveness_rating")}),
        ("Metadata", {"fields": ("created_by", "created_at"), "classes": ("collapse",)}),
    )


class GameSpecificImplementationInline(admin.TabularInline):
    """Inline admin for game-specific implementations."""

    model = GameSpecificImplementation
    extra = 1
    readonly_fields = ("created_at", "updated_at")


@admin.register(GameSpecificImplementation)
class GameSpecificImplementationAdmin(admin.ModelAdmin):
    """Admin for game-specific implementations."""

    list_display = (
        "implementation_name",
        "game",
        "universal_concept",
        "success_rate",
        "pro_success_rate",
        "updated_at",
    )
    list_filter = ("game", "universal_concept__category", "updated_at")
    search_fields = ("implementation_name", "formation_code", "playbook_location")
    readonly_fields = ("created_at", "updated_at")


# Add inline to UniversalConcept admin
UniversalConceptAdmin.inlines = [GameSpecificImplementationInline]


class SituationalClipInline(admin.TabularInline):
    """Inline admin for situational clips."""

    model = SituationalClip
    extra = 0
    readonly_fields = ("created_at",)
    fields = (
        "start_time",
        "end_time",
        "situation_type",
        "performance_tier",
        "performance_score",
        "user_tagged",
        "bookmarked",
    )


@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    """Admin for video analyses."""

    list_display = (
        "video_filename",
        "user",
        "game_detected",
        "analysis_context",
        "status",
        "analysis_confidence",
        "processing_time",
        "created_at",
    )
    list_filter = (
        "game_detected",
        "analysis_context",
        "status",
        "hardware_tier_used",
        "created_at",
    )
    search_fields = ("video_filename", "user__username", "opponent_username")
    readonly_fields = ("created_at", "completed_at")

    fieldsets = (
        (
            "Video Information",
            {"fields": ("user", "video_file", "video_filename", "video_duration", "game_detected")},
        ),
        ("Analysis Context", {"fields": ("analysis_context", "opponent_username")}),
        (
            "Results",
            {
                "fields": (
                    "total_situations_detected",
                    "key_moments_count",
                    "analysis_confidence",
                    "overall_performance_score",
                    "performance_tier_distribution",
                )
            },
        ),
        (
            "Performance",
            {
                "fields": ("hardware_tier_used", "processing_time", "memory_usage_mb"),
                "classes": ("collapse",),
            },
        ),
        (
            "Cross-Game Intelligence",
            {
                "fields": ("universal_concepts_detected", "cross_game_insights"),
                "classes": ("collapse",),
            },
        ),
        ("Status", {"fields": ("status", "error_message", "created_at", "completed_at")}),
    )

    inlines = [SituationalClipInline]

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")


@admin.register(SituationalClip)
class SituationalClipAdmin(admin.ModelAdmin):
    """Admin for situational clips."""

    list_display = (
        "video_analysis",
        "situation_type",
        "start_time",
        "end_time",
        "performance_tier",
        "performance_score",
        "user_tagged",
        "bookmarked",
    )
    list_filter = (
        "situation_type",
        "performance_tier",
        "user_tagged",
        "bookmarked",
        "down",
        "video_analysis__game_detected",
    )
    search_fields = ("video_analysis__video_filename", "notes", "user_tags")
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Clip Information", {"fields": ("video_analysis", "start_time", "end_time", "clip_file")}),
        (
            "Game Situation",
            {"fields": ("down", "distance", "yard_line", "score_differential", "time_remaining")},
        ),
        ("Strategic Context", {"fields": ("situation_type",)}),
        (
            "Performance Analysis",
            {
                "fields": (
                    "performance_tier",
                    "performance_score",
                    "yards_gained",
                    "professional_success_rate",
                    "user_vs_pro_comparison",
                )
            },
        ),
        ("User Actions", {"fields": ("user_tagged", "user_tags", "bookmarked", "notes")}),
        (
            "Cross-Game Intelligence",
            {"fields": ("universal_concepts", "transferable_to_games"), "classes": ("collapse",)},
        ),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )


@admin.register(GamePlan)
class GamePlanAdmin(admin.ModelAdmin):
    """Admin for game plans."""

    list_display = (
        "name",
        "user",
        "primary_game",
        "opponent_username",
        "times_used",
        "success_rate",
        "is_public",
        "updated_at",
    )
    list_filter = ("primary_game", "is_public", "times_used", "updated_at")
    search_fields = ("name", "description", "user__username", "opponent_username")
    readonly_fields = ("created_at", "updated_at", "last_used")

    fieldsets = (
        (
            "Gameplan Information",
            {"fields": ("user", "name", "description", "primary_game", "compatible_games")},
        ),
        (
            "Opponent Information",
            {"fields": ("opponent_username", "opponent_analysis"), "classes": ("collapse",)},
        ),
        (
            "Strategy Content",
            {"fields": ("situations_covered", "universal_concepts", "custom_strategies")},
        ),
        (
            "Cross-Game Intelligence",
            {"fields": ("source_analyses", "cross_game_adaptations"), "classes": ("collapse",)},
        ),
        ("Effectiveness Tracking", {"fields": ("times_used", "success_rate", "last_used")}),
        ("Sharing", {"fields": ("is_public", "shared_count", "likes_count")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    filter_horizontal = ("situations_covered", "universal_concepts", "source_analyses")


class StrategyCommentInline(admin.TabularInline):
    """Inline admin for strategy comments."""

    model = StrategyComment
    extra = 0
    readonly_fields = ("created_at",)


@admin.register(CommunityStrategy)
class CommunityStrategyAdmin(admin.ModelAdmin):
    """Admin for community strategies."""

    list_display = (
        "title",
        "creator",
        "primary_game",
        "effectiveness_rating",
        "views_count",
        "likes_count",
        "is_verified",
        "is_featured",
        "created_at",
    )
    list_filter = (
        "primary_game",
        "is_verified",
        "is_featured",
        "effectiveness_rating",
        "created_at",
    )
    search_fields = ("title", "description", "creator__username", "source_pro_player")
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (
            "Strategy Information",
            {"fields": ("creator", "title", "description", "primary_game", "universal_concept")},
        ),
        (
            "Professional Context",
            {"fields": ("source_pro_player", "pro_tournament_context", "effectiveness_rating")},
        ),
        (
            "Content",
            {"fields": ("strategy_data", "supporting_clips", "cross_game_implementations")},
        ),
        (
            "Community Engagement",
            {
                "fields": ("views_count", "likes_count", "downloads_count", "comments_count"),
                "classes": ("collapse",),
            },
        ),
        ("Moderation", {"fields": ("is_verified", "is_featured")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    filter_horizontal = ("supporting_clips",)
    inlines = [StrategyCommentInline]


@admin.register(StrategyComment)
class StrategyCommentAdmin(admin.ModelAdmin):
    """Admin for strategy comments."""

    list_display = ("strategy", "user", "content_preview", "likes_count", "created_at")
    list_filter = ("created_at", "likes_count")
    search_fields = ("content", "user__username", "strategy__title")
    readonly_fields = ("created_at",)

    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "Content Preview"


@admin.register(PerformanceTracking)
class PerformanceTrackingAdmin(admin.ModelAdmin):
    """Admin for performance tracking."""

    list_display = (
        "user",
        "session_date",
        "game",
        "session_type",
        "total_plays_analyzed",
        "average_performance_score",
        "created_at",
    )
    list_filter = ("game", "session_type", "session_date", "created_at")
    search_fields = ("user__username",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Session Information", {"fields": ("user", "session_date", "game", "session_type")}),
        (
            "Performance Metrics",
            {"fields": ("total_plays_analyzed", "average_performance_score", "tier_distribution")},
        ),
        (
            "Situational Performance",
            {
                "fields": (
                    "third_down_conversion_rate",
                    "red_zone_efficiency",
                    "two_minute_drill_success",
                )
            },
        ),
        (
            "Professional Comparison",
            {"fields": ("vs_pro_benchmarks", "improvement_areas"), "classes": ("collapse",)},
        ),
        ("Cross-Game Intelligence", {"fields": ("transferable_skills",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at",), "classes": ("collapse",)}),
    )


@admin.register(TournamentPrep)
class TournamentPrepAdmin(admin.ModelAdmin):
    """Admin for tournament preparation."""

    list_display = (
        "tournament_name",
        "user",
        "tournament_date",
        "game",
        "primary_opponent",
        "preparation_progress",
        "confidence_level",
    )
    list_filter = ("game", "tournament_date", "preparation_progress", "confidence_level")
    search_fields = ("tournament_name", "user__username", "primary_opponent")
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        (
            "Tournament Information",
            {"fields": ("user", "tournament_name", "tournament_date", "game")},
        ),
        ("Opponents", {"fields": ("primary_opponent", "potential_opponents", "opponent_analyses")}),
        ("Preparation Strategy", {"fields": ("gameplans", "practice_sessions", "key_strategies")}),
        (
            "Cross-Game Preparation",
            {
                "fields": ("strategy_sources", "universal_concepts_to_practice"),
                "classes": ("collapse",),
            },
        ),
        ("Progress Tracking", {"fields": ("preparation_progress", "confidence_level")}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    filter_horizontal = ("gameplans", "universal_concepts_to_practice")


@admin.register(APIUsageLog)
class APIUsageLogAdmin(admin.ModelAdmin):
    """Admin for API usage logs."""

    list_display = (
        "user",
        "endpoint",
        "method",
        "status_code",
        "processing_time",
        "video_duration",
        "timestamp",
    )
    list_filter = ("method", "status_code", "endpoint", "timestamp")
    search_fields = ("user__username", "endpoint", "ip_address", "error_message")
    readonly_fields = ("timestamp",)

    fieldsets = (
        (
            "Request Information",
            {"fields": ("user", "endpoint", "method", "ip_address", "user_agent")},
        ),
        (
            "Processing Information",
            {"fields": ("processing_time", "memory_usage_mb", "video_duration")},
        ),
        (
            "Response Information",
            {"fields": ("status_code", "response_size_bytes", "error_message")},
        ),
        ("Timestamp", {"fields": ("timestamp",)}),
    )

    def get_queryset(self, request):
        return super().get_queryset(request).select_related("user")


# Customize admin site
admin.site.site_header = "SpygateAI Administration"
admin.site.site_title = "SpygateAI Admin"
admin.site.index_title = "Welcome to SpygateAI Administration"
