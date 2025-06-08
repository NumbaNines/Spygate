from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from .models import ErrorReport, FeatureUsage, PerformanceMetric, UsageSession, UserMetrics


@admin.register(UserMetrics)
class UserMetricsAdmin(admin.ModelAdmin):
    list_display = ("user", "last_active", "total_downloads", "total_storage_used")
    list_filter = ("last_active",)
    search_fields = ("user__username", "user__email")
    ordering = ("-last_active",)
    readonly_fields = ("last_active", "total_downloads", "total_storage_used")


@admin.register(FeatureUsage)
class FeatureUsageAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "feature_name",
        "feature_category",
        "usage_count",
        "last_used",
        "success_rate",
    )
    list_filter = ("feature_category", "last_used")
    search_fields = ("user__username", "feature_name")
    ordering = ("-last_used",)
    readonly_fields = ("usage_count", "last_used", "success_rate", "average_duration")


@admin.register(ErrorReport)
class ErrorReportAdmin(admin.ModelAdmin):
    list_display = ("error_level", "user", "release", "timestamp", "resolved")
    list_filter = ("error_level", "resolved", "timestamp")
    search_fields = ("user__username", "error_message", "stack_trace")
    ordering = ("-timestamp",)
    readonly_fields = (
        "user",
        "release",
        "error_level",
        "error_message",
        "stack_trace",
        "system_info",
        "timestamp",
    )


@admin.register(PerformanceMetric)
class PerformanceMetricAdmin(admin.ModelAdmin):
    list_display = ("user", "metric_type", "value", "timestamp")
    list_filter = ("metric_type", "timestamp")
    search_fields = ("user__username",)
    ordering = ("-timestamp",)
    readonly_fields = ("user", "metric_type", "value", "timestamp", "system_info")


@admin.register(UsageSession)
class UsageSessionAdmin(admin.ModelAdmin):
    list_display = ("user", "start_time", "end_time", "duration", "ip_address")
    list_filter = ("start_time",)
    search_fields = ("user__username", "ip_address")
    ordering = ("-start_time",)
    readonly_fields = (
        "user",
        "start_time",
        "end_time",
        "duration",
        "ip_address",
        "user_agent",
        "features_used",
        "session_data",
    )
