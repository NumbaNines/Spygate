from django.contrib import admin
from django.utils.translation import gettext_lazy as _

from .models import APIKey, APIUsage, Webhook, WebhookDelivery


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "user",
        "permission_level",
        "is_active",
        "created_at",
        "expires_at",
        "is_valid",
    )
    list_filter = ("permission_level", "is_active", "created_at")
    search_fields = ("name", "user__username", "key")
    ordering = ("-created_at",)
    readonly_fields = ("key", "created_at", "last_used")
    fieldsets = (
        (None, {"fields": ("name", "user", "permission_level")}),
        (_("Settings"), {"fields": ("is_active", "expires_at", "allowed_ips")}),
        (_("Metadata"), {"fields": ("key", "created_at", "last_used")}),
    )


@admin.register(APIUsage)
class APIUsageAdmin(admin.ModelAdmin):
    list_display = ("api_key", "endpoint", "method", "status_code", "response_time", "timestamp")
    list_filter = ("method", "status_code", "timestamp")
    search_fields = ("api_key__name", "endpoint", "ip_address")
    ordering = ("-timestamp",)
    readonly_fields = (
        "api_key",
        "endpoint",
        "method",
        "status_code",
        "response_time",
        "timestamp",
        "ip_address",
        "user_agent",
        "request_data",
    )


@admin.register(Webhook)
class WebhookAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "url", "is_active", "last_triggered", "failure_count")
    list_filter = ("is_active", "created_at")
    search_fields = ("name", "user__username", "url")
    ordering = ("name",)
    readonly_fields = ("created_at", "last_triggered", "failure_count")


@admin.register(WebhookDelivery)
class WebhookDeliveryAdmin(admin.ModelAdmin):
    list_display = (
        "webhook",
        "event_type",
        "success",
        "response_status",
        "timestamp",
        "retry_count",
    )
    list_filter = ("event_type", "success", "timestamp")
    search_fields = ("webhook__name", "event_type")
    ordering = ("-timestamp",)
    readonly_fields = (
        "webhook",
        "event_type",
        "payload",
        "response_status",
        "response_body",
        "timestamp",
        "success",
        "retry_count",
    )
