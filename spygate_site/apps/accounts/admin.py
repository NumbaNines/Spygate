from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _

from .models import LoginHistory, User, UserProfile


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ("username", "email", "first_name", "last_name", "is_staff", "is_active")
    list_filter = ("is_staff", "is_active", "groups")
    search_fields = ("username", "first_name", "last_name", "email")
    ordering = ("username",)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "subscription_type", "download_count", "last_download", "storage_used")
    list_filter = ("subscription_type", "notify_updates")
    search_fields = ("user__username", "user__email")
    ordering = ("user__username",)


@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "timestamp", "ip_address", "user_agent", "success")
    list_filter = ("success", "timestamp")
    search_fields = ("user__username", "ip_address")
    ordering = ("-timestamp",)
    readonly_fields = ("user", "timestamp", "ip_address", "user_agent", "success")
