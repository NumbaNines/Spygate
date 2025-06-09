"""
URL configuration for SpygateAI Django API

Maps our REST API endpoints to their corresponding view functions.
"""

from django.urls import path

from . import views

app_name = "api"

urlpatterns = [
    # Engine status and health
    path("engine/status/", views.engine_status, name="engine_status"),
    path("health/", views.health_check, name="health_check"),
    path("info/", views.api_info, name="api_info"),
    # Video analysis endpoints
    path("analyze/video/", views.analyze_video, name="analyze_video"),
    path("detect/hud/", views.detect_hud, name="detect_hud"),
    # Tournament preparation
    path("tournament/prepare/", views.tournament_prepare, name="tournament_prepare"),
    # Situational library
    path("library/build/", views.build_situational_library, name="build_situational_library"),
    # Hardware optimization
    path("hardware/optimization/", views.hardware_optimization, name="hardware_optimization"),
]
