"""
URL configuration for spygate_web project.

Enhanced for SpygateAI API endpoints and media file serving.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path
from django.views.decorators.csrf import csrf_exempt


def api_root(request):
    """Root API endpoint with basic information."""
    return JsonResponse(
        {
            "service": "SpygateAI Django API",
            "version": "1.0.0",
            "description": "REST API for SpygateAI video analysis engine",
            "docs": "/api/info/",
            "health": "/api/health/",
            "engine_status": "/api/engine/status/",
            "endpoints": {
                "video_analysis": "/api/analyze/video/",
                "hud_detection": "/api/detect/hud/",
                "tournament_prep": "/api/tournament/prepare/",
                "situational_library": "/api/library/build/",
                "hardware_optimization": "/api/hardware/optimization/",
            },
        }
    )


urlpatterns = [
    # Django admin
    path("admin/", admin.site.urls),
    # API root
    path("api/", api_root, name="api_root"),
    # SpygateAI API endpoints
    path("api/", include("api.urls")),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
