import os

from apps.accounts.models import UserProfile
from apps.analysis.models import Formation, GameAnalysis, Play, Situation
from apps.analysis.serializers import (
    FormationSerializer,
    GameAnalysisSerializer,
    PlaySerializer,
    SituationSerializer,
)
from apps.analysis.tasks import analyze_video
from apps.downloads.models import Download, Release
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .serializers import DownloadSerializer, ReleaseSerializer


class ReleaseViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing software releases.
    """

    queryset = Release.objects.filter(is_active=True)
    serializer_class = ReleaseSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=["post"])
    def download(self, request, pk=None):
        """Handle release download with user quota checking."""
        release = self.get_object()
        profile = request.user.profile

        # Check user's download quota
        if not profile.can_download():
            return Response({"error": "Download quota exceeded"}, status=status.HTTP_403_FORBIDDEN)

        # Check file size against storage quota
        if not profile.has_storage_space(release.installer.size):
            return Response(
                {"error": "Insufficient storage space"}, status=status.HTTP_403_FORBIDDEN
            )

        # Create download record
        download = Download.objects.create(
            user=request.user, release=release, file_size=release.installer.size
        )

        # Update user metrics
        profile.download_count += 1
        profile.storage_used += release.installer.size
        profile.save()

        return Response(DownloadSerializer(download).data)

    @action(detail=True, methods=["get"])
    def analyze(self, request, pk=None):
        """
        Analyze video for game detection and initial processing.
        Integrates with core Python analysis engine.
        """
        release = self.get_object()

        try:
            # Here we'll integrate with the core Python analysis engine
            # This is a placeholder for the actual integration
            analysis_result = {
                "game_version": "madden_25",
                "hardware_tier": "standard",
                "estimated_processing_time": "120",
                "supported_features": ["hud_analysis", "formation_detection", "play_recognition"],
            }

            return Response(analysis_result)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GameAnalysisViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing game analysis.
    """

    serializer_class = GameAnalysisSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return GameAnalysis.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        analysis = serializer.save(user=self.request.user)
        # Start async analysis task
        analyze_video.delay(analysis.id)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        """Get the current status of the analysis."""
        analysis = self.get_object()
        return Response(
            {
                "status": analysis.processing_status,
                "completed_at": analysis.completed_at,
                "error_message": analysis.error_message,
            }
        )


class FormationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing detected formations.
    """

    serializer_class = FormationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Formation.objects.filter(analysis__user=self.request.user)


class PlayViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing detected plays.
    """

    serializer_class = PlaySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Play.objects.filter(analysis__user=self.request.user)

    @action(detail=False, methods=["get"])
    def by_formation(self, request):
        """Get plays grouped by formation."""
        formation_id = request.query_params.get("formation_id")
        if not formation_id:
            return Response(
                {"error": "formation_id parameter is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        plays = self.get_queryset().filter(formation_id=formation_id)
        serializer = self.get_serializer(plays, many=True)
        return Response(serializer.data)


class SituationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing game situations.
    """

    serializer_class = SituationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Situation.objects.filter(analysis__user=self.request.user)

    @action(detail=False, methods=["get"])
    def by_down(self, request):
        """Get situations filtered by down."""
        down = request.query_params.get("down")
        if not down:
            return Response(
                {"error": "down parameter is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        situations = self.get_queryset().filter(down=down)
        serializer = self.get_serializer(situations, many=True)
        return Response(serializer.data)
