from django.shortcuts import render
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.core.files.storage import default_storage
from django.conf import settings
from drf_spectacular.utils import extend_schema, OpenApiParameter
import os

from apps.analysis.models import (
    GameAnalysis, Formation, Play, Situation,
    UserMetrics, FeatureUsage, PerformanceMetric
)
from apps.analysis.serializers import (
    GameAnalysisSerializer, FormationSerializer, PlaySerializer, SituationSerializer,
    UserMetricsSerializer, FeatureUsageSerializer, PerformanceMetricSerializer
)
from apps.analysis.tasks import analyze_video

class GameAnalysisViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing game analysis.
    """
    serializer_class = GameAnalysisSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['processing_status', 'created_at']
    search_fields = ['title', 'description']
    ordering_fields = ['created_at', 'updated_at', 'completed_at']
    ordering = ['-created_at']

    def get_queryset(self):
        return GameAnalysis.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        analysis = serializer.save(user=self.request.user)
        analyze_video.delay(analysis.id)

    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Reprocess the video analysis."""
        analysis = self.get_object()
        analysis.processing_status = 'pending'
        analysis.save()
        analyze_video.delay(analysis.id)
        return Response({'status': 'reprocessing initiated'})

class FormationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing formations.
    """
    serializer_class = FormationSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['formation_type', 'analysis']
    search_fields = ['name']

    def get_queryset(self):
        return Formation.objects.filter(analysis__user=self.request.user)

class PlayViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing plays.
    """
    serializer_class = PlaySerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['play_type', 'analysis', 'formation']
    search_fields = ['name']

    def get_queryset(self):
        return Play.objects.filter(analysis__user=self.request.user)

class SituationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing situations.
    """
    serializer_class = SituationSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter]
    filterset_fields = ['down', 'is_red_zone', 'analysis']
    search_fields = ['metadata']

    def get_queryset(self):
        return Situation.objects.filter(analysis__user=self.request.user)

class UserMetricsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing user metrics.
    """
    serializer_class = UserMetricsSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserMetrics.objects.filter(user=self.request.user)

class FeatureUsageViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing feature usage statistics.
    """
    serializer_class = FeatureUsageSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['feature_category']

    def get_queryset(self):
        return FeatureUsage.objects.filter(user=self.request.user)

class PerformanceMetricViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing performance metrics.
    """
    serializer_class = PerformanceMetricSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['metric_type']
    ordering_fields = ['timestamp', 'value']
    ordering = ['-timestamp']

    def get_queryset(self):
        return PerformanceMetric.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user) 