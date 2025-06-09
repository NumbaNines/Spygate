"""
Django REST API Views for SpygateAI Integration

These views expose our proven SpygateAI engine functionality via REST API endpoints,
maintaining full performance while adding web accessibility.
"""

import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from .services import ensure_service_initialized, get_spygate_service

logger = logging.getLogger(__name__)


@api_view(["GET"])
def engine_status(request):
    """
    Get SpygateAI engine status and health check.

    GET /api/engine/status/
    Returns engine initialization status, hardware info, and performance metrics.
    """
    try:
        service = get_spygate_service()
        status_info = service.get_engine_status()

        return Response(
            {
                "success": True,
                "data": status_info,
                "message": "Engine status retrieved successfully",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "Failed to retrieve engine status"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def analyze_video(request):
    """
    Analyze a video file using SpygateAI engine.

    POST /api/analyze/video/

    Expected form data:
    - video_file: Video file to analyze
    - context: Analysis context (optional, default: 'web_upload')
    - confidence: Confidence threshold (optional, default: 0.8)
    """
    try:
        if not ensure_service_initialized():
            return Response(
                {
                    "success": False,
                    "error": "SpygateAI engine not initialized",
                    "message": "Engine initialization failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Check for video file
        if "video_file" not in request.FILES:
            return Response(
                {
                    "success": False,
                    "error": "No video file provided",
                    "message": "video_file is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        video_file = request.FILES["video_file"]

        # Get analysis options
        analysis_options = {
            "context": request.data.get("context", "web_upload"),
            "confidence": float(request.data.get("confidence", 0.8)),
        }

        # Run analysis
        service = get_spygate_service()
        result = service.analyze_video(video_file, analysis_options)

        if result["success"]:
            return Response(
                {
                    "success": True,
                    "data": result,
                    "message": "Video analysis completed successfully",
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "Video analysis failed"),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "Video analysis failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_hud(request):
    """
    Detect HUD elements in a video using YOLOv8.

    POST /api/detect/hud/

    Expected form data:
    - video_file: Video file to analyze
    - frame_number: Specific frame to analyze (optional)
    """
    try:
        if not ensure_service_initialized():
            return Response(
                {
                    "success": False,
                    "error": "SpygateAI engine not initialized",
                    "message": "Engine initialization failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Check for video file
        if "video_file" not in request.FILES:
            return Response(
                {
                    "success": False,
                    "error": "No video file provided",
                    "message": "video_file is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        video_file = request.FILES["video_file"]
        frame_number = request.data.get("frame_number")

        if frame_number:
            try:
                frame_number = int(frame_number)
            except ValueError:
                return Response(
                    {
                        "success": False,
                        "error": "Invalid frame_number",
                        "message": "frame_number must be an integer",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Run HUD detection
        service = get_spygate_service()
        result = service.detect_hud_elements(video_file, frame_number)

        if result["success"]:
            return Response(
                {
                    "success": True,
                    "data": result,
                    "message": "HUD detection completed successfully",
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "HUD detection failed"),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"HUD detection error: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "HUD detection failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def tournament_prepare(request):
    """
    Prepare tournament analysis for an opponent.

    POST /api/tournament/prepare/

    Expected form data:
    - opponent_name: Name of the opponent
    - video_files: Multiple video files of opponent footage
    """
    try:
        if not ensure_service_initialized():
            return Response(
                {
                    "success": False,
                    "error": "SpygateAI engine not initialized",
                    "message": "Engine initialization failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Check for required data
        opponent_name = request.data.get("opponent_name")
        if not opponent_name:
            return Response(
                {
                    "success": False,
                    "error": "No opponent name provided",
                    "message": "opponent_name is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Get video files
        video_files = request.FILES.getlist("video_files")
        if not video_files:
            return Response(
                {
                    "success": False,
                    "error": "No video files provided",
                    "message": "At least one video file is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Run tournament preparation
        service = get_spygate_service()
        result = service.prepare_tournament_analysis(opponent_name, video_files)

        if result["success"]:
            return Response(
                {
                    "success": True,
                    "data": result,
                    "message": f"Tournament preparation completed for {opponent_name}",
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "Tournament preparation failed"),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Tournament preparation error: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "Tournament preparation failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@parser_classes([JSONParser])
def build_situational_library(request):
    """
    Build situational library for specific game situations.

    POST /api/library/build/

    Expected JSON:
    {
        "situation_type": "3rd_long" | "red_zone" | "2_minute_drill" | etc.
    }
    """
    try:
        if not ensure_service_initialized():
            return Response(
                {
                    "success": False,
                    "error": "SpygateAI engine not initialized",
                    "message": "Engine initialization failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # Check for situation type
        situation_type = request.data.get("situation_type")
        if not situation_type:
            return Response(
                {
                    "success": False,
                    "error": "No situation type provided",
                    "message": "situation_type is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Run situational library building
        service = get_spygate_service()
        result = service.build_situational_library(situation_type)

        if result["success"]:
            return Response(
                {
                    "success": True,
                    "data": result,
                    "message": f"Situational library built for {situation_type}",
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "Situational library building failed"),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Situational library building error: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "Situational library building failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
def hardware_optimization(request):
    """
    Get hardware optimization status.

    GET /api/hardware/optimization/
    Returns current hardware optimization settings and performance metrics.
    """
    try:
        if not ensure_service_initialized():
            return Response(
                {
                    "success": False,
                    "error": "SpygateAI engine not initialized",
                    "message": "Engine initialization failed",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        service = get_spygate_service()
        result = service.get_hardware_optimization_status()

        if result["success"]:
            return Response(
                {
                    "success": True,
                    "data": result,
                    "message": "Hardware optimization status retrieved",
                },
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "message": result.get("message", "Failed to get hardware optimization status"),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Hardware optimization status error: {e}")
        return Response(
            {
                "success": False,
                "error": str(e),
                "message": "Failed to get hardware optimization status",
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
def health_check(request):
    """
    Simple health check endpoint.

    GET /api/health/
    Returns basic service health status.
    """
    try:
        is_initialized = ensure_service_initialized()

        return Response(
            {
                "success": True,
                "data": {
                    "service": "SpygateAI Django API",
                    "status": "healthy" if is_initialized else "degraded",
                    "engine_initialized": is_initialized,
                    "version": "1.0.0",
                },
                "message": "Health check completed",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return Response(
            {"success": False, "error": str(e), "message": "Health check failed"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Additional utility views for development and testing


@api_view(["GET"])
def api_info(request):
    """
    Get API information and available endpoints.

    GET /api/info/
    Returns information about available API endpoints.
    """
    api_endpoints = {
        "engine": {
            "GET /api/engine/status/": "Get engine status and health check",
        },
        "analysis": {
            "POST /api/analyze/video/": "Analyze video using SpygateAI engine",
            "POST /api/detect/hud/": "Detect HUD elements using YOLOv8",
        },
        "tournament": {
            "POST /api/tournament/prepare/": "Prepare tournament analysis for opponent",
        },
        "library": {
            "POST /api/library/build/": "Build situational library",
        },
        "system": {
            "GET /api/hardware/optimization/": "Get hardware optimization status",
            "GET /api/health/": "Basic health check",
            "GET /api/info/": "API information (this endpoint)",
        },
    }

    return Response(
        {
            "success": True,
            "data": {
                "service": "SpygateAI Django REST API",
                "version": "1.0.0",
                "description": "REST API for SpygateAI video analysis engine",
                "endpoints": api_endpoints,
                "engine_status": ensure_service_initialized(),
            },
            "message": "API information retrieved",
        },
        status=status.HTTP_200_OK,
    )
