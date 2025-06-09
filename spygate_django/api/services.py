"""
SpygateAI Django Integration Service

This service provides the bridge between Django and our SpygateAI engine,
handling video analysis, HUD detection, and tournament preparation workflows.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile

from spygate.core.hardware import HardwareDetector

# Import our SpygateAI engine
from spygate.core.spygate_engine import SpygateAI
from spygate.ml.yolov8_model import EnhancedYOLOv8

logger = logging.getLogger(__name__)


class SpygateService:
    """
    Main service class that integrates SpygateAI engine with Django.

    This class maintains our proven SpygateAI engine performance while
    providing a Django-compatible interface for web applications.
    """

    def __init__(self):
        self.engine = None
        self.hardware = None
        self.yolo_model = None
        self._initialized = False

    def initialize(self) -> dict[str, Any]:
        """Initialize the SpygateAI engine for Django usage."""
        try:
            # Initialize hardware detection
            self.hardware = HardwareDetector()

            # Initialize our proven SpygateAI engine
            project_root = settings.SPYGATE_ENGINE_CONFIG["PROJECT_ROOT"]
            self.engine = SpygateAI(project_root=project_root)

            # Get engine status
            status = self.engine.get_system_status()

            self._initialized = True

            logger.info(f"SpygateAI Django service initialized successfully")
            logger.info(f"Engine status: {status['status']}")
            logger.info(f"Hardware tier: {self.hardware.tier.name}")

            return {
                "success": True,
                "engine_status": status,
                "hardware_tier": self.hardware.tier.name,
                "systems_ready": status["systems_count"],
                "message": "SpygateAI engine initialized successfully",
            }

        except Exception as e:
            logger.error(f"Failed to initialize SpygateAI service: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to initialize SpygateAI engine",
            }

    def get_engine_status(self) -> dict[str, Any]:
        """Get current engine status."""
        if not self._initialized:
            return {"initialized": False, "message": "Engine not initialized"}

        try:
            status = self.engine.get_system_status()
            return {
                "initialized": True,
                "engine_status": status,
                "hardware_info": self.hardware.get_system_info() if self.hardware else {},
                "performance_optimizations": self.engine.get_optimization_status(),
            }
        except Exception as e:
            logger.error(f"Error getting engine status: {e}")
            return {
                "initialized": True,
                "error": str(e),
                "message": "Error retrieving engine status",
            }

    def analyze_video(
        self, video_file: UploadedFile, analysis_options: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Analyze a video file using the SpygateAI engine.

        Args:
            video_file: Django uploaded file object
            analysis_options: Optional analysis configuration

        Returns:
            Dict containing analysis results
        """
        if not self._initialized:
            return {"success": False, "error": "Engine not initialized"}

        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                for chunk in video_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            try:
                # Run analysis using our proven engine
                analysis_result = self.engine.analyze_any_footage(
                    video_file=tmp_path,
                    context=analysis_options.get("context", "web_upload"),
                    auto_export=analysis_options.get("auto_export", False),
                )

                return {
                    "success": True,
                    "analysis": analysis_result,
                    "video_name": video_file.name,
                    "video_size": video_file.size,
                    "message": "Video analysis completed successfully",
                }

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {"success": False, "error": str(e), "message": "Video analysis failed"}

    def detect_hud_elements(
        self, video_file: UploadedFile, frame_number: int = None
    ) -> dict[str, Any]:
        """
        Detect HUD elements in a video frame using YOLOv8.

        Args:
            video_file: Django uploaded file object
            frame_number: Specific frame to analyze (optional)

        Returns:
            Dict containing HUD detection results
        """
        if not self._initialized:
            return {"success": False, "error": "Engine not initialized"}

        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                for chunk in video_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name

            try:
                # Use the quick_analysis method which can focus on specific frames
                if frame_number is not None:
                    # For specific frame analysis, use quick analysis
                    hud_result = self.engine.quick_analysis(
                        video_file=tmp_path, situation_filter="hud_analysis"
                    )
                else:
                    # For general HUD analysis, use full analysis
                    hud_result = self.engine.analyze_any_footage(
                        video_file=tmp_path,
                        context="hud_detection",
                        auto_export=False,
                    )

                return {
                    "success": True,
                    "hud_elements": hud_result,
                    "frame_number": frame_number,
                    "video_name": video_file.name,
                    "message": "HUD detection completed successfully",
                }

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"HUD detection failed: {e}")
            return {"success": False, "error": str(e), "message": "HUD detection failed"}

    def prepare_tournament_analysis(
        self, opponent_name: str, video_files: list[UploadedFile]
    ) -> dict[str, Any]:
        """
        Prepare tournament analysis using our proven tournament preparation workflow.

        Args:
            opponent_name: Name of the opponent
            video_files: List of opponent footage files

        Returns:
            Dict containing tournament preparation results
        """
        if not self._initialized:
            return {"success": False, "error": "Engine not initialized"}

        try:
            # Save uploaded files to temporary locations
            temp_files = []

            for video_file in video_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    for chunk in video_file.chunks():
                        tmp_file.write(chunk)
                    temp_files.append(tmp_file.name)

            try:
                # Run tournament preparation using the correct method
                tournament_result = self.engine.prepare_for_tournament_match(
                    opponent_username=opponent_name,
                    opponent_footage_files=temp_files,
                    tournament_type="web_analysis",
                    game_version="madden_25",
                )

                return {
                    "success": True,
                    "tournament_analysis": tournament_result,
                    "opponent_name": opponent_name,
                    "videos_analyzed": len(video_files),
                    "message": f"Tournament preparation completed for {opponent_name}",
                }

            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Tournament preparation failed: {e}")
            return {"success": False, "error": str(e), "message": "Tournament preparation failed"}

    def build_situational_library(
        self, situation_type: str, video_files: list[UploadedFile] = None
    ) -> dict[str, Any]:
        """
        Build situational library using our proven situational analysis.

        Args:
            situation_type: Type of situation to analyze (e.g., "3rd_long", "red_zone")
            video_files: Optional additional footage files

        Returns:
            Dict containing situational library results
        """
        if not self._initialized:
            return {"success": False, "error": "Engine not initialized"}

        try:
            # Use our proven situational library builder
            library_result = self.engine.build_situational_library(situation_type)

            return {
                "success": True,
                "situational_library": library_result,
                "situation_type": situation_type,
                "message": f"Situational library built for {situation_type}",
            }

        except Exception as e:
            logger.error(f"Situational library building failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Situational library building failed",
            }

    def get_hardware_optimization_status(self) -> dict[str, Any]:
        """Get hardware optimization status."""
        if not self._initialized:
            return {"initialized": False}

        try:
            optimization_status = self.engine.optimize_for_hardware()

            return {
                "success": True,
                "optimization_status": optimization_status,
                "hardware_tier": self.hardware.tier.name,
                "gpu_available": self.hardware.has_cuda,
                "message": "Hardware optimization status retrieved",
            }

        except Exception as e:
            logger.error(f"Error getting hardware optimization status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get hardware optimization status",
            }


# Global service instance
_spygate_service = None


def get_spygate_service() -> SpygateService:
    """Get or create the global SpygateAI service instance."""
    global _spygate_service

    if _spygate_service is None:
        _spygate_service = SpygateService()
        result = _spygate_service.initialize()

        if not result["success"]:
            logger.error(f"Failed to initialize SpygateAI service: {result}")

    return _spygate_service


def ensure_service_initialized() -> bool:
    """Ensure the SpygateAI service is initialized."""
    service = get_spygate_service()
    return service._initialized
