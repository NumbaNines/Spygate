"""
Spygate - Madden NFL 25 Game Analysis Tool
Main application entry point
"""

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from spygate.database.config import init_db
from spygate.gui.components.main_window import MainWindow
from spygate.services.analysis_service import AnalysisService
from spygate.services.video_service import VideoService
from spygate.utils.error_tracking import init_error_tracking
from spygate.utils.logging import setup_logging


def initialize_services():
    """Initialize application services."""
    video_service = VideoService()
    analysis_service = AnalysisService(video_service)
    return video_service, analysis_service


def main():
    """Application entry point."""
    # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(log_dir / "spygate.log")
    logger = logging.getLogger(__name__)
    logger.info("Starting Spygate application")

    try:
        # Initialize error tracking
        init_error_tracking()
        logger.info("Error tracking initialized")

        # Initialize the database
        init_db()
        logger.info("Database initialized successfully")

        # Initialize services
        video_service, analysis_service = initialize_services()
        logger.info("Services initialized successfully")

        # Create Qt application
        app = QApplication(sys.argv)

        # Create and show main window
        window = MainWindow(
            video_service=video_service, analysis_service=analysis_service
        )
        window.show()
        logger.info("Main window displayed")

        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Fatal error during application startup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
