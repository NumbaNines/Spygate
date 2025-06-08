"""
Error handling utilities for video import and processing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoImportError(Exception):
    """Base exception for video import errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(VideoImportError):
    """Error during video file validation."""

    pass


class StorageError(VideoImportError):
    """Error during file storage operations."""

    pass


class DatabaseError(VideoImportError):
    """Error during database operations."""

    pass


class PlayerError(VideoImportError):
    """Error related to player identification."""

    pass


def handle_import_error(
    error: Exception, file_path: Optional[str] = None
) -> Tuple[str, str]:
    """
    Handle video import errors and return user-friendly messages.

    Args:
        error: The exception that occurred
        file_path: Optional path to the video file

    Returns:
        Tuple[str, str]: Title and detailed message for the error
    """
    file_name = os.path.basename(file_path) if file_path else "video file"

    if isinstance(error, ValidationError):
        title = "Invalid Video File"
        message = f"The file '{file_name}' could not be imported:\n{str(error)}"
        if error.details.get("codec"):
            message += f"\nUnsupported codec: {error.details['codec']}"

    elif isinstance(error, StorageError):
        title = "Storage Error"
        message = f"Failed to store '{file_name}':\n{str(error)}"
        if error.details.get("space_needed"):
            message += f"\nRequired space: {error.details['space_needed']} MB"

    elif isinstance(error, DatabaseError):
        title = "Database Error"
        message = f"Failed to save '{file_name}' to database:\n{str(error)}"

    elif isinstance(error, PlayerError):
        title = "Player Identification Error"
        message = str(error)

    elif isinstance(error, FileNotFoundError):
        title = "File Not Found"
        message = f"The file '{file_name}' was not found or was moved during import."

    elif isinstance(error, PermissionError):
        title = "Permission Denied"
        message = f"Access denied while processing '{file_name}'. Please check file permissions."

    else:
        title = "Import Error"
        message = (
            f"An unexpected error occurred while importing '{file_name}':\n{str(error)}"
        )

    # Log the error with full details
    logger.error(
        f"Video import error: {title}\nFile: {file_path}\nError: {str(error)}",
        exc_info=True,
    )

    return title, message


def cleanup_failed_import(
    storage_path: Optional[Path] = None, video_id: Optional[int] = None
) -> None:
    """
    Clean up resources after a failed import.

    Args:
        storage_path: Optional path to the stored video file
        video_id: Optional ID of the video in the database
    """
    try:
        if storage_path and storage_path.exists():
            storage_path.unlink()
            logger.info(f"Cleaned up failed import file: {storage_path}")

        # Database cleanup will be handled by the VideoManager

    except Exception as e:
        logger.error(f"Error during import cleanup: {e}", exc_info=True)
