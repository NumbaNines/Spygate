"""
Error tracking configuration using Sentry for Spygate application.
"""

import logging
import os
from typing import Optional

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

logger = logging.getLogger(__name__)


def init_error_tracking(dsn: Optional[str] = None) -> None:
    """
    Initialize Sentry error tracking.

    Args:
        dsn: Optional Sentry DSN. If not provided, will try to get from environment variable.
    """
    try:
        # Get DSN from environment if not provided
        dsn = dsn or os.getenv("SENTRY_DSN")

        if not dsn:
            logger.warning("Sentry DSN not provided. Error tracking disabled.")
            return

        # Initialize Sentry with integrations
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=1.0,  # Capture 100% of transactions for performance monitoring
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,  # Minimum log level to capture
                    event_level=logging.ERROR,  # Send errors as events
                ),
                SqlalchemyIntegration(),  # Track database operations
            ],
            # Configure environment
            environment=os.getenv("SPYGATE_ENV", "development"),
            # Add release version if available
            release=os.getenv("SPYGATE_VERSION", "0.1.0"),
            # Enable performance monitoring
            enable_tracing=True,
            # Set custom tags
            default_tags={"app": "spygate", "component": "desktop"},
        )

        logger.info("Sentry error tracking initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)
        # Don't raise the exception - we want the app to continue even if error tracking fails


def capture_exception(error: Exception, context: dict = None) -> None:
    """
    Capture an exception with optional context data.

    Args:
        error: The exception to capture
        context: Optional dictionary of additional context data
    """
    if context:
        with sentry_sdk.configure_scope() as scope:
            for key, value in context.items():
                scope.set_extra(key, value)

    sentry_sdk.capture_exception(error)
