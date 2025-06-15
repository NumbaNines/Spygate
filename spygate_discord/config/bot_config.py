#!/usr/bin/env python3
"""
Configuration settings for SpygateAI Discord Bot
"""

import os
from pathlib import Path


class BotConfig:
    """Bot configuration constants and settings"""

    # Bot Settings
    COMMAND_PREFIX = "!spygate "
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    # File Paths
    TEMP_UPLOAD_DIR = Path("temp_discord_uploads")
    LOGS_DIR = Path("logs")

    # Analysis Settings
    DEFAULT_ANALYSIS_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_ANALYSES = 3

    # Discord Settings
    EMBED_COLOR_SUCCESS = 0x00FF00  # Green
    EMBED_COLOR_ERROR = 0xFF0000  # Red
    EMBED_COLOR_INFO = 0x0099FF  # Blue
    EMBED_COLOR_WARNING = 0xFF9900  # Orange

    # Rate Limiting
    ANALYSIS_COOLDOWN = 30  # seconds between analyses per user
    MAX_UPLOADS_PER_HOUR = 10

    # Feature Flags
    ENABLE_AUTO_ANALYSIS = True
    ENABLE_DETAILED_LOGS = True
    ENABLE_DEMO_MODE = not os.getenv("SPYGATE_PRODUCTION", False)

    # Admin Settings
    ADMIN_USER_IDS = []  # Add admin Discord user IDs here
    LOG_CHANNEL_ID = None  # Set to channel ID for bot logs

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_bot_token(cls):
        """Get Discord bot token from environment"""
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN not found in environment variables!")
        return token

    @classmethod
    def is_admin(cls, user_id):
        """Check if user is a bot admin"""
        return user_id in cls.ADMIN_USER_IDS
