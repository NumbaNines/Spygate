#!/usr/bin/env python3
"""
Startup script for SpygateAI Discord Bot
Run this file to start the bot
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spygate_discord.bot.spygate_bot import main

if __name__ == "__main__":
    print("🚀 Starting SpygateAI Discord Bot...")
    main() 