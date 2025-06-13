# SpygateAI Discord Bot Setup Guide 🎮

## Quick Setup Overview

This guide will help you set up the SpygateAI Discord Bot to analyze Madden gameplay clips directly in your Discord server!

## Prerequisites

- Python 3.8+ installed
- Discord account
- Server where you have Administrator permissions

## Step 1: Discord Developer Portal Setup

### 1.1 Create Discord Application

1. Go to https://discord.com/developers/applications
2. Click **"New Application"**
3. Name it **"SpygateAI"** (or your preferred name)
4. Click **"Create"**

### 1.2 Create Bot User

1. In your application, click **"Bot"** in the left sidebar
2. Click **"Add Bot"**
3. Confirm by clicking **"Yes, do it!"**
4. **IMPORTANT**: Copy the **Bot Token** (you'll need this!)
   - Click **"Copy"** under the Token section
   - Save this token securely - you'll add it to your `.env` file

### 1.3 Configure Bot Permissions

In the Bot settings:

- ✅ Enable **"MESSAGE CONTENT INTENT"** (required for file uploads)
- ✅ Enable **"SERVER MEMBERS INTENT"** (for user management)
- ✅ Enable **"GUILDS INTENT"** (enabled by default)

### 1.4 Generate Invite Link

1. Click **"OAuth2"** → **"URL Generator"** in left sidebar
2. Under **Scopes**, select:
   - ✅ `bot`
   - ✅ `applications.commands`
3. Under **Bot Permissions**, select:
   - ✅ Send Messages
   - ✅ Embed Links
   - ✅ Attach Files
   - ✅ Read Message History
   - ✅ Use Slash Commands
   - ✅ Add Reactions
4. Copy the generated URL at the bottom

### 1.5 Invite Bot to Your Server

1. Open the invite URL you just copied
2. Select your Discord server
3. Click **"Authorize"**
4. Complete the captcha

## Step 2: Environment Setup

### 2.1 Create Environment File

Create a `.env` file in your main project directory (same level as this file):

```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_bot_token_here

# Optional: Production Mode
SPYGATE_PRODUCTION=false
```

**Replace `your_bot_token_here` with the actual bot token from Step 1.2!**

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have all dependencies, install specifically:

```bash
pip install discord.py aiohttp aiofiles python-dotenv psutil
```

## Step 3: Run the Bot

### 3.1 Start the Bot

```bash
python spygate_discord/run_bot.py
```

Or from the main directory:

```bash
python -m spygate_discord.run_bot
```

### 3.2 Verify Bot is Online

You should see:

```
🚀 Starting SpygateAI Discord Bot...
🤖 SpygateAI Discord Bot starting up...
✅ Loaded commands.analysis_commands
✅ Loaded commands.utility_commands
✅ Loaded commands.help_commands
✅ Synced X slash commands
🎮 SpygateAI Bot is online!
📊 Connected as: SpygateAI#1234
🏷️ Bot ID: 123456789012345678
🌐 Connected to 1 servers
```

## Step 4: Test the Bot

### 4.1 Basic Commands

In your Discord server, try:

- `/ping` - Check if bot responds
- `/info` - Get bot information
- `/help` - View all available commands

### 4.2 Analysis Testing

1. **Auto Analysis**: Upload a Madden video file (.mp4, .mov, etc.) to any channel
   - Bot should automatically analyze it
2. **Manual Analysis**: Use `/analyze video:<upload_file>`

   - More detailed analysis options

3. **Other Features**:
   - `/hud-detect` - Extract HUD elements
   - `/formation` - Identify formations
   - `/compare` - Compare two clips

## Features Overview

### 🎮 Core Features

- **Auto-Analysis**: Upload any Madden clip, get instant analysis
- **HUD Detection**: Extract score, down/distance, time, field position
- **Formation ID**: Identify offensive and defensive formations
- **Play Comparison**: Compare two clips side-by-side
- **Strategic Insights**: Get AI-powered gameplay suggestions

### 📊 Slash Commands

- `/analyze` - Complete clip analysis
- `/hud-detect` - HUD element extraction
- `/formation` - Formation identification
- `/compare` - Clip comparison
- `/ping` - Bot status check
- `/stats` - Usage statistics
- `/info` - Bot information
- `/help` - Command help
- `/tutorial` - Interactive tutorial
- `/examples` - See example outputs

### ⚙️ Admin Commands

- `/system` - System information (admin only)
- `/feedback` - Submit feedback/bug reports

## Troubleshooting

### Bot Not Responding

1. Check the bot token in `.env` file
2. Verify bot has proper permissions in your server
3. Ensure "MESSAGE CONTENT INTENT" is enabled in Discord Developer Portal

### Analysis Not Working

1. Check file format (must be .mp4, .mov, .avi, .mkv, .webm, .m4v)
2. Verify file size is under 50MB
3. Ensure SpygateAI core is properly installed

### Slash Commands Not Appearing

1. Wait a few minutes after starting the bot
2. Try typing `/` in Discord to refresh command list
3. Restart the bot if commands still don't appear

## Support

- Use `/feedback` command in Discord for bug reports
- Check console output for error messages
- Verify all dependencies are installed correctly

## Security Notes

- **Never share your bot token publicly**
- Keep your `.env` file secure and never commit it to version control
- Add `.env` to your `.gitignore` file

## Next Steps

Once your bot is running:

1. Create dedicated channels like `#madden-analysis`
2. Pin the `/info` command output for user reference
3. Test with various Madden clips to verify analysis quality
4. Share the bot with your gaming community!

---

**Ready to analyze some Madden clips?** 🏈

Upload a video file to any channel or use `/analyze` to get started!
