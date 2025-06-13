#!/usr/bin/env python3
"""
Help Commands for SpygateAI Discord Bot
Help system and tutorial commands
"""

import discord
from discord.ext import commands
from discord import app_commands

class HelpCommands(commands.Cog):
    """Help and tutorial Discord commands"""
    
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="help", description="Get help with SpygateAI commands")
    @app_commands.describe(category="Specific help category (optional)")
    async def help_command(self, interaction: discord.Interaction, category: str = None):
        """Display help information"""
        
        if category and category.lower() == "analysis":
            embed = self.create_analysis_help()
        elif category and category.lower() == "utility":
            embed = self.create_utility_help()
        elif category and category.lower() == "setup":
            embed = self.create_setup_help()
        else:
            embed = self.create_main_help()
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    def create_main_help(self):
        """Create main help embed"""
        embed = discord.Embed(
            title="🆘 SpygateAI Help Center",
            description="Welcome to SpygateAI! Here's how to get started:",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="🎮 Quick Start",
            value="1. Upload a Madden clip (drag & drop)\n"
                  "2. Bot automatically analyzes it\n"
                  "3. Get instant insights!\n"
                  "4. Use `/analyze` for detailed breakdown",
            inline=False
        )
        
        embed.add_field(
            name="📋 Command Categories",
            value="• **Analysis Commands** - Video analysis features\n"
                  "• **Utility Commands** - Bot stats and information\n"
                  "• **Help Commands** - Tutorials and guides\n\n"
                  "Use `/help category:<name>` for specific help",
            inline=False
        )
        
        embed.add_field(
            name="🚀 Popular Commands",
            value="`/analyze` - Full clip analysis\n"
                  "`/hud-detect` - Extract HUD data\n"
                  "`/formation` - Identify formations\n"
                  "`/compare` - Compare two clips\n"
                  "`/info` - About SpygateAI",
            inline=True
        )
        
        embed.add_field(
            name="💡 Tips",
            value="• Upload in 1080p+ for best results\n"
                  "• Ensure HUD is visible\n"
                  "• Keep files under 50MB\n"
                  "• Try auto-upload for quick analysis",
            inline=True
        )
        
        embed.add_field(
            name="🔗 Getting Started",
            value="Use `/tutorial` for a step-by-step guide\n"
                  "Need help? Use `/feedback` to contact us",
            inline=False
        )
        
        embed.set_footer(text="Use /help category:analysis for detailed command help")
        return embed

    def create_analysis_help(self):
        """Create analysis commands help"""
        embed = discord.Embed(
            title="🎮 Analysis Commands Help",
            description="Detailed guide to SpygateAI analysis features",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="/analyze",
            value="**Usage:** `/analyze video:<file> detailed:<true/false>`\n"
                  "**Purpose:** Complete gameplay analysis\n"
                  "**Output:** Game state, formations, suggestions\n"
                  "**Example:** Upload clip → Get strategic insights",
            inline=False
        )
        
        embed.add_field(
            name="/hud-detect",
            value="**Usage:** `/hud-detect video:<file>`\n"
                  "**Purpose:** Extract HUD elements\n"
                  "**Output:** Score, down/distance, time, position\n"
                  "**Best for:** Verifying game state data",
            inline=False
        )
        
        embed.add_field(
            name="/formation",
            value="**Usage:** `/formation video:<file>`\n"
                  "**Purpose:** Identify offensive/defensive formations\n"
                  "**Output:** Formation names, personnel, matchups\n"
                  "**Best for:** Understanding pre-snap setup",
            inline=False
        )
        
        embed.add_field(
            name="/compare",
            value="**Usage:** `/compare clip1:<file> clip2:<file> focus:<aspect>`\n"
                  "**Purpose:** Side-by-side clip comparison\n"
                  "**Output:** Performance comparison, recommendations\n"
                  "**Best for:** Choosing between play concepts",
            inline=False
        )
        
        embed.add_field(
            name="📁 File Requirements",
            value="• **Formats:** MP4, MOV, AVI, MKV, WEBM, M4V\n"
                  "• **Size:** Maximum 50MB per file\n"
                  "• **Quality:** 1080p+ recommended\n"
                  "• **Length:** Any duration (shorter = faster)",
            inline=False
        )
        
        return embed

    def create_utility_help(self):
        """Create utility commands help"""
        embed = discord.Embed(
            title="🛠️ Utility Commands Help",
            description="Bot information and utility features",
            color=discord.Color.orange()
        )
        
        embed.add_field(
            name="/ping",
            value="Check bot response time and status",
            inline=True
        )
        
        embed.add_field(
            name="/stats",
            value="View usage statistics and performance",
            inline=True
        )
        
        embed.add_field(
            name="/info",
            value="Detailed information about SpygateAI",
            inline=True
        )
        
        embed.add_field(
            name="/system",
            value="System information (Admin only)",
            inline=True
        )
        
        embed.add_field(
            name="/feedback",
            value="Submit feedback or bug reports",
            inline=True
        )
        
        embed.add_field(
            name="/help",
            value="This help system",
            inline=True
        )
        
        return embed

    def create_setup_help(self):
        """Create setup help for server administrators"""
        embed = discord.Embed(
            title="⚙️ Server Setup Help",
            description="Guide for setting up SpygateAI in your server",
            color=discord.Color.red()
        )
        
        embed.add_field(
            name="🎯 Recommended Channels",
            value="• `#madden-analysis` - Main analysis channel\n"
                  "• `#clip-uploads` - Video upload area\n"
                  "• `#spygate-help` - Support and questions",
            inline=False
        )
        
        embed.add_field(
            name="🔒 Permissions Needed",
            value="• Send Messages\n• Embed Links\n• Attach Files\n• Read Message History\n• Use Slash Commands",
            inline=True
        )
        
        embed.add_field(
            name="📊 Best Practices",
            value="• Pin `/info` message in analysis channel\n"
                  "• Create role for frequent users\n"
                  "• Set up auto-delete for large files\n"
                  "• Enable slow mode for upload channels",
            inline=True
        )
        
        embed.add_field(
            name="🎮 Usage Tips",
            value="• Encourage 1080p+ uploads\n"
                  "• Remind users about 50MB limit\n"
                  "• Promote `/analyze detailed:True` for learning\n"
                  "• Use `/stats` to track server usage",
            inline=False
        )
        
        return embed

    @app_commands.command(name="tutorial", description="Interactive tutorial for new users")
    async def tutorial(self, interaction: discord.Interaction):
        """Interactive tutorial for new users"""
        embed = discord.Embed(
            title="🎓 SpygateAI Tutorial",
            description="Learn how to use SpygateAI effectively!",
            color=discord.Color.purple()
        )
        
        embed.add_field(
            name="📚 Step 1: Upload Your First Clip",
            value="• Drag and drop a Madden video file into any channel\n"
                  "• The bot will automatically start analyzing it\n"
                  "• Wait for the analysis results embed\n"
                  "• Review the game state and formation data",
            inline=False
        )
        
        embed.add_field(
            name="🔍 Step 2: Try Detailed Analysis",
            value="• Use `/analyze video:<your_clip> detailed:True`\n"
                  "• Get extended breakdown with suggestions\n"
                  "• Learn about route concepts and timing\n"
                  "• Understand defensive coverage reactions",
            inline=False
        )
        
        embed.add_field(
            name="⚖️ Step 3: Compare Plays",
            value="• Upload two similar plays\n"
                  "• Use `/compare clip1:<first> clip2:<second>`\n"
                  "• See which play concept worked better\n"
                  "• Apply insights to future gameplay",
            inline=False
        )
        
        embed.add_field(
            name="🎯 Step 4: Optimize Your Uploads",
            value="• Record in 1080p or higher resolution\n"
                  "• Ensure HUD elements are clearly visible\n"
                  "• Keep clips focused on key plays\n"
                  "• Upload before/after play for context",
            inline=False
        )
        
        embed.add_field(
            name="💡 Pro Tips",
            value="• Use specific play situation clips for best analysis\n"
                  "• Try different camera angles to see formations\n"
                  "• Upload both successful and failed plays to learn\n"
                  "• Ask for feedback using `/feedback` command",
            inline=False
        )
        
        embed.set_footer(text="Ready to analyze? Upload a clip or use /analyze!")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="examples", description="See example analysis outputs")
    async def examples(self, interaction: discord.Interaction):
        """Show example analysis outputs"""
        embed = discord.Embed(
            title="📋 Analysis Examples",
            description="See what SpygateAI analysis looks like",
            color=discord.Color.gold()
        )
        
        embed.add_field(
            name="🎮 Example: Passing Play Analysis",
            value="**Game State:** 2nd & 7 at OWN 35\n"
                  "**Formation:** Singleback vs Nickel\n"
                  "**Result:** 12 yard completion\n"
                  "**Success:** ✅ First down achieved",
            inline=False
        )
        
        embed.add_field(
            name="💡 Example Suggestions",
            value="• Good route concept against nickel coverage\n"
                  "• Consider motion to identify coverage pre-snap\n"
                  "• Excellent timing on the throw\n"
                  "• Try same concept from trips formation",
            inline=False
        )
        
        embed.add_field(
            name="🏃 Example: Formation Detection",
            value="**Offense:** Singleback - 11 Personnel\n"
                  "**Defense:** Nickel - 4-2-5 Personnel\n"
                  "**Matchups:** Favorable WR vs DB outside\n"
                  "**Recommendation:** Target slot receiver on LB",
            inline=False
        )
        
        embed.add_field(
            name="📊 Example: HUD Data",
            value="**Score:** Home 14 - Away 10\n"
                  "**Time:** 8:42 remaining (2nd quarter)\n"
                  "**Position:** Own 35-yard line\n"
                  "**Confidence:** 95-98% accuracy",
            inline=False
        )
        
        embed.set_footer(text="Upload your own clips to see personalized analysis!")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

async def setup(bot):
    """Setup function for the cog"""
    await bot.add_cog(HelpCommands(bot)) 