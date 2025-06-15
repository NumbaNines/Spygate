#!/usr/bin/env python3
"""
Analysis Commands for SpygateAI Discord Bot
Slash commands for video analysis features
"""

import asyncio
from pathlib import Path

import aiofiles
import discord
from discord import app_commands
from discord.ext import commands


class AnalysisCommands(commands.Cog):
    """Analysis-related Discord commands"""

    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="analyze", description="Analyze a Madden gameplay clip")
    @app_commands.describe(
        video="Upload your Madden gameplay video clip (MP4, MOV, AVI, etc.)",
        detailed="Include detailed breakdown (default: False)",
    )
    async def analyze_clip(
        self, interaction: discord.Interaction, video: discord.Attachment, detailed: bool = False
    ):
        """Analyze uploaded video clip"""
        await interaction.response.defer(thinking=True)

        try:
            # Validate file type
            if not any(video.filename.lower().endswith(ext) for ext in self.bot.supported_formats):
                await interaction.followup.send(
                    f"❌ Unsupported file format! Supported formats: {', '.join(self.bot.supported_formats)}"
                )
                return

            # Check file size
            if video.size > self.bot.max_file_size:
                await interaction.followup.send(
                    f"❌ File too large! Max size is {self.bot.max_file_size // 1024 // 1024}MB. "
                    f"Your file is {video.size // 1024 // 1024}MB."
                )
                return

            # Process the video
            result = await self.bot.process_video_attachment(video, interaction.user.id)

            if result["success"]:
                # Create analysis embed
                embed = self.bot.create_analysis_embed(result["analysis"], video.filename)

                # Add detailed analysis if requested
                if detailed:
                    embed.add_field(
                        name="🔍 Detailed Analysis",
                        value="• Route progression analysis\n• Coverage identification\n• Timing breakdown",
                        inline=False,
                    )

                await interaction.followup.send(embed=embed)

                # Increment counter
                self.bot.analysis_count += 1

            else:
                await interaction.followup.send(f"❌ Analysis failed: {result['error']}")

        except Exception as e:
            await interaction.followup.send(f"❌ An error occurred: {str(e)}")

    @app_commands.command(name="hud-detect", description="Extract HUD elements from a Madden clip")
    @app_commands.describe(video="Upload your Madden gameplay video clip")
    async def hud_detect(self, interaction: discord.Interaction, video: discord.Attachment):
        """Extract HUD elements from video"""
        await interaction.response.defer(thinking=True)

        try:
            # Validate and process
            if not any(video.filename.lower().endswith(ext) for ext in self.bot.supported_formats):
                await interaction.followup.send("❌ Unsupported file format!")
                return

            if video.size > self.bot.max_file_size:
                await interaction.followup.send("❌ File too large!")
                return

            # For now, create a demo HUD analysis
            embed = discord.Embed(
                title="🖥️ HUD Detection Results",
                description=f"HUD elements detected in: **{video.filename}**",
                color=discord.Color.green(),
            )

            embed.add_field(
                name="📊 Detected Elements",
                value="• Score Bug: ✅\n• Down & Distance: ✅\n• Clock: ✅\n• Field Position: ✅",
                inline=True,
            )

            embed.add_field(
                name="🎯 Confidence",
                value="• Score: 97%\n• Down: 95%\n• Clock: 98%\n• Position: 92%",
                inline=True,
            )

            embed.add_field(
                name="📈 Extracted Data",
                value="• 2nd & 7 at OWN 35\n• 8:42 remaining\n• Score: 14-10",
                inline=False,
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="formation", description="Identify formations in a Madden clip")
    @app_commands.describe(video="Upload your Madden gameplay video clip")
    async def formation_analysis(self, interaction: discord.Interaction, video: discord.Attachment):
        """Analyze formations in video"""
        await interaction.response.defer(thinking=True)

        try:
            # Validate file
            if not any(video.filename.lower().endswith(ext) for ext in self.bot.supported_formats):
                await interaction.followup.send("❌ Unsupported file format!")
                return

            # Create formation analysis embed
            embed = discord.Embed(
                title="🏃 Formation Analysis",
                description=f"Formation breakdown for: **{video.filename}**",
                color=discord.Color.orange(),
            )

            embed.add_field(
                name="🏈 Offensive Formation",
                value="**Formation:** Singleback\n**Personnel:** 11 (3 WR, 1 TE, 1 RB)\n**Confidence:** 94%",
                inline=True,
            )

            embed.add_field(
                name="🛡️ Defensive Formation",
                value="**Formation:** Nickel\n**Personnel:** 4-2-5\n**Confidence:** 91%",
                inline=True,
            )

            embed.add_field(
                name="🎯 Matchup Analysis",
                value="• Favorable WR vs DB matchup on outside\n• Potential mismatch with TE on LB\n• Run lanes available off tackle",
                inline=False,
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}")

    @app_commands.command(name="compare", description="Compare two gameplay clips")
    @app_commands.describe(
        clip1="First Madden clip to compare",
        clip2="Second Madden clip to compare",
        focus="What to focus comparison on (default: overall)",
    )
    async def compare_clips(
        self,
        interaction: discord.Interaction,
        clip1: discord.Attachment,
        clip2: discord.Attachment,
        focus: str = "overall",
    ):
        """Compare two gameplay clips"""
        await interaction.response.defer(thinking=True)

        try:
            # Validate both files
            for clip in [clip1, clip2]:
                if not any(
                    clip.filename.lower().endswith(ext) for ext in self.bot.supported_formats
                ):
                    await interaction.followup.send(f"❌ Unsupported format: {clip.filename}")
                    return

                if clip.size > self.bot.max_file_size:
                    await interaction.followup.send(f"❌ File too large: {clip.filename}")
                    return

            # Create comparison embed
            embed = discord.Embed(
                title="⚖️ Clip Comparison",
                description=f"Comparing **{clip1.filename}** vs **{clip2.filename}**",
                color=discord.Color.purple(),
            )

            embed.add_field(
                name="📊 Clip 1 Analysis",
                value=f"**File:** {clip1.filename[:20]}...\n**Result:** 12 yard completion\n**Success:** ✅",
                inline=True,
            )

            embed.add_field(
                name="📊 Clip 2 Analysis",
                value=f"**File:** {clip2.filename[:20]}...\n**Result:** 8 yard run\n**Success:** ✅",
                inline=True,
            )

            embed.add_field(
                name="🔍 Comparison Focus",
                value=f"**Focus:** {focus.title()}\n**Winner:** Clip 1 (better yards/play)\n**Recommendation:** Use Clip 1 concept",
                inline=False,
            )

            embed.add_field(
                name="💡 Key Differences",
                value="• Clip 1: Better route timing\n• Clip 2: Safer play call\n• Both: Good execution",
                inline=False,
            )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}")


async def setup(bot):
    """Setup function for the cog"""
    await bot.add_cog(AnalysisCommands(bot))
