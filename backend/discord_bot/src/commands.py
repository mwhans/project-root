from discord.ext import commands
from discord import app_commands
import discord
from loguru import logger
import json
import os

class ServerSetup(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        
    @commands.Cog.listener()
    async def on_ready(self):
        """When the bot is ready, set up the server structure"""
        logger.info(f"Bot is ready as {self.bot.user}")
        
        # Load server structure
        config_path = os.path.join("config", "hospitality-server.json")
        with open(config_path, "r") as f:
            structure = json.load(f)
            
        # List all available servers
        logger.info("Available servers:")
        for guild in self.bot.guilds:
            logger.info(f"- {guild.name} (ID: {guild.id})")
            
        # Get the target server (you can modify this to target your specific server)
        target_server_name = "Hospitality System"  # Replace with your server name
        guild = discord.utils.get(self.bot.guilds, name=target_server_name)
        
        if not guild:
            logger.error(f"Could not find server named '{target_server_name}'")
            logger.info("Please make sure to use one of the available server names listed above")
            return
            
        logger.info(f"Setting up server structure for {guild.name}")
        
        try:
            await self.bot.channel_manager.setup_categories(guild, structure["categories"])
            logger.info("Server setup complete!")
        except Exception as e:
            logger.error(f"Failed to set up server: {e}")

    @commands.command(name='delete_all')
    @commands.has_permissions(administrator=True)
    async def delete_all(self, ctx):
        """Delete all channels and categories in the server. Requires administrator permissions."""
        try:
            guild = ctx.guild
            logger.info(f"Starting deletion of all channels and categories in {guild.name}")
            
            # Delete all channels
            for channel in guild.channels:
                try:
                    await channel.delete()
                    logger.info(f"Deleted channel: {channel.name}")
                except Exception as e:
                    logger.error(f"Failed to delete channel {channel.name}: {e}")
            
            logger.info("Server cleanup complete!")
            
        except Exception as e:
            logger.error(f"Failed to clean up server: {e}")

async def setup(bot):
    await bot.add_cog(ServerSetup(bot)) 