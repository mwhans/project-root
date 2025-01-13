import discord
from discord.ext import commands
import json
import os
from dotenv import load_dotenv
from loguru import logger
from .channel_manager import ChannelManager
from .permissions_manager import PermissionsManager
from .utils import setup_logging

load_dotenv()
setup_logging()

class LegalSystemBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        intents.members = True  # Needed for role management
        
        super().__init__(
            command_prefix='!legal ',
            intents=intents,
            description='Legal System Server Setup Bot'
        )
        
        self.channel_manager = ChannelManager(self)
        self.permissions_manager = PermissionsManager(self)

    async def setup_hook(self):
        logger.info("Setting up bot hooks and extensions...")
        try:
            await self.load_extension('src.commands')
        except Exception as e:
            logger.error(f"Failed to load extensions: {e}")
            raise

    async def on_ready(self):
        logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="the courts"
            )
        )

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setup_server(self, ctx):
        """Sets up the entire server structure"""
        logger.info(f"Starting server setup in guild {ctx.guild.name}")
        status_message = await ctx.send("Beginning server setup...")
        
        try:
            with open('config/server_structure.json', 'r') as f:
                structure = json.load(f)
            
            # Setup roles first
            await self.permissions_manager.setup_basic_roles(ctx.guild)
            await status_message.edit(content="✅ Roles created\n⏳ Setting up channels...")
            
            # Then setup channels with proper permissions
            await self.channel_manager.setup_categories(ctx.guild, structure['categories'])
            await status_message.edit(content="✅ Roles created\n✅ Channels setup\n🎉 Server setup completed!")
            
            logger.info(f"Server setup completed for guild {ctx.guild.name}")
            
        except FileNotFoundError:
            error_msg = "Server structure configuration file not found!"
            logger.error(error_msg)
            await status_message.edit(content=f"❌ Error: {error_msg}")
        except discord.Forbidden as e:
            error_msg = "Bot doesn't have required permissions!"
            logger.error(f"{error_msg} Details: {e}")
            await status_message.edit(content=f"❌ Error: {error_msg}")
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Setup failed: {e}", exc_info=True)
            await status_message.edit(content=f"❌ Error: {error_msg}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def backup_server(self, ctx):
        """Creates a backup of the current server structure"""
        logger.info(f"Starting server backup for guild {ctx.guild.name}")
        await ctx.send("Creating server backup...")
        
        try:
            backup = await self.channel_manager.create_backup(ctx.guild)
            backup_file = f"backups/server_backup_{ctx.guild.id}_{discord.utils.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            
            with open(backup_file, 'w') as f:
                json.dump(backup, f, indent=4)
            
            await ctx.send(f"✅ Server backup created: `{backup_file}`")
            logger.info(f"Backup completed for guild {ctx.guild.name}")
            
        except Exception as e:
            error_msg = f"Failed to create backup: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await ctx.send(f"❌ Error: {error_msg}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def archive_channel(self, ctx, days: int = 30):
        """Archives channels that have been inactive for the specified number of days"""
        logger.info(f"Starting channel archival process in {ctx.guild.name}")
        await ctx.send(f"Scanning for channels inactive for {days} days...")
        
        try:
            archived = await self.channel_manager.archive_inactive_channels(ctx.guild, days)
            await ctx.send(f"✅ Archived {len(archived)} channels")
            
        except Exception as e:
            error_msg = f"Failed to archive channels: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await ctx.send(f"❌ Error: {error_msg}") 