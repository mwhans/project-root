import discord
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger
import asyncio
from discord.errors import Forbidden, NotFound

class ChannelManager:
    def __init__(self, bot):
        self.bot = bot
        self.rate_limit_delay = 1.0  # Delay between channel creations to avoid rate limits

    async def setup_categories(self, guild: discord.Guild, categories: List[Dict]):
        """Sets up all categories and their channels with rate limiting"""
        logger.info(f"Setting up {len(categories)} categories in {guild.name}")
        
        for category_data in categories:
            try:
                # Create category
                category = await guild.create_category(name=category_data['name'])
                logger.info(f"Created category: {category.name}")
                
                # Add delay to avoid rate limits
                await asyncio.sleep(self.rate_limit_delay)
                
                # Create channels within category
                await self.setup_channels(guild, category, category_data['channels'])
                
            except Forbidden as e:
                logger.error(f"Permission error creating category {category_data['name']}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error creating category {category_data['name']}: {e}")
                raise

    async def setup_channels(self, 
                           guild: discord.Guild, 
                           category: discord.CategoryChannel, 
                           channels: List[Dict]):
        """Sets up channels within a category with rate limiting"""
        for channel_data in channels:
            try:
                channel_type = channel_data.get('type', 'text')
                
                if channel_type == 'text':
                    channel = await guild.create_text_channel(
                        name=channel_data['name'],
                        category=category
                    )
                elif channel_type == 'voice':
                    channel = await guild.create_voice_channel(
                        name=channel_data['name'],
                        category=category
                    )
                
                logger.info(f"Created channel: {channel.name} in {category.name}")
                
                # Apply permissions
                if 'permissions' in channel_data:
                    await self.apply_channel_permissions(channel, channel_data['permissions'])
                
                # Add delay to avoid rate limits
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error creating channel {channel_data['name']}: {e}")
                raise

    async def apply_channel_permissions(self, 
                                     channel: discord.abc.GuildChannel, 
                                     permissions: Dict):
        """Applies permission overwrites to a channel"""
        try:
            overwrites = {}
            
            # Default role permissions
            if permissions.get('read_only'):
                overwrites[channel.guild.default_role] = discord.PermissionOverwrite(
                    send_messages=False,
                    add_reactions=False
                )
            
            # Role-specific permissions
            roles_map = {
                'admin_only': 'Admin',
                'judge_only': 'Judge',
                'lawyer_only': 'Lawyer'
            }
            
            for perm_key, role_name in roles_map.items():
                if permissions.get(perm_key):
                    role = discord.utils.get(channel.guild.roles, name=role_name)
                    if role:
                        overwrites[role] = discord.PermissionOverwrite(
                            send_messages=True,
                            add_reactions=True,
                            read_messages=True
                        )
            
            await channel.edit(overwrites=overwrites)
            logger.info(f"Applied permissions to channel: {channel.name}")
            
        except Exception as e:
            logger.error(f"Error applying permissions to {channel.name}: {e}")
            raise

    async def create_backup(self, guild: discord.Guild) -> Dict:
        """Creates a backup of the current server structure"""
        backup = {
            "guild_name": guild.name,
            "guild_id": guild.id,
            "timestamp": datetime.utcnow().isoformat(),
            "categories": []
        }
        
        for category in guild.categories:
            category_data = {
                "name": category.name,
                "position": category.position,
                "channels": []
            }
            
            for channel in category.channels:
                channel_data = {
                    "name": channel.name,
                    "type": "text" if isinstance(channel, discord.TextChannel) else "voice",
                    "position": channel.position,
                    "permissions": {}
                }
                
                # Backup permissions
                for target, overwrite in channel.overwrites.items():
                    channel_data["permissions"][target.name] = {
                        "send_messages": overwrite.send_messages,
                        "read_messages": overwrite.read_messages,
                        "add_reactions": overwrite.add_reactions
                    }
                
                category_data["channels"].append(channel_data)
            
            backup["categories"].append(category_data)
        
        return backup

    async def archive_inactive_channels(self, guild: discord.Guild, days: int) -> List[str]:
        """Archives channels that have been inactive for the specified number of days"""
        archived_channels = []
        archive_category = await self._get_or_create_archive_category(guild)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        for channel in guild.text_channels:
            try:
                if channel.category == archive_category:
                    continue
                
                # Get last message
                last_message = await channel.history(limit=1).flatten()
                
                if not last_message or last_message[0].created_at < cutoff_date:
                    await channel.edit(category=archive_category)
                    archived_channels.append(channel.name)
                    logger.info(f"Archived channel: {channel.name}")
                    
                    # Add delay to avoid rate limits
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except NotFound:
                logger.warning(f"Channel {channel.name} was deleted during archival process")
            except Exception as e:
                logger.error(f"Error archiving channel {channel.name}: {e}")
        
        return archived_channels

    async def _get_or_create_archive_category(self, guild: discord.Guild) -> discord.CategoryChannel:
        """Gets or creates the archive category"""
        archive_category = discord.utils.get(guild.categories, name="ARCHIVED")
        
        if not archive_category:
            archive_category = await guild.create_category(name="ARCHIVED")
            logger.info("Created archive category")
        
        return archive_category 