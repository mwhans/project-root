import discord
from typing import Dict, List
from loguru import logger
import asyncio

class PermissionsManager:
    def __init__(self, bot):
        self.bot = bot
        self.rate_limit_delay = 1.0  # Delay between role operations to avoid rate limits
        
        self.role_definitions = [
            {
                "name": "Supreme Court Justice",
                "color": discord.Color.gold(),
                "permissions": discord.Permissions(
                    administrator=True,
                    manage_guild=True,
                    manage_roles=True,
                    manage_channels=True,
                    manage_messages=True,
                    mention_everyone=True,
                    mute_members=True,
                    deafen_members=True,
                    move_members=True,
                    view_audit_log=True
                ),
                "position": 4,
                "hoist": True  # Show users separately in online list
            },
            {
                "name": "Judge",
                "color": discord.Color.blue(),
                "permissions": discord.Permissions(
                    manage_channels=True,
                    manage_messages=True,
                    mention_everyone=True,
                    mute_members=True,
                    move_members=True,
                    view_audit_log=True
                ),
                "position": 3,
                "hoist": True
            },
            {
                "name": "Lawyer",
                "color": discord.Color.green(),
                "permissions": discord.Permissions(
                    send_messages=True,
                    read_messages=True,
                    add_reactions=True,
                    attach_files=True,
                    embed_links=True,
                    read_message_history=True,
                    connect=True,
                    speak=True
                ),
                "position": 2,
                "hoist": True
            },
            {
                "name": "Citizen",
                "color": discord.Color.light_grey(),
                "permissions": discord.Permissions(
                    send_messages=True,
                    read_messages=True,
                    add_reactions=True,
                    connect=True,
                    speak=True
                ),
                "position": 1,
                "hoist": False
            }
        ]

    async def setup_basic_roles(self, guild: discord.Guild) -> List[discord.Role]:
        """Sets up the basic roles needed for the legal system"""
        logger.info(f"Setting up basic roles in {guild.name}")
        created_roles = []
        
        try:
            # Sort roles by position to create them in the right order
            for role_def in sorted(self.role_definitions, key=lambda x: x["position"], reverse=True):
                role = await self._create_or_update_role(guild, role_def)
                created_roles.append(role)
                await asyncio.sleep(self.rate_limit_delay)
            
            logger.info(f"Successfully created {len(created_roles)} roles")
            return created_roles
            
        except Exception as e:
            logger.error(f"Error setting up roles: {e}")
            raise

    async def _create_or_update_role(self, guild: discord.Guild, role_def: Dict) -> discord.Role:
        """Creates or updates a role with the specified definition"""
        try:
            existing_role = discord.utils.get(guild.roles, name=role_def["name"])
            
            if existing_role:
                logger.info(f"Updating existing role: {role_def['name']}")
                await existing_role.edit(
                    permissions=role_def["permissions"],
                    color=role_def["color"],
                    hoist=role_def["hoist"],
                    reason="Updating role permissions"
                )
                return existing_role
            else:
                logger.info(f"Creating new role: {role_def['name']}")
                return await guild.create_role(
                    name=role_def["name"],
                    permissions=role_def["permissions"],
                    color=role_def["color"],
                    hoist=role_def["hoist"],
                    reason="Creating legal system role"
                )
                
        except Exception as e:
            logger.error(f"Error creating/updating role {role_def['name']}: {e}")
            raise

    async def assign_role(self, member: discord.Member, role_name: str) -> bool:
        """Assigns a role to a member"""
        try:
            role = discord.utils.get(member.guild.roles, name=role_name)
            if not role:
                logger.error(f"Role {role_name} not found")
                return False
            
            await member.add_roles(role, reason=f"Assigned {role_name} role")
            logger.info(f"Assigned {role_name} to {member.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error assigning role {role_name} to {member.name}: {e}")
            return False

    async def remove_role(self, member: discord.Member, role_name: str) -> bool:
        """Removes a role from a member"""
        try:
            role = discord.utils.get(member.guild.roles, name=role_name)
            if not role:
                logger.error(f"Role {role_name} not found")
                return False
            
            await member.remove_roles(role, reason=f"Removed {role_name} role")
            logger.info(f"Removed {role_name} from {member.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing role {role_name} from {member.name}: {e}")
            return False

    def get_role_hierarchy(self) -> List[str]:
        """Returns the role hierarchy in order of highest to lowest position"""
        return [role["name"] for role in sorted(self.role_definitions, 
                                              key=lambda x: x["position"], 
                                              reverse=True)]

    async def sync_role_positions(self, guild: discord.Guild):
        """Syncs role positions according to the defined hierarchy"""
        try:
            # Get all roles that we manage
            managed_roles = {role.name: role for role in guild.roles 
                           if role.name in [r["name"] for r in self.role_definitions]}
            
            # Calculate new positions starting from the highest non-managed role
            highest_position = max([r.position for r in guild.roles if r.name not in managed_roles])
            
            positions = {}
            for role_def in sorted(self.role_definitions, key=lambda x: x["position"], reverse=True):
                if role_def["name"] in managed_roles:
                    highest_position -= 1
                    positions[managed_roles[role_def["name"]]] = highest_position
            
            # Update positions
            if positions:
                await guild.edit_role_positions(positions)
                logger.info("Successfully synced role positions")
                
        except Exception as e:
            logger.error(f"Error syncing role positions: {e}")
            raise 