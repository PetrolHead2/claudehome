"""
Discord Bot - User Interface for ClaudeHome
"""

import os
import json
import asyncio
from typing import Optional
import discord
from discord import app_commands
from discord.ext import tasks
import redis.asyncio as redis
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
PROACTIVE_ENGINE_URL = os.getenv("PROACTIVE_ENGINE_URL", "http://proactive_engine:8000")
ANOMALY_DETECTOR_URL = os.getenv("ANOMALY_DETECTOR_URL", "http://anomaly_detector:8000")

# Discord client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

redis_client = None
notification_channel_id = None


@client.event
async def on_ready():
    global redis_client, notification_channel_id
    
    logger.info(f"Discord bot logged in as {client.user}")
    
    # Connect to Redis
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    
    # Load notification channel
    saved_channel = await redis_client.get("claudehome:discord:notification_channel")
    if saved_channel:
        notification_channel_id = int(saved_channel)
        logger.info(f"Loaded notification channel: {notification_channel_id}")
    
    # Sync commands
    if DISCORD_GUILD_ID:
        guild = discord.Object(id=int(DISCORD_GUILD_ID))
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
        logger.info(f"Synced commands to guild {DISCORD_GUILD_ID}")
    else:
        await tree.sync()
        logger.info("Synced commands globally")
    
    # Start background tasks
    check_suggestions.start()
    
    logger.info("Discord bot ready!")


@tree.command(name="status", description="Get ClaudeHome system status")
async def status(interaction: discord.Interaction):
    """Get system status"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ORCHESTRATOR_URL}/health") as resp:
                data = await resp.json()
        
        embed = discord.Embed(
            title="🏠 ClaudeHome System Status",
            color=discord.Color.green() if data.get("status") == "healthy" else discord.Color.orange()
        )
        
        services = data.get("services", {})
        for service, status in services.items():
            embed.add_field(
                name=service.replace("_", " ").title(),
                value="✅ Healthy" if status else "❌ Down",
                inline=True
            )
        
        embed.add_field(
            name="Processing",
            value="✅ Active" if data.get("processing") else "⏸️ Paused",
            inline=False
        )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error getting status: {str(e)}")


@tree.command(name="suggestions", description="Get pending automation suggestions")
async def suggestions(interaction: discord.Interaction):
    """Get pending suggestions"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROACTIVE_ENGINE_URL}/suggestions/pending") as resp:
                data = await resp.json()
        
        suggestions = data.get("suggestions", [])
        
        if not suggestions:
            await interaction.followup.send("📭 No pending suggestions")
            return
        
        for suggestion in suggestions[:3]:
            embed = discord.Embed(
                title=f"💡 {suggestion['title']}",
                description=suggestion['description'],
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="Confidence",
                value=f"{suggestion['confidence']*100:.0f}%",
                inline=True
            )
            
            view = SuggestionView(suggestion['suggestion_id'])
            await interaction.followup.send(embed=embed, view=view)
        
        if len(suggestions) > 3:
            await interaction.followup.send(f"... and {len(suggestions)-3} more suggestions")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="anomalies", description="Get recent anomalies")
@app_commands.describe(severity="Filter by severity (HIGH, MEDIUM, LOW)")
async def anomalies(interaction: discord.Interaction, severity: Optional[str] = "HIGH"):
    """Get recent anomalies"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ANOMALY_DETECTOR_URL}/anomalies/{severity}") as resp:
                data = await resp.json()
        
        anomalies_list = data.get("anomalies", [])
        
        if not anomalies_list:
            await interaction.followup.send(f"✅ No {severity} anomalies detected")
            return
        
        for anomaly in anomalies_list[:3]:
            color = {
                "CRITICAL": discord.Color.red(),
                "HIGH": discord.Color.orange(),
                "MEDIUM": discord.Color.yellow(),
                "LOW": discord.Color.blue()
            }.get(anomaly['severity'], discord.Color.grey())
            
            embed = discord.Embed(
                title=f"⚠️ {anomaly['anomaly_type']}",
                description=anomaly['description'],
                color=color
            )
            
            embed.add_field(name="Entity", value=anomaly['entity_id'], inline=True)
            embed.add_field(name="Severity", value=anomaly['severity'], inline=True)
            embed.add_field(name="Value", value=str(anomaly['current_value']), inline=True)
            
            await interaction.followup.send(embed=embed)
        
        if len(anomalies_list) > 3:
            await interaction.followup.send(f"... and {len(anomalies_list)-3} more")
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="setup", description="Set up notification channel")
async def setup(interaction: discord.Interaction):
    """Set the current channel for notifications"""
    global notification_channel_id
    
    notification_channel_id = interaction.channel_id
    await redis_client.set("claudehome:discord:notification_channel", str(notification_channel_id))
    
    await interaction.response.send_message(
        f"✅ Notifications will be sent to this channel: <#{notification_channel_id}>"
    )


class SuggestionView(discord.ui.View):
    """Interactive buttons for suggestions"""
    
    def __init__(self, suggestion_id: str):
        super().__init__(timeout=86400)
        self.suggestion_id = suggestion_id
    
    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.success)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{PROACTIVE_ENGINE_URL}/suggestions/{self.suggestion_id}/approve") as resp:
                    data = await resp.json()
            
            await interaction.followup.send(f"✅ Suggestion approved!")
            
            for item in self.children:
                item.disabled = True
            await interaction.message.edit(view=self)
            
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}")
    
    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.danger)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{PROACTIVE_ENGINE_URL}/suggestions/{self.suggestion_id}/reject",
                    params={"reason": "User rejected"}
                ) as resp:
                    data = await resp.json()
            
            await interaction.followup.send(f"❌ Suggestion rejected")
            
            for item in self.children:
                item.disabled = True
            await interaction.message.edit(view=self)
            
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}")


@tasks.loop(minutes=5)
async def check_suggestions():
    """Check for new suggestions every 5 minutes"""
    if not notification_channel_id:
        return
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{PROACTIVE_ENGINE_URL}/suggestions/pending") as resp:
                data = await resp.json()
        
        suggestions = data.get("suggestions", [])
        
        for suggestion in suggestions:
            suggestion_id = suggestion['suggestion_id']
            notified_key = f"claudehome:discord:notified:{suggestion_id}"
            
            if await redis_client.exists(notified_key):
                continue
            
            channel = client.get_channel(notification_channel_id)
            if channel:
                embed = discord.Embed(
                    title=f"💡 New Automation Suggestion",
                    description=suggestion['description'],
                    color=discord.Color.blue()
                )
                
                embed.add_field(
                    name="Confidence",
                    value=f"{suggestion['confidence']*100:.0f}%",
                    inline=True
                )
                
                view = SuggestionView(suggestion_id)
                await channel.send(embed=embed, view=view)
                
                await redis_client.setex(notified_key, 86400, "1")
        
    except Exception as e:
        logger.error(f"Error checking suggestions: {e}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not set!")
        exit(1)
    
    client.run(DISCORD_TOKEN)