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
EXPLANATION_ENGINE_URL = os.getenv("EXPLANATION_ENGINE_URL", "http://explanation_engine:8000")

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
    check_anomalies.start()  # Phase 3: Anomaly monitoring
    
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


# ========== Phase 3: Anomaly Detection Commands ==========

@tree.command(name="anomaly_stats", description="Get anomaly detection statistics")
async def anomaly_stats(interaction: discord.Interaction):
    """Get anomaly statistics"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ANOMALY_DETECTOR_URL}/stats") as resp:
                stats = await resp.json()
        
        embed = discord.Embed(
            title="📊 Anomaly Detection Statistics",
            color=discord.Color.blue()
        )
        
        # Anomaly counts
        embed.add_field(name="🔴 Critical", value=str(stats.get('critical_count', 0)), inline=True)
        embed.add_field(name="🟠 High", value=str(stats.get('high_count', 0)), inline=True)
        embed.add_field(name="🟡 Medium", value=str(stats.get('medium_count', 0)), inline=True)
        embed.add_field(name="🟢 Low", value=str(stats.get('low_count', 0)), inline=True)
        embed.add_field(name="📈 Total", value=str(stats.get('total_anomalies', 0)), inline=True)
        embed.add_field(name="🎯 Baselines", value=str(stats.get('total_baselines', 0)), inline=True)
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="baselines", description="View learned baselines")
async def baselines(interaction: discord.Interaction):
    """View learned baselines"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ANOMALY_DETECTOR_URL}/baselines") as resp:
                data = await resp.json()
        
        baselines_list = data.get("baselines", [])
        
        if not baselines_list:
            await interaction.followup.send("ℹ️ No baselines learned yet!")
            return
        
        embed = discord.Embed(
            title="🎯 Learned Baselines",
            color=discord.Color.green(),
            description=f"Total: {len(baselines_list)} baselines"
        )
        
        for entity_id in baselines_list[:15]:
            embed.add_field(name=entity_id, value="✅ Active", inline=True)
        
        if len(baselines_list) > 15:
            embed.set_footer(text=f"... and {len(baselines_list) - 15} more")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="learn_baseline", description="Learn baseline for an entity")
@app_commands.describe(
    entity_id="Entity ID (e.g., light.kitchen)",
    days="Days of history (default: 7)"
)
async def learn_baseline(interaction: discord.Interaction, entity_id: str, days: int = 7):
    """Learn baseline for entity"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ANOMALY_DETECTOR_URL}/baseline/learn",
                json={"entity_id": entity_id, "lookback_days": days}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    await interaction.followup.send(f"❌ {error[:200]}")
                    return
                data = await resp.json()
        
        baseline = data.get("baseline", {})
        
        embed = discord.Embed(
            title=f"✅ Baseline Learned",
            description=entity_id,
            color=discord.Color.green()
        )
        
        embed.add_field(name="Data Points", value=str(baseline.get('data_points', 0)), inline=True)
        embed.add_field(name="Type", value=baseline.get('state_type', 'unknown').title(), inline=True)
        
        if baseline.get('state_type') == 'binary':
            embed.add_field(
                name="Typical ON Hours",
                value=str(baseline.get('typical_on_hours', [])),
                inline=False
            )
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


# ========== Phase 4: Explanation Engine Commands ==========

@tree.command(name="explain", description="Explain an anomaly")
@app_commands.describe(
    anomaly_id="Anomaly ID to explain",
    detail="Detail level: simple or detailed (default: simple)"
)
async def explain_anomaly_cmd(
    interaction: discord.Interaction,
    anomaly_id: str,
    detail: str = "simple"
):
    """Explain why an anomaly was detected"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{EXPLANATION_ENGINE_URL}/explain/anomaly/{anomaly_id}",
                params={"detail_level": detail}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    await interaction.followup.send(f"❌ {error[:200]}")
                    return
                explanation = await resp.json()
        
        # Create rich embed
        severity_colors = {
            "low": discord.Color.green(),
            "medium": discord.Color.gold(),
            "high": discord.Color.orange(),
            "critical": discord.Color.red()
        }
        
        severity = explanation.get('metadata', {}).get('severity', 'unknown')
        color = severity_colors.get(severity, discord.Color.blue())
        
        embed = discord.Embed(
            title=f"💡 Anomaly Explanation",
            description=explanation['summary'],
            color=color
        )
        
        # Add metadata
        embed.add_field(
            name="Entity",
            value=explanation['metadata']['entity_id'],
            inline=True
        )
        embed.add_field(
            name="Type",
            value=explanation['metadata']['anomaly_type'].replace('_', ' ').title(),
            inline=True
        )
        embed.add_field(
            name="Severity",
            value=severity.upper(),
            inline=True
        )
        
        # Add probable causes
        if explanation.get('probable_causes'):
            causes_text = ""
            for idx, cause in enumerate(explanation['probable_causes'][:3], 1):
                prob = cause['probability'] * 100
                causes_text += f"{idx}. {cause['cause']} ({prob:.0f}%)\n"
            
            embed.add_field(
                name="Probable Causes",
                value=causes_text,
                inline=False
            )
        
        # Add cascade effects
        if explanation.get('cascade_effects'):
            effects_text = ""
            for effect in explanation['cascade_effects'][:3]:
                emoji = "🔴" if effect['severity'] == "high" else "🟡" if effect['severity'] == "medium" else "🟢"
                effects_text += f"{emoji} {effect['effect']} ({effect['probability']*100:.0f}%)\n"
            
            embed.add_field(
                name="Cascade Effects",
                value=effects_text,
                inline=False
            )
        
        embed.set_footer(text=f"Confidence: {explanation['confidence']*100:.0f}% | {detail.title()} detail")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="explain_prediction", description="Explain a prediction")
@app_commands.describe(
    entity_id="Entity to explain (e.g., light.kitchen)",
    detail="Detail level: simple or detailed (default: simple)"
)
async def explain_prediction_cmd(
    interaction: discord.Interaction,
    entity_id: str,
    detail: str = "simple"
):
    """Explain how a prediction was made"""
    await interaction.response.defer()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{EXPLANATION_ENGINE_URL}/explain/prediction/{entity_id}",
                params={"detail_level": detail}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    await interaction.followup.send(f"❌ {error[:200]}")
                    return
                explanation = await resp.json()
        
        embed = discord.Embed(
            title=f"🔮 Prediction Explanation",
            description=explanation['summary'],
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="Entity",
            value=explanation['entity_id'],
            inline=True
        )
        embed.add_field(
            name="Accuracy",
            value=f"{explanation['model_accuracy']*100:.1f}%",
            inline=True
        )
        embed.add_field(
            name="Training Data",
            value=f"{explanation['training_records']} records",
            inline=True
        )
        
        # Add feature importance (detailed only)
        if detail == "detailed" and explanation.get('factors'):
            factors_text = ""
            for factor in explanation['factors']:
                factors_text += f"{factor['rank']}. {factor['feature']} ({factor['importance']})\n"
            
            embed.add_field(
                name="Top Features",
                value=factors_text,
                inline=False
            )
        
        embed.set_footer(text=f"{detail.title()} detail level")
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


@tree.command(name="explain_last", description="Explain the most recent anomaly")
async def explain_last_anomaly(interaction: discord.Interaction):
    """Explain the most recent anomaly detected"""
    await interaction.response.defer()
    
    try:
        # Get most recent anomaly
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ANOMALY_DETECTOR_URL}/anomalies",
                params={"limit": 1}
            ) as resp:
                if resp.status != 200:
                    await interaction.followup.send("❌ No anomalies found")
                    return
                data = await resp.json()
        
        anomalies = data.get('anomalies', [])
        if not anomalies:
            await interaction.followup.send("✅ No anomalies detected!")
            return
        
        # Get explanation for most recent
        anomaly_id = anomalies[0]['anomaly_id']
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{EXPLANATION_ENGINE_URL}/explain/anomaly/{anomaly_id}",
                params={"detail_level": "detailed"}
            ) as resp:
                if resp.status != 200:
                    await interaction.followup.send("❌ Failed to generate explanation")
                    return
                explanation = await resp.json()
        
        # Create embed (same as explain_anomaly_cmd)
        severity = explanation.get('metadata', {}).get('severity', 'unknown')
        severity_colors = {
            "low": discord.Color.green(),
            "medium": discord.Color.gold(),
            "high": discord.Color.orange(),
            "critical": discord.Color.red()
        }
        color = severity_colors.get(severity, discord.Color.blue())
        
        embed = discord.Embed(
            title=f"💡 Latest Anomaly Explanation",
            description=explanation['summary'],
            color=color
        )
        
        embed.add_field(name="Entity", value=explanation['metadata']['entity_id'], inline=True)
        embed.add_field(name="Type", value=explanation['metadata']['anomaly_type'].replace('_', ' ').title(), inline=True)
        embed.add_field(name="Severity", value=severity.upper(), inline=True)
        
        # Probable causes
        if explanation.get('probable_causes'):
            causes_text = "\n".join([
                f"{i+1}. {c['cause']} ({c['probability']*100:.0f}%)"
                for i, c in enumerate(explanation['probable_causes'][:3])
            ])
            embed.add_field(name="Probable Causes", value=causes_text, inline=False)
        
        # Cascade effects
        if explanation.get('cascade_effects'):
            effects_text = "\n".join([
                f"{'🔴' if e['severity']=='high' else '🟡' if e['severity']=='medium' else '🟢'} {e['effect']}"
                for e in explanation['cascade_effects'][:3]
            ])
            embed.add_field(name="Cascade Effects", value=effects_text, inline=False)
        
        await interaction.followup.send(embed=embed)
        
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {str(e)}")


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


@tasks.loop(minutes=2)
async def check_anomalies():
    """Check for new anomalies every 2 minutes (Phase 3)"""
    if not notification_channel_id:
        return
    
    try:
        # Get high and critical severity anomalies
        async with aiohttp.ClientSession() as session:
            critical_resp = await session.get(
                f"{ANOMALY_DETECTOR_URL}/anomalies",
                params={"severity": "critical", "limit": 10}
            )
            high_resp = await session.get(
                f"{ANOMALY_DETECTOR_URL}/anomalies",
                params={"severity": "high", "limit": 10}
            )
            
            critical_data = await critical_resp.json()
            high_data = await high_resp.json()
        
        # Combine and check for unnotified anomalies
        all_anomalies = (
            critical_data.get("anomalies", []) + 
            high_data.get("anomalies", [])
        )
        
        for anomaly in all_anomalies:
            anomaly_id = anomaly['anomaly_id']
            notified_key = f"claudehome:discord:notified_anomaly:{anomaly_id}"
            
            if await redis_client.exists(notified_key):
                continue
            
            # Send notification
            channel = client.get_channel(notification_channel_id)
            if channel:
                severity = anomaly['severity']
                
                # Color and emoji based on severity
                if severity == 'critical':
                    color = discord.Color.red()
                    emoji = "🔴"
                    title = "CRITICAL ANOMALY DETECTED"
                else:
                    color = discord.Color.orange()
                    emoji = "🟠"
                    title = "High Severity Anomaly"
                
                embed = discord.Embed(
                    title=f"{emoji} {title}",
                    description=anomaly['description'],
                    color=color
                )
                
                embed.add_field(
                    name="Entity",
                    value=anomaly['entity_id'],
                    inline=True
                )
                embed.add_field(
                    name="Type",
                    value=anomaly['anomaly_type'].replace('_', ' ').title(),
                    inline=True
                )
                embed.add_field(
                    name="Severity",
                    value=severity.upper(),
                    inline=True
                )
                
                if anomaly.get('current_value'):
                    embed.add_field(
                        name="Current Value",
                        value=str(anomaly['current_value']),
                        inline=True
                    )
                
                if anomaly.get('expected_value'):
                    embed.add_field(
                        name="Expected",
                        value=str(anomaly['expected_value']),
                        inline=True
                    )
                
                embed.add_field(
                    name="Score",
                    value=f"{anomaly['score']:.2f}",
                    inline=True
                )
                
                embed.set_footer(text=anomaly['timestamp'][:19].replace('T', ' '))
                
                await channel.send(embed=embed)
                
                # Mark as notified
                await redis_client.setex(notified_key, 86400, "1")
        
    except Exception as e:
        logger.error(f"Error checking anomalies: {e}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not set!")
        exit(1)
    
    client.run(DISCORD_TOKEN)