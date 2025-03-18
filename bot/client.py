"""
TradeMaster 2.0 Discord Client
Handles Discord events and interactions.
"""

import discord
from discord import app_commands
from discord.ext import commands
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

# Fix imports by using the "sys.path" approach
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from core directly
from core.llm import LLMEngine
from core.context import ContextManager
from core.gatekeeper import Gatekeeper

# Set up logger for this module
logger = logging.getLogger("TradeMaster.Client")

class TradeMasterClient(commands.Bot):
    """Main Discord bot client for TradeMaster."""
    
    def __init__(self):
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        
        # Initialize bot with command prefix
        super().__init__(
            command_prefix="!",  # Traditional prefix for fallback
            intents=intents,
            help_command=None  # Disable default help command
        )
        
        # Initialize components (placeholders for now)
        self.context_manager = ContextManager()
        self.llm_engine = LLMEngine()
        self.gatekeeper = Gatekeeper()
        
        logger.info("TradeMaster client initialized")
    
    async def setup_hook(self):
        """Register slash commands when the bot is being set up."""
        # Load commands from bot/commands.py
        from bot.commands import setup_commands
        await setup_commands(self)
        
        # Sync commands with Discord
        await self.tree.sync()
        logger.info("Slash commands synchronized")
    
    async def on_ready(self):
        """Handle the bot's connection to Discord."""
        logger.info(f"Bot connected as {self.user}")
        logger.info(f"Connected to {len(self.guilds)} servers")
        
        # Set presence (status)
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="markets | /help"
        )
        await self.change_presence(activity=activity)
    
    async def on_message(self, message):
        """Handle incoming messages."""
        # Ignore own messages
        if message.author == self.user:
            return
        
        # Process commands with the command prefix
        await self.process_commands(message)
        
        # Ignore messages with a prefix or from bots
        if message.content.startswith(self.command_prefix) or message.author.bot:
            return
        
        # Update user context
        user_id = str(message.author.id)
        self.context_manager.update_last_message(user_id, message.content)
        
        try:
            # For Phase 1: Simple response to mentions
            bot_mentioned = self.user.mentioned_in(message)
            
            # In Phase 2, we'll replace this with proper gatekeeper logic
            should_respond = bot_mentioned
            
            if should_respond:
                async with message.channel.typing():
                    # For Phase 1: Simple static response
                    # In Phase 2, we'll use the LLM for responses
                    response = "Hello! I'm TradeMaster, a trading assistant bot. I'm currently being rebuilt with improved capabilities. For now, I can only respond to basic mentions."
                    
                    await message.reply(response)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if bot_mentioned:
                await message.reply("I encountered an error. My developers are working on it!")