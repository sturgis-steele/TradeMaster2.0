"""
TradeMaster 2.0 Discord Client
Handles Discord events and interactions.
"""

import discord
from discord import app_commands
from discord.ext import commands, tasks
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import os

# Fix imports by using the "sys.path" approach
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from core directly
from core.llm import LLMEngine
from core.context import ContextManager

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
        
        # Initialize components
        self.context_manager = ContextManager()
        self.llm_engine = LLMEngine()
        
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
        
        # Start background tasks
        self.cleanup_contexts.start()
    
    async def on_message(self, message):
        """Handle incoming messages."""
        # Log all incoming messages for debugging
        logger.debug(f"Received message: {message.content[:50]}... from {message.author.name} in {message.channel.name}")
        
        # Ignore own messages
        if message.author == self.user:
            logger.debug("Ignoring own message")
            return
        
        # Process commands with the command prefix
        await self.process_commands(message)
        
        # Ignore messages with a prefix or from bots
        if message.content.startswith(self.command_prefix):
            logger.debug(f"Ignoring message with prefix: {self.command_prefix}")
            return
            
        if message.author.bot:
            logger.debug("Ignoring message from another bot")
            return
        
        # Update user context
        user_id = str(message.author.id)
        channel_id = str(message.channel.id)
        self.context_manager.update_last_message(user_id, message.content)
        
        # Add channel ID to context
        user_context = self.context_manager.get_context(user_id)
        user_context['channel_id'] = channel_id
        
        try:
            # Check if the bot is mentioned (for logging purposes only)
            bot_mentioned = self.user.mentioned_in(message)
            
            # Direct all messages to the LLM engine without gatekeeper filtering
            async with message.channel.typing():
                # Generate response using LLM engine
                response = await self.llm_engine.generate_response(
                    message.content,
                    user_id,
                    context=user_context
                )
                
                # Update context with bot's response
                self.context_manager.add_bot_response(user_id, response)
                
                # Reply to the message
                await message.reply(response)
                logger.info(f"Responded to message from {message.author.name}: {message.content[:50]}...")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if bot_mentioned:
                await message.reply("I encountered an error processing your request. Please try again later.")
    
    @tasks.loop(hours=6)
    async def cleanup_contexts(self):
        """Periodically clean up expired user contexts."""
        try:
            self.context_manager.clean_expired_contexts()
        except Exception as e:
            logger.error(f"Error cleaning up contexts: {e}")
            
    @cleanup_contexts.before_loop
    async def before_cleanup(self):
        """Wait until the bot is ready before starting the cleanup task."""
        await self.wait_until_ready()