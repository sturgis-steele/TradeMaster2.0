# TradeMaster Discord Bot - Client
# This file defines the TradeMasterClient class which handles Discord events and command processing.
# It manages user interactions, message filtering, and communication with the LLM engine.
#
# File Interactions:
# - main.py: Imports and instantiates TradeMasterClient
# - utils/gatekeeper.py: Uses Gatekeeper for message filtering
# - core/llm_engine.py: Uses LLMEngine for processing user messages
# - Discord API: Connects to Discord and handles events

import discord
from discord import app_commands
from discord.ext import commands
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

# Import our gatekeeper and main LLM
from utils.gatekeeper import Gatekeeper
from core.llm_engine import LLMEngine  # This would be your main LLM component

# Logger for this module
logger = logging.getLogger("TradeMaster.Client")

class TradeMasterClient(commands.Bot):
    """
    The main Discord bot client for TradeMaster.
    Handles Discord events and interactions.
    """
    
    def __init__(self):
        # Set up intents (permissions) for the bot
        intents = discord.Intents.default()
        intents.message_content = True  # Needed to read message content
        intents.members = True  # Needed for user-related features
        
        # Initialize the bot with slash commands
        super().__init__(
            command_prefix="!",  # We'll still have a prefix for legacy commands but won't use it
            intents=intents,
            help_command=None,  # Disable default help command
        )
        
        # Dictionary to store active user conversations/contexts
        self.user_contexts: Dict[int, Any] = {}
        
        # Initialize the gatekeeper
        self.gatekeeper = Gatekeeper(
            model_name="gemma3:1b",  # Using available local model
            ollama_base_url="http://localhost:11434"  # Adjust as needed
        )
        logger.info("Initialized gatekeeper for message filtering")
        
        # Initialize the main LLM engine
        self.llm_engine = LLMEngine()  # This would be your main LLM component
        logger.info("Initialized main LLM engine")
    
    async def setup_hook(self):
        """
        Called when the bot is being set up.
        We'll use this to register slash commands.
        """
        # Register slash commands
        @self.tree.command(name="help", description="Get help with using TradeMaster")
        async def help_command(interaction: discord.Interaction):
            """Slash command for getting help."""
            embed = discord.Embed(
                title="TradeMaster Help",
                description="I'm an AI assistant for trading discussions. You can talk to me normally in any channel or use /help for this message.",
                color=discord.Color.blue()
            )
            await interaction.response.send_message(embed=embed)
        
        # Sync the commands with Discord
        await self.tree.sync()
        logger.info("Slash commands synchronized")
    
    async def on_ready(self):
        """
        Event handler called when the bot has connected to Discord.
        """
        logger.info(f'{self.user} has connected to Discord!')
        logger.info(f'Bot is connected to {len(self.guilds)} guilds')
        
        # Set the bot's status/activity
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="markets | /help"
        )
        await self.change_presence(activity=activity)
    
    async def on_message(self, message):
        """
        Event handler called when a message is received.
        """
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Skip processing for messages that look like commands (they'll be handled by Discord's interaction system)
        if message.content.startswith('/'):
            return
        
        # Get or create user context
        user_id = message.author.id
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        
        # Update context with the latest message time
        self.user_contexts[user_id]["last_seen"] = datetime.now().isoformat()
        
        try:
            # First, use the gatekeeper to decide if this message needs a response
            verdict = await self.gatekeeper.evaluate_message(
                message.content,
                str(user_id),
                self.user_contexts[user_id]
            )
            
            # Log the gatekeeper's decision at INFO level to ensure it's captured
            logger.info(f"Gatekeeper verdict: {verdict.should_respond} ({verdict.confidence:.2f}) - {verdict.reason}")
            # Also log at debug level for more detailed logs
            logger.debug(f"Full gatekeeper verdict details: {verdict}")
            
            # Only process with the main LLM if the gatekeeper approves
            if verdict.should_respond:
                # Show typing indicator while processing
                async with message.channel.typing():
                    # Log user_id type before passing to LLM engine
                    logger.debug(f"Passing user_id to LLM engine: {user_id} (type: {type(user_id).__name__})")
                    logger.debug(f"Context being passed to LLM engine: {list(self.user_contexts[user_id].keys())}")
                    
                    # Call the main LLM engine to process the message
                    response, tool_used = await self.llm_engine.process_message(
                        message.content, 
                        user_id=user_id,  # This is an integer from message.author.id
                        context=self.user_contexts[user_id]
                    )
                    
                    logger.debug(f"Received response from LLM engine: {response[:50]}...")
                    
                    # Send the response - try to reply, but fall back to regular send if permissions are missing
                    try:
                        await message.reply(response)
                    except discord.errors.Forbidden as e:
                        if e.code == 160002:  # Cannot reply without permission to read message history
                            logger.warning("Cannot reply due to missing permissions, using channel.send instead")
                            await message.channel.send(f"{message.author.mention} {response}")
                        else:
                            # Re-raise if it's a different permission issue
                            raise
                    
                    # Update context with interaction info
                    self.user_contexts[user_id]["last_interaction_time"] = datetime.now().isoformat()
                    self.user_contexts[user_id]["last_message"] = message.content
                    
                    # Log the tool usage for analytics
                    if tool_used:
                        logger.info(f"Tool used: {tool_used} by user {message.author.name}")
                        self.user_contexts[user_id]["last_tool"] = tool_used
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Only reply with an error if the gatekeeper thought we should respond
            if 'verdict' in locals() and verdict.should_respond:
                await message.reply("I encountered an error processing your request. Please try again later.")