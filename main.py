# TradeMaster Discord Bot - Entry Point
# This file serves as the main entry point for the TradeMaster Discord bot.
# It sets up logging, loads environment variables from the .env file,
# initializes the bot client, and starts the Discord connection.
#
# File Interactions:
# - bot/client.py: Imports and instantiates TradeMasterClient
# - config/.env: Loads environment variables and Discord token
# - data/trademaster.log: Writes log output
# - utils/logging.py: Uses logging configuration

import asyncio  # For running asynchronous code (allows the bot to handle multiple tasks at once)
import discord  # For Discord API interactions and error types
import logging  # For creating log files and console output to track what the bot is doing
import os  # For accessing environment variables and file paths
from dotenv import load_dotenv  # For loading API keys and secrets from a .env file

# Import the main Discord bot client class from our bot module
from bot.client import TradeMasterClient

# Set up logging to keep track of what the bot is doing
# This creates both a log file and prints messages to the console
logging.basicConfig(
    level=logging.INFO,  # Only show messages that are 'INFO' level or higher (ignores DEBUG messages)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format: Time - Logger Name - Level - Message
    handlers=[
        logging.FileHandler("data/trademaster.log"),  # Save logs to this file
        logging.StreamHandler()  # Also print logs to the console
    ]
)
# Create a logger specifically for our bot
logger = logging.getLogger("TradeMaster")

# Load secret keys and configuration from the .env file
load_dotenv("config/.env")
# Get the Discord bot token (this is like the bot's password to connect to Discord)
TOKEN = os.getenv("DISCORD_TOKEN")

async def main():
    """The main function that starts the bot"""
    # Create an instance of our Discord bot client
    client = TradeMasterClient()
    
    # Validate the token before attempting to connect
    if not TOKEN or len(TOKEN) < 50:  # Discord tokens are typically longer than 50 chars
        logger.error("Invalid Discord token! Please check your .env file.")
        logger.error("The bot will not be able to connect to Discord without a valid token.")
        logger.error("You may need to regenerate your token in the Discord Developer Portal.")
        return
        
    try:
        # Connect to Discord and start the bot (using the token from our .env file)
        # This is an 'await' call because connecting to Discord happens asynchronously
        logger.info("Attempting to connect to Discord...")
        await client.start(TOKEN)
    except discord.errors.LoginFailure as e:
        logger.error(f"Failed to login to Discord: {e}")
        logger.error("Please check your token in the .env file and ensure it is valid.")
        logger.error("You may need to regenerate your token in the Discord Developer Portal.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while connecting to Discord: {e}")
        logger.error("Please check your internet connection and Discord's status.")

# This is the entry point of the program
# It only runs if this file is executed directly (not imported by another file)
if __name__ == "__main__":
    # Run the main function using asyncio
    # asyncio.run() is the proper way to start an async function from a non-async context
    asyncio.run(main())