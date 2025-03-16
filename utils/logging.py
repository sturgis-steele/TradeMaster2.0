# TradeMaster Discord Bot - Logging Configuration
# This file provides a centralized logging configuration for the TradeMaster bot.
# It sets up proper logging levels, formats, and handlers for different components.

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    Configure the logging system for the TradeMaster bot.
    Sets up console and file logging with appropriate formatting.
    """
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler with rotation (10 MB max size, keep 5 backup files)
    file_handler = RotatingFileHandler(
        "data/trademaster.log", 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set up specific loggers with appropriate levels
    logging.getLogger('discord').setLevel(logging.INFO)
    logging.getLogger('discord.http').setLevel(logging.WARNING)
    logging.getLogger('discord.gateway').setLevel(logging.INFO)
    
    # Create a logger specifically for our bot
    bot_logger = logging.getLogger("TradeMaster")
    bot_logger.setLevel(logging.DEBUG)  # Capture all bot logs
    
    return bot_logger