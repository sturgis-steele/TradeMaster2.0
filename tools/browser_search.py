"""Browser Search Tool for TradeMaster 2.0

This tool enables the TradeMaster bot to search and extract information from websites
using browser automation. It complements the API-based tools by providing access to
data that might not be available through the current APIs.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_tool import BaseTool
# Removed import from api_utils

# Import browser-use library
try:
    from browser_use import Agent as BrowserAgent
    from langchain_openai import ChatOpenAI
    import openai
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

logger = logging.getLogger("TradeMaster.Tools.BrowserSearch")

class BrowserSearchTool(BaseTool):
    """
    Tool for searching and extracting information from websites using browser automation.
    
    This tool leverages the browser-use library to control a browser and extract
    information from websites. It can be used to access data that is not available
    through the current API integrations, such as:
    
    - Real-time market data from financial websites
    - News and analysis from financial news sites
    - Social media sentiment about specific assets
    - Alternative data sources not covered by existing APIs
    
    The tool uses a headless browser to navigate to websites, interact with them,
    and extract the requested information. It can handle complex web interactions
    like form filling, button clicking, and data extraction from dynamic content.
    
    This provides a powerful complement to the existing API-based tools, allowing
    the TradeMaster bot to access a wider range of data sources.
    """
    
    @property
    def name(self) -> str:
        return "browser_search"
    
    @property
    def description(self) -> str:
        return "Searches and extracts information from websites using browser automation"
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "query",
                "type": "string",
                "description": "The search query or specific task to perform (e.g., 'Compare prices of BTC on Coinbase and Binance')",
                "required": True
            },
            {
                "name": "max_steps",
                "type": "number",
                "description": "Maximum number of browser interaction steps to perform (default: 10)",
                "required": False,
                "default": 10
            }
        ]
    
    @property
    def examples(self) -> List[str]:
        return [
            "browser_search query='Compare the price of BTC on Coinbase and Binance'",
            "browser_search query='Find the latest news about Ethereum'",
            "browser_search query='Check the current gas fees on Ethereum network'"
        ]
    
    def __init__(self):
        """Initialize the browser search tool."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        logger.info("BrowserSearch tool initialized with Groq API")
    
    def format_error_response(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Format a standardized error response."""
        return {
            "error": f"Failed to get data for {symbol}: {error_msg}",
            "symbol": symbol,
            "success": False
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the browser search tool.
        
        Args:
            query: The search query or specific task to perform
            max_steps: Maximum number of browser interaction steps to perform
            
        Returns:
            Dictionary with search results
        """
        # Check if browser-use is available
        if not BROWSER_USE_AVAILABLE:
            logger.error("browser-use library is not installed")
            return {
                "error": "Failed to get data for browser_search: The browser-use library is not installed. Please install it with 'pip install browser-use' and run 'playwright install'.",
                "symbol": "browser_search",
                "success": False
            }
        
        # Check if Playwright browsers are installed
        try:
            import playwright.async_api
            # Try to create a simple browser context to check if browsers are installed
            # Use async context manager for async API
            try:
                # Create a coroutine to check if browsers are installed
                async def check_browser_installation():
                    async with playwright.async_api.async_playwright() as p:
                        try:
                            browser = await p.chromium.launch(headless=True)
                            await browser.close()
                            return True
                        except Exception as e:
                            error_msg = str(e)
                            if "Executable doesn't exist" in error_msg or "Please run the following command to download new browsers" in error_msg:
                                logger.error("Playwright browsers are not installed")
                                return False
                            else:
                                # Some other error with launching the browser
                                logger.error(f"Error launching Playwright browser: {e}")
                                return False
                
                # Run the coroutine to check browser installation
                if not await check_browser_installation():
                    return {
                        "error": "Failed to get data for browser_search: Playwright browsers are not installed. Please run 'playwright install' or 'python -m playwright install' to download the required browsers.",
                        "symbol": "browser_search",
                        "success": False
                    }
            except Exception as e:
                logger.error(f"Error checking Playwright installation: {e}")
                return {
                    "error": f"Failed to get data for browser_search: Error checking Playwright installation: {str(e)}",
                    "symbol": "browser_search",
                    "success": False
                }
        except ImportError:
            # If playwright is not installed, this will be caught by the BROWSER_USE_AVAILABLE check above
            pass
        
        # Check if Groq API key is available
        if not self.groq_api_key:
            logger.error("Groq API key not found")
            return {
                "error": "Failed to get data for browser_search: Groq API key not configured. Please add GROQ_API_KEY to your environment variables.",
                "symbol": "browser_search",
                "success": False
            }
        
        query = kwargs.get("query")
        if not query:
            return {"error": "query parameter is required"}
        
        max_steps = int(kwargs.get("max_steps", 10))
        
        logger.info(f"Executing browser search with query: {query}")
        
        try:
            # Initialize the browser agent with Groq API (OpenAI compatible)
            # Configure the OpenAI client for Groq
            openai_client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.groq_api_key
            )
            
            # Create a ChatOpenAI instance from langchain_openai instead of using OpenAI client directly
            # This provides the necessary methods expected by BrowserAgent
            from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
            
            # Define a custom message formatter to ensure content is always a string
            def groq_message_formatter(messages):
                formatted_messages = []
                for message in messages:
                    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                        # Ensure content is always a string
                        if message.content is None:
                            content = ""
                        elif isinstance(message.content, str):
                            content = message.content
                        else:
                            # Convert non-string content to string
                            content = str(message.content)
                        
                        # Create a properly formatted message dict
                        formatted_message = {
                            "role": message.type,
                            "content": content
                        }
                        formatted_messages.append(formatted_message)
                    else:
                        # Handle dict-style messages
                        formatted_message = dict(message)
                        if "content" in formatted_message and formatted_message["content"] is not None:
                            if not isinstance(formatted_message["content"], str):
                                formatted_message["content"] = str(formatted_message["content"])
                        else:
                            formatted_message["content"] = ""
                        formatted_messages.append(formatted_message)
                return formatted_messages
            
            chat_model = ChatOpenAI(
                model="llama3-70b-8192",  # Use an appropriate Groq model
                openai_api_key=self.groq_api_key,
                openai_api_base="https://api.groq.com/openai/v1",
                # Add custom message formatter to ensure content is always a string
                message_formatter=groq_message_formatter
            )
            
            # Create the browser agent with proper async configuration
            # Initialize browser with proper configuration
            from browser_use.browser.browser import Browser, BrowserConfig
            
            # Create a browser instance first
            browser = Browser(
                config=BrowserConfig(
                    headless=False,  # Set to False for visibility, True for headless mode
                    disable_security=True  # Disable security for better compatibility
                )
            )
            
            # Create the agent with the browser instance
            agent = BrowserAgent(
                task=query,
                llm=chat_model,  # Use the ChatOpenAI instance with message formatter
                browser=browser  # Pass the browser instance instead of browser_config
            )
                
            # Run the agent asynchronously
            # The browser-use library's Agent.run() method is already async
            # so we can await it directly
            result = await agent.run()
            
            # Format the response
            return {
                "query": query,
                "result": result,
                "time": datetime.now().isoformat(),
                "source": "browser-use"
            }
            
        except Exception as e:
            logger.error(f"Error executing browser search: {e}")
            return {
                "error": f"Failed to get data for browser_search: Error executing browser search: {str(e)}",
                "symbol": "browser_search",
                "success": False
            }