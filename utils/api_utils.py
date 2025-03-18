"""API Utilities for TradeMaster 2.0

This module provides shared utilities for making API calls and handling responses
across different tools in the TradeMaster system.
"""

import logging
import os
import aiohttp
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("TradeMaster.Utils.API")

# API Base URLs
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

async def make_api_request(url: str, headers: Dict[str, str] = None, 
                         params: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Make an API request and handle common error cases.
    
    Args:
        url: The URL to request
        headers: Optional headers to include
        params: Optional query parameters
        
    Returns:
        Tuple of (success, data) where success is a boolean and data is the response
        or error information
    """
    if not headers:
        headers = {}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"API request failed: {response.status} - {url}")
                    return False, {"error": f"API request failed with status {response.status}", 
                                  "status": response.status}
                
                data = await response.json()
                return True, data
    
    except aiohttp.ClientError as e:
        logger.error(f"API connection error: {e} - {url}")
        return False, {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in API request: {e} - {url}")
        return False, {"error": f"Unexpected error: {str(e)}"}

def get_coingecko_url(endpoint: str, use_pro: bool = True) -> Tuple[str, Dict[str, str]]:
    """
    Get the appropriate CoinGecko URL and headers based on whether a Pro API key is available.
    
    Args:
        endpoint: The API endpoint (without leading slash)
        use_pro: Whether to use the Pro API if available
        
    Returns:
        Tuple of (url, headers)
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    
    if api_key and use_pro:
        base_url = COINGECKO_PRO_BASE_URL
        headers = {"x-cg-pro-api-key": api_key}
    else:
        base_url = COINGECKO_BASE_URL
        headers = {}
    
    return f"{base_url}/{endpoint}", headers

def get_alphavantage_params(function: str, symbol: str, **kwargs) -> Dict[str, str]:
    """
    Get parameters for Alpha Vantage API calls.
    
    Args:
        function: The Alpha Vantage function to call
        symbol: The stock symbol
        **kwargs: Additional parameters to include
        
    Returns:
        Dictionary of parameters including the API key
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.warning("Alpha Vantage API key not configured")
    
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key
    }
    
    # Add any additional parameters
    params.update(kwargs)
    
    return params

def format_error_response(symbol: str, error_msg: str) -> Dict[str, Any]:
    """
    Format a standardized error response.
    
    Args:
        symbol: The symbol that was being queried
        error_msg: The error message
        
    Returns:
        Formatted error response dictionary
    """
    return {
        "error": f"Failed to get data for {symbol}: {error_msg}",
        "symbol": symbol,
        "success": False
    }