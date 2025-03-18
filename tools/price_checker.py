"""
Market Price Checker Tool for TradeMaster 2.0

This tool retrieves current price information for stocks and cryptocurrencies.
It uses both CoinGecko for crypto prices and Alpha Vantage for stock prices.
"""

import logging
import os
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_tool import BaseTool

logger = logging.getLogger("TradeMaster.Tools.PriceChecker")

class PriceCheckerTool(BaseTool):
    """
    Tool for checking current market prices of stocks and cryptocurrencies.
    """
    
    @property
    def name(self) -> str:
        return "price_checker"
    
    @property
    def description(self) -> str:
        return "Checks current prices of stocks and cryptocurrencies"
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "symbol",
                "type": "string",
                "description": "The symbol to check (e.g. BTC, ETH, AAPL, MSFT)",
                "required": True
            },
            {
                "name": "market_type",
                "type": "string",
                "description": "The market type: 'crypto' or 'stock'",
                "required": False,
                "default": "auto"  # Auto-detect based on symbol
            }
        ]
    
    @property
    def examples(self) -> List[str]:
        return [
            "price_checker symbol=BTC",
            "price_checker symbol=AAPL market_type=stock",
            "price_checker symbol=ETH market_type=crypto"
        ]
    
    def __init__(self):
        """Initialize the price checker tool with API keys."""
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.alphavantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Common stock symbols for auto-detection
        self.common_stocks = {
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
            "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "DIS",
            "PYPL", "INTC", "CMCSA", "NFLX", "CSCO", "ADBE", "CRM", "VZ"
        }
        
        # Common crypto symbols for auto-detection
        self.common_cryptos = {
            "BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "AVAX", "DOT", "DOGE",
            "MATIC", "LINK", "UNI", "LTC", "BCH", "ATOM", "XLM", "ALGO", "NEAR"
        }
        
        # CoinGecko ID mapping
        self.coingecko_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOT": "polkadot",
            "DOGE": "dogecoin",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "ATOM": "cosmos",
            "XLM": "stellar",
            "ALGO": "algorand",
            "NEAR": "near"
        }
        
        logger.info("PriceChecker tool initialized")
    
    def _detect_market_type(self, symbol: str) -> str:
        """
        Auto-detect whether a symbol is for crypto or stock.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            Either 'crypto' or 'stock'
        """
        symbol_upper = symbol.upper()
        
        # Check common lists first
        if symbol_upper in self.common_cryptos:
            return "crypto"
        if symbol_upper in self.common_stocks:
            return "stock"
        
        # If not in common lists, use heuristics
        # Crypto symbols tend to be 3-4 letters
        # Stocks can be 1-5 letters and often all caps
        if len(symbol) <= 4 and symbol.isalpha():
            return "crypto"
        else:
            return "stock"
    
    async def _get_crypto_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get cryptocurrency price from CoinGecko.
        
        Args:
            symbol: The crypto symbol (e.g., BTC, ETH)
            
        Returns:
            Dictionary with price information
        """
        symbol_upper = symbol.upper()
        
        # Convert symbol to CoinGecko ID if known
        coin_id = self.coingecko_ids.get(symbol_upper, symbol.lower())
        
        # Set up API URL
        if self.coingecko_api_key:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}"
            headers = {"x-cg-pro-api-key": self.coingecko_api_key}
        else:
            # Fallback to free API (with rate limits)
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            headers = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching crypto price: {response.status}")
                        return {"error": f"Could not fetch price for {symbol}", "status": response.status}
                    
                    data = await response.json()
                    
                    # Extract relevant information
                    price_usd = data.get("market_data", {}).get("current_price", {}).get("usd")
                    price_change_24h = data.get("market_data", {}).get("price_change_percentage_24h")
                    market_cap = data.get("market_data", {}).get("market_cap", {}).get("usd")
                    volume_24h = data.get("market_data", {}).get("total_volume", {}).get("usd")
                    
                    return {
                        "symbol": symbol_upper,
                        "name": data.get("name", "Unknown"),
                        "price_usd": price_usd,
                        "price_change_24h": price_change_24h,
                        "market_cap": market_cap,
                        "volume_24h": volume_24h,
                        "time": datetime.now().isoformat(),
                        "source": "CoinGecko"
                    }
        
        except Exception as e:
            logger.error(f"Error fetching crypto price: {e}")
            return {"error": f"Failed to get price for {symbol}: {str(e)}"}
    
    async def _get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock price from Alpha Vantage.
        
        Args:
            symbol: The stock symbol (e.g., AAPL, MSFT)
            
        Returns:
            Dictionary with price information
        """
        if not self.alphavantage_api_key:
            logger.error("Alpha Vantage API key not found")
            return {"error": "Alpha Vantage API key not configured"}
        
        symbol_upper = symbol.upper()
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol_upper}&apikey={self.alphavantage_api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching stock price: {response.status}")
                        return {"error": f"Could not fetch price for {symbol}", "status": response.status}
                    
                    data = await response.json()
                    
                    # Check for API errors or empty responses
                    if "Error Message" in data:
                        return {"error": data["Error Message"]}
                    
                    if "Global Quote" not in data or not data["Global Quote"]:
                        return {"error": f"No data found for stock symbol {symbol}"}
                    
                    quote = data["Global Quote"]
                    
                    # Extract relevant information
                    price = float(quote.get("05. price", 0))
                    change_percent = quote.get("10. change percent", "0%").replace("%", "")
                    
                    return {
                        "symbol": symbol_upper,
                        "price_usd": price,
                        "price_change": float(quote.get("09. change", 0)),
                        "price_change_percent": float(change_percent),
                        "volume": int(float(quote.get("06. volume", 0))),
                        "latest_trading_day": quote.get("07. latest trading day", ""),
                        "time": datetime.now().isoformat(),
                        "source": "Alpha Vantage"
                    }
        
        except Exception as e:
            logger.error(f"Error fetching stock price: {e}")
            return {"error": f"Failed to get price for {symbol}: {str(e)}"}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the price checker tool.
        
        Args:
            symbol: The symbol to check
            market_type: The market type ('crypto', 'stock', or 'auto')
            
        Returns:
            Dictionary with price information
        """
        symbol = kwargs.get("symbol")
        if not symbol:
            return {"error": "Symbol parameter is required"}
        
        market_type = kwargs.get("market_type", "auto")
        
        # Auto-detect market type if set to "auto"
        if market_type == "auto":
            market_type = self._detect_market_type(symbol)
        
        logger.info(f"Checking price for {symbol} as {market_type}")
        
        # Get price based on market type
        if market_type.lower() == "crypto":
            return await self._get_crypto_price(symbol)
        elif market_type.lower() == "stock":
            return await self._get_stock_price(symbol)
        else:
            return {"error": f"Invalid market type: {market_type}. Use 'crypto' or 'stock'"}