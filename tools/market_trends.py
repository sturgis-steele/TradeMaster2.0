"""
Market Trends Tool for TradeMaster 2.0

This tool retrieves market trend data and top movers for both
stock and cryptocurrency markets.
"""

import logging
import os
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from .base_tool import BaseTool

logger = logging.getLogger("TradeMaster.Tools.MarketTrends")

class MarketTrendsTool(BaseTool):
    """
    Tool for getting market trends, top gainers, and top losers.
    """
    
    @property
    def name(self) -> str:
        return "market_trends"
    
    @property
    def description(self) -> str:
        return "Gets market trends, top gainers, and top losers for crypto or stocks"
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "market_type",
                "type": "string",
                "description": "The market type: 'crypto' or 'stock'",
                "required": True
            },
            {
                "name": "category",
                "type": "string",
                "description": "Category to retrieve: 'gainers', 'losers', or 'trending'",
                "required": False,
                "default": "trending"
            },
            {
                "name": "limit",
                "type": "number",
                "description": "Number of results to return (1-25)",
                "required": False,
                "default": 5
            }
        ]
    
    @property
    def examples(self) -> List[str]:
        return [
            "market_trends market_type=crypto",
            "market_trends market_type=stock category=gainers",
            "market_trends market_type=crypto category=losers limit=10"
        ]
    
    def __init__(self):
        """Initialize the market trends tool with API keys."""
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.alphavantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        logger.info("MarketTrends tool initialized")
    
    async def _get_crypto_trends(self, category: str, limit: int) -> Dict[str, Any]:
        """
        Get cryptocurrency trends from CoinGecko.
        
        Args:
            category: The category to retrieve ('gainers', 'losers', or 'trending')
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with trend information
        """
        # Ensure limit is reasonable
        limit = max(1, min(25, limit))
        
        # For trending coins
        if category == "trending":
            # Set up API URL
            if self.coingecko_api_key:
                url = "https://pro-api.coingecko.com/api/v3/search/trending"
                headers = {"x-cg-pro-api-key": self.coingecko_api_key}
            else:
                # Fallback to free API (with rate limits)
                url = "https://api.coingecko.com/api/v3/search/trending"
                headers = {}
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Error fetching crypto trends: {response.status}")
                            return {"error": "Could not fetch trending cryptos", "status": response.status}
                        
                        data = await response.json()
                        
                        # Extract trending coins
                        trending_coins = data.get("coins", [])
                        
                        # Format results
                        results = []
                        for i, coin_data in enumerate(trending_coins[:limit]):
                            coin = coin_data.get("item", {})
                            results.append({
                                "rank": i + 1,
                                "symbol": coin.get("symbol", "").upper(),
                                "name": coin.get("name", "Unknown"),
                                "market_cap_rank": coin.get("market_cap_rank"),
                                "price_btc": coin.get("price_btc"),
                                "id": coin.get("id")
                            })
                        
                        return {
                            "category": "trending",
                            "market_type": "crypto",
                            "results": results,
                            "time": datetime.now().isoformat(),
                            "source": "CoinGecko"
                        }
            
            except Exception as e:
                logger.error(f"Error fetching crypto trends: {e}")
                return {"error": f"Failed to get trending cryptos: {str(e)}"}
        
        # For gainers and losers
        else:
            # Set up API URL for market data
            if self.coingecko_api_key:
                url = "https://pro-api.coingecko.com/api/v3/coins/markets"
                headers = {"x-cg-pro-api-key": self.coingecko_api_key}
            else:
                # Fallback to free API (with rate limits)
                url = "https://api.coingecko.com/api/v3/coins/markets"
                headers = {}
            
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,  # Get a good sample to find gainers/losers
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h"
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status != 200:
                            logger.error(f"Error fetching crypto market data: {response.status}")
                            return {"error": "Could not fetch crypto market data", "status": response.status}
                        
                        data = await response.json()
                        
                        # Sort by price change percentage
                        if category == "gainers":
                            sorted_data = sorted(data, key=lambda x: x.get("price_change_percentage_24h", 0), reverse=True)
                        else:  # losers
                            sorted_data = sorted(data, key=lambda x: x.get("price_change_percentage_24h", 0))
                        
                        # Format results
                        results = []
                        for i, coin in enumerate(sorted_data[:limit]):
                            results.append({
                                "rank": i + 1,
                                "symbol": coin.get("symbol", "").upper(),
                                "name": coin.get("name", "Unknown"),
                                "price_usd": coin.get("current_price"),
                                "price_change_24h": coin.get("price_change_24h"),
                                "price_change_percentage_24h": coin.get("price_change_percentage_24h"),
                                "market_cap": coin.get("market_cap"),
                                "volume_24h": coin.get("total_volume")
                            })
                        
                        return {
                            "category": category,
                            "market_type": "crypto",
                            "results": results,
                            "time": datetime.now().isoformat(),
                            "source": "CoinGecko"
                        }
            
            except Exception as e:
                logger.error(f"Error fetching crypto market data: {e}")
                return {"error": f"Failed to get {category} cryptos: {str(e)}"}
    
    async def _get_stock_trends(self, category: str, limit: int) -> Dict[str, Any]:
        """
        Get stock market trends from Alpha Vantage.
        
        Args:
            category: The category to retrieve ('gainers', 'losers', or 'trending')
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with trend information
        """
        if not self.alphavantage_api_key:
            logger.error("Alpha Vantage API key not found")
            return {"error": "Alpha Vantage API key not configured"}
        
        # Ensure limit is reasonable
        limit = max(1, min(25, limit))
        
        # Map category to Alpha Vantage function
        function = "TOP_GAINERS_LOSERS"
        
        url = f"https://www.alphavantage.co/query?function={function}&apikey={self.alphavantage_api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Error fetching stock trends: {response.status}")
                        return {"error": f"Could not fetch stock trends", "status": response.status}
                    
                    data = await response.json()
                    
                    # Check for API errors or empty responses
                    if "Error Message" in data:
                        return {"error": data["Error Message"]}
                    
                    # Get the appropriate category from response
                    if category == "gainers":
                        category_data = data.get("top_gainers", [])
                    elif category == "losers":
                        category_data = data.get("top_losers", [])
                    else:  # trending - Alpha Vantage doesn't have a direct trending endpoint, so use most active
                        category_data = data.get("most_actively_traded", [])
                    
                    # Format results
                    results = []
                    for i, stock in enumerate(category_data[:limit]):
                        change_percent = stock.get("change_percentage", "0%").replace("%", "")
                        try:
                            change_percent_float = float(change_percent)
                        except:
                            change_percent_float = 0.0
                            
                        results.append({
                            "rank": i + 1,
                            "symbol": stock.get("ticker", ""),
                            "price_usd": float(stock.get("price", 0)),
                            "change_amount": float(stock.get("change_amount", 0)),
                            "change_percentage": change_percent_float,
                            "volume": int(stock.get("volume", 0))
                        })
                    
                    return {
                        "category": category,
                        "market_type": "stock",
                        "results": results,
                        "time": datetime.now().isoformat(),
                        "source": "Alpha Vantage"
                    }
        
        except Exception as e:
            logger.error(f"Error fetching stock trends: {e}")
            return {"error": f"Failed to get {category} stocks: {str(e)}"}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the market trends tool.
        
        Args:
            market_type: The market type ('crypto' or 'stock')
            category: Category to retrieve ('gainers', 'losers', or 'trending')
            limit: Number of results to return (1-25)
            
        Returns:
            Dictionary with trend information
        """
        market_type = kwargs.get("market_type")
        if not market_type:
            return {"error": "market_type parameter is required"}
        
        category = kwargs.get("category", "trending")
        limit = int(kwargs.get("limit", 5))
        
        # Validate category
        valid_categories = ["gainers", "losers", "trending"]
        if category not in valid_categories:
            return {"error": f"Invalid category: {category}. Valid categories: {', '.join(valid_categories)}"}
        
        logger.info(f"Getting {category} for {market_type} market (limit: {limit})")
        
        # Get trends based on market type
        if market_type.lower() == "crypto":
            return await self._get_crypto_trends(category, limit)
        elif market_type.lower() == "stock":
            return await self._get_stock_trends(category, limit)
        else:
            return {"error": f"Invalid market type: {market_type}. Use 'crypto' or 'stock'"}