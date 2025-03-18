"""
Gatekeeper for TradeMaster 2.0

This module determines which messages warrant a response from the bot.
It uses a local LLM (Ollama with Gemma 3 1B) to intelligently filter messages.
"""

import logging
import os
import re
import json
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from pydantic import BaseModel
import aiohttp

logger = logging.getLogger("TradeMaster.Gatekeeper")

class MessageVerdict(BaseModel):
    """Represents the verdict on whether a message needs a response."""
    should_respond: bool
    confidence: float
    reason: str

class Gatekeeper:
    """
    Determines which messages warrant a response from the bot.
    Uses Ollama with Gemma 3 1B for intelligent message filtering.
    """
    
    def __init__(self, model_name="gemma3:1b", ollama_base_url=None, timeout=10):
        """
        Initialize the gatekeeper.
        
        Args:
            model_name: The name of the Ollama model to use
            ollama_base_url: Base URL for Ollama API
            timeout: Timeout in seconds for Ollama API calls
        """
        # Set up Ollama configuration
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        self.ollama_available = False  # Track if Ollama is available
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Prompt template for the gatekeeper
        self.system_prompt = """
        You are a message filter for a trading assistant Discord bot named TradeMaster. Your task is to determine if a message needs a response from the main bot's AI.

        RESPOND TO:
        1. ANY message that mentions trading, investing, markets, finance, stocks, crypto, or financial terms
        2. ANY direct questions about trading, investing, or financial advice
        3. ANY message that explicitly mentions or addresses the bot (TradeMaster, bot, assistant)
        4. ANY message continuing a conversation about trading/finance topics
        5. ANY message asking for price predictions, market updates, or trade recommendations

        DO NOT RESPOND TO:
        1. General chat completely unrelated to finance, trading, or the bot
        2. Random messages with no clear question or intent related to trading
        3. Non-sequiturs or spam messages

        IMPORTANT KEYWORDS TO DETECT (always respond to messages containing these):
        - trade, trading, trades
        - invest, investing, investment
        - stock, stocks, etf, fund, mutual fund
        - crypto, cryptocurrency, bitcoin, ethereum, btc, eth
        - market, markets, price, pricing
        - analysis, technical, fundamental
        - strategy, strategies, buy, sell
        - portfolio, position, holdings
        - profit, loss, gain, risk
        - chart, pattern, trend

        ALWAYS respond to messages that ask "what are the best trades" or any variation of this.

        You must respond with exactly ONE line containing:
        - "YES" or "NO" (whether the bot should respond)
        - A confidence score between 0.0 and 1.0
        - A brief reason for your decision

        Example correct responses:
        YES 0.95 Contains trading terms ("best trades today")
        YES 0.90 Direct question about trading/investing
        YES 0.85 Mentions crypto prices
        YES 0.80 Continues a conversation about financial markets
        NO 0.90 General greeting unrelated to trading or finance
        """
        
        # Track channel response times to prevent excessive responses
        self.last_response_time = {}
        
        # Initialize trading keywords for fallback mechanism
        self._init_keywords()
        
        # Check Ollama connection (async)
        asyncio.create_task(self._check_ollama_connection())
        
        logger.info(f"Ollama Gatekeeper initialized with model '{model_name}'")
    
    def _init_keywords(self):
        """Initialize keyword sets for fallback mechanism."""
        # Trading and investing terms
        self.trading_keywords = {
            'stock', 'trade', 'trades', 'trading', 'invest', 'investing', 'investor', 'portfolio',
            'market', 'buy', 'sell', 'long', 'short', 'position', 'entry', 'exit',
            'analysis', 'chart', 'pattern', 'breakout', 'support', 'resistance', 
            'bullish', 'bearish', 'trend', 'uptrend', 'downtrend', 'backtest',
            'strategy', 'risk', 'reward', 'profit', 'loss', 'gain', 'roi', 'return',
            'crypto', 'bitcoin', 'btc', 'ethereum', 'eth', 'altcoin', 'token', 'coin',
            'blockchain', 'wallet', 'address', 'exchange', 'dex', 'defi', 'nft',
            'mining', 'staking', 'yield', 'dao', 'smart contract', 'gas', 'gwei',
            'hodl', 'fud', 'fomo', 'whale', 'dump', 'pump', 'moon', 'lambo',
            'price', 'prediction', 'forecast', 'outlook', 'recommendation',
            'best', 'worst', 'today', 'tomorrow', 'etf', 'fund', 'mutual'
        }
        
        # Key phrases (checked separately from individual keywords)
        self.key_phrases = [
            'best trades', 'trading advice', 'should i buy', 'should i sell',
            'what stocks', 'which crypto', 'price prediction', 'market analysis',
            'investment advice', 'portfolio advice', 'trading strategy'
        ]
        
        # Direct address terms
        self.direct_address = {
            'trademaster', 'bot', 'assistant', 'help', 'hey', 'hello', 'hi', 'ai'
        }
    
    async def _check_ollama_connection(self):
        """Check if Ollama is available and responding."""
        url = f"{self.ollama_base_url}/api/version"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        self.ollama_available = True
                        logger.info(f"Successfully connected to Ollama at {self.ollama_base_url}")
                        # Check if model is available
                        model_url = f"{self.ollama_base_url}/api/tags"
                        async with session.get(model_url) as model_response:
                            if model_response.status == 200:
                                models_data = await model_response.json()
                                models = [m.get('name') for m in models_data.get('models', [])]
                                if self.model_name in models:
                                    logger.info(f"Model '{self.model_name}' is available in Ollama")
                                else:
                                    logger.warning(f"Model '{self.model_name}' not found in Ollama. Available models: {models}")
                                    logger.warning(f"You may need to run: ollama pull {self.model_name}")
                    else:
                        self.ollama_available = False
                        logger.warning(f"Ollama is running but returned status {response.status}")
        except Exception as e:
            self.ollama_available = False
            logger.warning(f"Failed to connect to Ollama: {e}")
            logger.warning("Make sure Ollama is running (ollama serve) and the model is pulled (ollama pull gemma3:1b)")
    
    async def _call_ollama(self, messages):
        """
        Make an async call to the Ollama API with retries.
        
        Args:
            messages: List of message dictionaries for the conversation
            
        Returns:
            The model's response text or None if failed
        """
        if not self.ollama_available and self.connection_attempts >= self.max_connection_attempts:
            # Skip if we've already failed multiple times
            return None
            
        url = f"{self.ollama_base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent decisions
                "num_predict": 100   # Limit response length
            }
        }
        
        try:
            self.connection_attempts += 1
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.timeout) as response:
                    if response.status != 200:
                        logger.error(f"Ollama API error: {response.status} - {await response.text()}")
                        return None
                    
                    result = await response.json()
                    
                    # If we get here, Ollama is working
                    self.ollama_available = True
                    self.connection_attempts = 0  # Reset counter on success
                    
                    return result.get("message", {}).get("content", "")
        except asyncio.TimeoutError:
            logger.error(f"Timeout while calling Ollama API (timeout={self.timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None
    
    def _fallback_evaluation(self, message, context, bot_mentioned):
        """
        Fallback method using heuristics when Ollama is unavailable.
        
        Args:
            message: The message content
            context: User context dictionary
            bot_mentioned: Whether the bot was mentioned
            
        Returns:
            MessageVerdict object
        """
        # If bot was directly mentioned, always respond
        if bot_mentioned:
            return MessageVerdict(
                should_respond=True,
                confidence=0.95,
                reason="Fallback: Bot was directly mentioned"
            )
        
        # Simple keyword check
        message_lower = message.lower()
        words = set(re.findall(r'\b\w+\b', message_lower))
        
        # Check for direct address of the bot
        direct_address_match = bool(words.intersection(self.direct_address))
        
        # Check for trading keywords
        trading_match = bool(words.intersection(self.trading_keywords))
        
        # Check for key phrases (exact matches)
        phrase_match = False
        matched_phrase = ""
        for phrase in self.key_phrases:
            if phrase in message_lower:
                phrase_match = True
                matched_phrase = phrase
                break
        
        # Check for questions
        is_question = "?" in message or any(q_word in message_lower.split() for q_word in 
                                            ["what", "how", "when", "why", "where", "who", "which"])
        
        # Special case for "best trades" or similar queries
        best_trades_match = "best trade" in message_lower or "best stock" in message_lower or "best crypto" in message_lower
        
        # Recent interaction increases likelihood of response
        recent_interaction = False
        if "last_interaction_time" in context:
            try:
                last_time = datetime.fromisoformat(context["last_interaction_time"])
                if (datetime.now() - last_time).total_seconds() < 300:  # 5 minutes
                    recent_interaction = True
            except (ValueError, TypeError):
                pass
        
        # Determine if we should respond
        should_respond = (
            bot_mentioned or 
            direct_address_match or 
            phrase_match or
            best_trades_match or
            (trading_match and (is_question or recent_interaction))
        )
        
        if should_respond:
            confidence = 0.9 if (phrase_match or best_trades_match) else 0.8 if (trading_match or direct_address_match) else 0.6
            reasons = []
            if direct_address_match:
                reasons.append("directly addressed the bot")
            if trading_match:
                reasons.append("contains trading terminology")
            if phrase_match:
                reasons.append(f"contains key trading phrase '{matched_phrase}'")
            if best_trades_match:
                reasons.append("asking about best trades/investments")
            if is_question:
                reasons.append("asks a question")
            if recent_interaction:
                reasons.append("continues recent conversation")
                
            reason = "Fallback: " + ", ".join(reasons)
        else:
            confidence = 0.7
            reason = "Fallback: Message does not appear to be trading related or directed at the bot"
        
        return MessageVerdict(
            should_respond=should_respond,
            confidence=confidence,
            reason=reason
        )
    
    async def should_respond(self, message: str, user_id: str, 
                             bot_mentioned: bool = False,
                             context: Optional[Dict[str, Any]] = None) -> MessageVerdict:
        """
        Determine if a message should get a response.
        
        Args:
            message: The user's message
            user_id: The user's ID
            bot_mentioned: Whether the bot was mentioned
            context: Optional context information
            
        Returns:
            A MessageVerdict object
        """
        if context is None:
            context = {}
        
        # Always respond to direct mentions
        if bot_mentioned:
            return MessageVerdict(
                should_respond=True,
                confidence=0.98,
                reason="Bot was directly mentioned"
            )
        
        # Rate limiting - don't respond too frequently in the same channel
        channel_id = context.get("channel_id", "unknown")
        current_time = datetime.now()
        
        if channel_id in self.last_response_time:
            seconds_since_last = (current_time - self.last_response_time[channel_id]).total_seconds()
            if seconds_since_last < 5:  # 5 second cooldown
                return MessageVerdict(
                    should_respond=False,
                    confidence=0.9,
                    reason=f"Rate limiting: responded in this channel {seconds_since_last:.1f} seconds ago"
                )
        
        # If we've determined Ollama is unavailable after multiple attempts, use fallback immediately
        if not self.ollama_available and self.connection_attempts >= self.max_connection_attempts:
            logger.debug("Using fallback directly due to previous Ollama connection failures")
            return self._fallback_evaluation(message, context, bot_mentioned)
        
        # Format context for the LLM
        recent_history = context.get("message_history", [])[-5:] if "message_history" in context else []
        history_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_history])
        
        try:
            # Prepare messages for Ollama
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
                USER MESSAGE: {message}
                
                USER ID: {user_id}
                
                RECENT CONVERSATION:
                {history_text}
                
                Should TradeMaster respond to this message? Reply with YES/NO, confidence score, and reason on a single line.
                """}
            ]
            
            # Call Ollama API
            response = await self._call_ollama(messages)
            
            if not response:
                logger.warning("Failed to get response from Ollama, falling back to heuristics")
                return self._fallback_evaluation(message, context, bot_mentioned)
            
            # Parse the response
            # Expected format: "YES/NO 0.XX reason text here"
            response = response.strip()
            
            # Extract decision, confidence, and reason
            match = re.match(r'^(YES|NO)\s+(\d+\.\d+)\s+(.+)$', response, re.IGNORECASE)
            
            if match:
                decision = match.group(1).upper() == "YES"
                try:
                    confidence = float(match.group(2))
                    confidence = max(0.0, min(1.0, confidence))  # Ensure between 0 and 1
                except ValueError:
                    confidence = 0.5  # Default if conversion fails
                
                reason = match.group(3)
            else:
                # If response doesn't match expected format, use smarter parsing
                decision = "YES" in response.upper() and not ("NO" in response.upper() and response.upper().index("NO") < response.upper().index("YES"))
                
                # Try to extract a confidence score if present
                confidence_match = re.search(r'(\d+\.\d+)', response)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                
                # Use the whole response as the reason
                reason = response
            
            # Create verdict
            verdict = MessageVerdict(
                should_respond=decision,
                confidence=confidence,
                reason=reason
            )
            
            # Update last response time if we're responding
            if decision:
                self.last_response_time[channel_id] = current_time
            
            logger.info(f"Ollama verdict: {'RESPOND' if decision else 'IGNORE'} ({confidence:.2f}) - {reason[:100]}...")
            return verdict
            
        except Exception as e:
            logger.error(f"Error using Ollama for message filtering: {e}")
            # Fall back to simple heuristics
            return self._fallback_evaluation(message, context, bot_mentioned)