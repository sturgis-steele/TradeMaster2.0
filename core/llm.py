"""LLM Engine for TradeMaster 2.0

This module implements the core language model engine for the TradeMaster trading assistant.
It provides integration with Groq LLM API to generate responses to user queries about
trading, markets, and financial concepts.
"""

import logging
import os
import json
import aiohttp
import random
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("TradeMaster.LLM")

class LLMEngine:
    """LLM engine for TradeMaster.
    
    This implementation integrates with Groq LLM API to generate
    responses to user queries. It maintains conversation history and provides a consistent
    trading assistant persona through a well-defined system prompt.
    
    Features:
    1. Integration with Groq LLM API
    2. Conversation history management
    3. Trading assistant persona with market knowledge
    4. Fallback mechanisms for API failures
    """
    
    def __init__(self):
        # Initialize API configurations
        self.groq_api_key = os.getenv("GROQ_API_KEY")  # API key for Groq LLM service
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-70b-8192")  # Default model for Groq
        
        # Define system prompt for trading assistant persona
        self.system_prompt = self._get_system_prompt()
        
        # Initialize fallback responses for when API calls fail
        self._init_fallback_responses()
        
        # Log initialization with available APIs
        self._log_initialization()
    
    def _get_system_prompt(self) -> str:
        """Define the system prompt that establishes the assistant's persona.
        
        Returns:
            A comprehensive system prompt string that defines the assistant's identity,
            knowledge areas, and interaction style.
        """
        return """
        You are TradeMaster, an expert trading assistant with deep knowledge of financial markets, 
        trading strategies, and investment concepts. Your purpose is to provide accurate, 
        educational, and actionable insights to traders of all experience levels.
        
        Your areas of expertise include:
        1. Technical analysis (chart patterns, indicators, price action)
        2. Fundamental analysis (economic indicators, financial statements, market news)
        3. Trading psychology and risk management
        4. Market structure and mechanics across different asset classes
        5. Trading strategies and their implementation
        
        When responding to queries:
        - Provide educational content that helps users understand concepts, not just answers
        - Be clear about the limitations of your knowledge and avoid making specific price predictions
        - Emphasize risk management principles and responsible trading practices
        - Adapt your explanations to the user's apparent level of expertise
        - Use precise terminology and explain jargon when necessary
        
        You have access to tools that can provide real-time market data and analysis.
        When appropriate, suggest using these tools to enhance your responses with current information.
        """.strip()
    
    def _init_fallback_responses(self):
        """Initialize fallback responses for when API calls fail.
        
        These responses are used only when external LLM API calls fail, to ensure
        the system can still provide some value to users.
        """
        self.fallback_responses = [
            "I'm having trouble connecting to my knowledge base right now. As a trading assistant, I can tell you that successful trading typically involves a combination of technical analysis, fundamental research, and disciplined risk management. Could you try your question again in a moment?",
            "It seems I'm experiencing a temporary issue accessing my full capabilities. In general, when analyzing markets, it's important to consider multiple timeframes and confirm signals across different indicators. I should be back to normal shortly.",
            "I apologize for the inconvenience, but I'm currently unable to process your request fully. Remember that proper position sizing and risk management are foundational to any successful trading strategy. Please try again soon."
        ]
    
    def _log_initialization(self):
        """Log the initialization status of the LLM Engine."""
        if self.groq_api_key:
            logger.info(f"LLM Engine initialized with Groq API ({self.groq_model})")
        else:
            logger.warning("LLM Engine initialized without Groq API key. Will use fallback responses only.")
    
    async def generate_response(self, message: str, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to a user message using Groq LLM API.
        
        This is the main entry point for processing user queries. It attempts to use
        the Groq LLM API, with fallbacks if API calls fail.
        
        Args:
            message: The user's message text
            user_id: The user's ID for conversation history management
            context: Optional context information containing conversation history
            
        Returns:
            A formatted response string addressing the user's query
        """
        # Log the incoming message
        logger.info(f"Generating response for user {user_id}: {message[:50]}...")
        
        # Get conversation history from context if available
        conversation_history = []
        if context and 'message_history' in context:
            conversation_history = context.get('message_history', [])
        
        # Try Groq API if available
        if self.groq_api_key:
            try:
                response = await self._call_groq_api(message, conversation_history)
                logger.info("Generated response using Groq API")
                return response
            except Exception as e:
                logger.error(f"Groq API call failed: {str(e)}")
                # Use fallback response if API call fails
                logger.warning("API call failed, using fallback response")
                return random.choice(self.fallback_responses)
        else:
            # Use fallback response if no API key is available
            logger.warning("No Groq API key available, using fallback response")
            return random.choice(self.fallback_responses)
    
    async def _call_groq_api(self, message: str, conversation_history: List[Dict[str, str]]) -> str:
        """Call the Groq LLM API to generate a response.
        
        Args:
            message: The user's message
            conversation_history: List of previous messages in the conversation
            
        Returns:
            The generated response text
        
        Raises:
            Exception: If the API call fails
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare messages array with system prompt and conversation history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history - filter out unsupported fields like 'timestamp'
        for msg in conversation_history:
            if 'role' in msg and 'content' in msg:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add the current user message
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": self.groq_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API returned status {response.status}: {error_text}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]