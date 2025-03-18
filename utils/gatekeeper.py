# TradeMaster Discord Bot - Gatekeeper
# This file implements a message filter that determines which Discord messages
# warrant the attention of the main LLM. It uses a lightweight local model
# to save costs by only forwarding relevant messages.
#
# File Interactions:
# - bot/client.py: Imports and uses Gatekeeper for message filtering
# - main.py: Indirectly connected through bot/client.py
# - Ollama API: Connects to local Ollama instance for lightweight inference
# - LangChain: Uses LangChain library for LLM integration

import logging
from typing import Dict, Any, Optional, List
import asyncio
from pydantic import BaseModel, Field
import re
import requests
import os

# LangChain imports for integrating with Ollama (local LLM)
try:
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Falling back to heuristic evaluation.")

# Set up logging for this module
logger = logging.getLogger("TradeMaster.Gatekeeper")

class MessageVerdict(BaseModel):
    """
    The verdict on whether a message needs response.
    This class defines the structure of the decision made about each message.
    """
    should_respond: bool = Field(description="Whether the message requires a response")
    confidence: float = Field(description="Confidence level (0.0-1.0)")
    reason: str = Field(description="Explanation for the decision")

class Gatekeeper:
    """
    The Gatekeeper analyzes all Discord messages to determine if they warrant
    the attention of the main LLM. This saves costs by only forwarding relevant messages.
    
    It uses a local LLM (Ollama) to make intelligent filtering decisions, with
    fallback to heuristic evaluation if Ollama is unavailable.
    """
    
    def __init__(self, model_name="gemma3:1b", ollama_base_url="http://localhost:11434"):
        """
        Initialize the gatekeeper with a lightweight local model.
        
        Args:
            model_name: The name of the Ollama model to use (default: "gemma3:1b")
            ollama_base_url: The base URL for the Ollama API (default: "http://localhost:11434")
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.chain = None
        self.ollama = None
        
        # Store the available models from Ollama (if reachable)
        self.available_models: List[str] = []
        
        # Check if LangChain is available
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain is not available. Gatekeeper will use heuristic evaluation only.")
            return
        
        # Try to initialize Ollama and get available models
        try:
            # Check if Ollama is running and get available models
            if self._check_ollama_connection():
                self._initialize_ollama()
            else:
                logger.warning(f"Ollama not available at {ollama_base_url}. Using fallback heuristics.")
        except Exception as e:
            logger.error(f"Error during Gatekeeper initialization: {e}")
            
        logger.info("Gatekeeper initialized")
    
    def _check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and get available models.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            # Try to connect to Ollama
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                # Get available models
                models_data = response.json()
                if "models" in models_data:
                    self.available_models = [model["name"] for model in models_data["models"]]
                else:
                    # Older Ollama API returns different format
                    self.available_models = [model["name"] for model in models_data]
                
                logger.info(f"Available Ollama models: {', '.join(self.available_models)}")
                return True
            else:
                logger.warning(f"Failed to get models from Ollama: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
            return False
    
    def _initialize_ollama(self) -> None:
        """
        Initialize the Ollama model with LangChain.
        
        This method initializes the Ollama model and creates a prompt template
        for message evaluation.
        """
        try:
            # Check if the requested model is available
            if self.model_name not in self.available_models:
                # Try to find an alternative model
                alternative_models = ["gemma", "mistral", "llama", "tinyllama", "phi"]
                for model_prefix in alternative_models:
                    alternative = next((m for m in self.available_models if model_prefix in m.lower()), None)
                    if alternative:
                        logger.info(f"Requested model '{self.model_name}' not available. Using '{alternative}' instead.")
                        self.model_name = alternative
                        break
                else:
                    # If no alternative is found, use the first available model
                    if self.available_models:
                        self.model_name = self.available_models[0]
                        logger.info(f"Using available model: {self.model_name}")
                    else:
                        logger.warning("No models available in Ollama. Using fallback heuristics.")
                        return
            
            # Create an instance of the Ollama model
            self.ollama = OllamaLLM(model=self.model_name, base_url=self.ollama_base_url)
            
            # Create a prompt template that instructs the model how to evaluate messages
            self.prompt_template = PromptTemplate(
                input_variables=["message", "context"],
                template="""
                You are a message filter for a trading assistant bot in a Discord server.
                
                Your task is to determine if the following message requires a response from the main AI.
                Only messages related to trading, investing, markets, cryptocurrencies, or financial topics
                should be forwarded to the main AI. General chit-chat or unrelated discussions should be filtered out.
                
                Message: {message}
                
                User context: {context}
                
                Decide if this message requires the trading bot's attention.
                
                Important guidelines:
                - Messages directly addressing the bot should get a response
                - Questions about trading, investing, or finance should get a response
                - Discussions containing multiple trading terms likely need a response
                - If the user recently interacted with the bot, consider continuing the conversation
                - Filter out off-topic conversation or general chat not related to trading
                
                Answer with YES or NO, followed by a confidence score (0.0-1.0) and a brief reason.
                Example: "YES 0.85 reason: This is a direct question about stock options pricing"
                """
            )
            
            # Create the chain using RunnableSequence pattern
            self.chain = self.prompt_template | self.ollama
            logger.info(f"Ollama model '{self.model_name}' initialized successfully")
        except Exception as e:
            # Log the error if Ollama initialization fails
            logger.error(f"Failed to initialize Ollama model: {e}")
            # Set chain to None to indicate initialization failed
            self.chain = None
    
    def _simple_heuristic_evaluation(self, message: str, context: Dict[str, Any]) -> MessageVerdict:
        """
        A simple rule-based fallback when Ollama is unavailable.
        Uses basic heuristics to determine if a message needs a response.
        
        Args:
            message: The message content to evaluate
            context: The user's conversation context
            
        Returns:
            A MessageVerdict with the decision
        """
        message_lower = message.lower()
        
        # Keywords that suggest trading/finance topics
        finance_keywords = [
            "trade", "trading", "stock", "crypto", "bitcoin", "eth", "market", 
            "invest", "portfolio", "chart", "price", "bull", "bear", "token", 
            "coin", "exchange", "buy", "sell", "profit", "loss", "analysis",
            "indicator", "candle", "trend", "support", "resistance", "volume",
            "dividend", "etf", "fund", "option", "futures", "forex", "hedge",
            "leverage", "margin", "liquidity", "volatility", "arbitrage"
        ]
        
        # Check if message directly addresses the bot
        direct_address = any(term in message_lower for term in ["bot", "trademaster", "assistant", "@trademaster"])
        
        # Check if message contains finance-related keywords
        finance_related = any(keyword in message_lower for keyword in finance_keywords)
        
        # Check if message is a question
        is_question = "?" in message or any(q_word in message_lower.split() for q_word in ["what", "how", "when", "why", "where", "who", "which"])
        
        # Recent interaction suggests continuing the conversation
        recent_interaction = "last_interaction_time" in context
        
        # Determine if we should respond based on these factors
        should_respond = direct_address or finance_related or (is_question and (finance_related or recent_interaction))
        
        # Calculate confidence based on the number of signals
        confidence_signals = sum([direct_address, finance_related, is_question and recent_interaction])
        confidence = min(0.5 + (confidence_signals * 0.15), 0.95)  # Scale between 0.5 and 0.95
        
        # Determine reason based on the triggered conditions
        if direct_address:
            reason = "User directly addressed the bot"
        elif finance_related:
            reason = "Message contains finance/trading related keywords"
        elif is_question and recent_interaction:
            reason = "Question in an ongoing conversation"
        else:
            reason = "Message does not appear to require the bot's attention"
        
        return MessageVerdict(
            should_respond=should_respond,
            confidence=confidence,
            reason=reason
        )
    
    async def evaluate_message(self, message: str, user_id: str, 
                               context: Dict[str, Any]) -> MessageVerdict:
        """
        Evaluate a Discord message to determine if it requires a response.
        This method uses the Ollama model to make an intelligent decision.
        If Ollama is unavailable, falls back to simple heuristics.
        
        Args:
            message: The message content to evaluate
            user_id: The Discord user ID who sent the message
            context: The user's conversation context (history and metadata)
        
        Returns:
            A MessageVerdict object with the decision on whether to respond
        """
        # If the message is too short, it's likely not meaningful enough for a response
        if len(message.strip()) < 2:
            return MessageVerdict(
                should_respond=False,
                confidence=0.9,
                reason="Message is too short to require a response"
            )
        
        # Direct mentions/commands should always get a response
        if "@trademaster" in message.lower() or message.strip().startswith("!"):
            return MessageVerdict(
                should_respond=True,
                confidence=0.95,
                reason="Direct mention or command to the bot"
            )
        
        # Format the context as a string for the LLM
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        
        # Check if Ollama was properly initialized
        if not self.chain or not self.ollama:
            logger.info("Using heuristic evaluation (Ollama not initialized)")
            return self._simple_heuristic_evaluation(message, context)
        
        try:
            # Run the LLM evaluation asynchronously to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.chain.invoke({"message": message, "context": context_str})
            )
            
            # Parse the LLM's response
            # Extract YES/NO decision
            should_respond = "YES" in result.upper()
            
            # Extract the confidence score if provided
            confidence_match = re.search(r'(\d+\.\d+)', result)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            # Extract the reason if provided
            reason_match = re.search(r'reason:(.*?)(?:\n|$)', result.lower(), re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else result
            
            # Check for contradictions in the response
            # If the reason suggests it should respond despite a NO, override to YES
            trading_keywords = ["trading", "market", "stock", "crypto", "invest", "finance", "question", "requires", "assistance"]
            has_trading_keywords = any(keyword in reason.lower() for keyword in trading_keywords)
            
            # Determine if we should respond based on YES/NO and the reason content
            if should_respond or has_trading_keywords:
                # Log the decision at INFO level to ensure it's captured
                logger.info(f"Message evaluation: SHOULD RESPOND ({confidence:.2f}) - {reason}")
                logger.debug(f"Full LLM response: {result}")
                
                return MessageVerdict(
                    should_respond=True,
                    confidence=confidence,
                    reason=reason
                )
            else:
                # LLM determined no response is needed
                # Log the decision at INFO level to ensure it's captured
                logger.info(f"Message evaluation: SHOULD NOT RESPOND - {result}")
                
                return MessageVerdict(
                    should_respond=False,
                    confidence=confidence if confidence_match else 0.7,
                    reason=reason if reason != result else "Ollama model determined no response needed"
                )
        except Exception as e:
            # Log any errors with the LLM 
            logger.error(f"Error using Ollama for message evaluation: {e}")
            # If there's an error, fall back to simple heuristics
            logger.info("Falling back to simple heuristic evaluation")
            return self._simple_heuristic_evaluation(message, context)
    
    def update_training_data(self, message: str, should_have_responded: bool):
        """
        Update training data for the local model based on feedback.
        This method would collect data to improve the model over time.
        
        Args:
            message: The original message that was evaluated
            should_have_responded: Whether the bot should have responded (feedback)
        
        Note: This is currently a placeholder for future implementation.
        """
        # This could be implemented to:
        # 1. Store messages and correct responses in a database
        # 2. Periodically retrain the model with this data
        # 3. Track performance metrics over time
        pass