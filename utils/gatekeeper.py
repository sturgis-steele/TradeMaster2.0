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
from typing import Dict, Any
import asyncio
from pydantic import BaseModel, Field
import re

# LangChain imports for integrating with Ollama (local LLM)
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    
    It uses a local LLM (Ollama) to make intelligent filtering decisions.
    """
    
    def __init__(self, model_name="gemma3:1b", ollama_base_url="http://localhost:11434"):
        """
        Initialize the gatekeeper with a lightweight local model.
        
        Args:
            model_name: The name of the Ollama model to use (default: "gemma3:1b")
            ollama_base_url: The base URL for the Ollama API (default: "http://localhost:11434")
        """
        # Initialize the Ollama model with LangChain
        try:
            # Create an instance of the Ollama model
            self.ollama = OllamaLLM(model=model_name, base_url=ollama_base_url)
            
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
            
            # Create the chain using RunnableSequence pattern instead of LLMChain
            self.chain = self.prompt_template | self.ollama
            logger.info(f"Ollama model '{model_name}' initialized successfully")
        except Exception as e:
            # Log the error if Ollama initialization fails
            logger.error(f"Failed to initialize Ollama model: {e}")
            # Set chain to None to indicate initialization failed
            self.chain = None
            
        logger.info("Gatekeeper initialized")
    
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
            "coin", "exchange", "buy", "sell", "profit", "loss", "analysis"
        ]
        
        # Check if message directly addresses the bot
        direct_address = any(term in message_lower for term in ["bot", "trademaster", "assistant"])
        
        # Check if message contains finance-related keywords
        finance_related = any(keyword in message_lower for keyword in finance_keywords)
        
        # Check if message is a question
        is_question = "?" in message or any(q_word in message_lower.split() for q_word in ["what", "how", "when", "why", "where", "who", "which"])
        
        # Recent interaction suggests continuing the conversation
        recent_interaction = "last_interaction_time" in context
        
        # Determine if we should respond based on these factors
        should_respond = direct_address or finance_related or (is_question and (finance_related or recent_interaction))
        
        if should_respond:
            return MessageVerdict(
                should_respond=True,
                confidence=0.6,
                reason="Simple heuristic evaluation determined this message needs a response"
            )
        else:
            return MessageVerdict(
                should_respond=False,
                confidence=0.6,
                reason="Simple heuristic evaluation determined this message does not need a response"
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
        # Format the context as a string for the LLM
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        
        # Check if Ollama was properly initialized
        if not self.chain:
            logger.warning("Cannot evaluate message: Ollama model not initialized, using fallback heuristics")
            return self._simple_heuristic_evaluation(message, context)
        
        try:
            # Run the LLM evaluation asynchronously to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.chain.invoke({"message": message, "context": context_str})
            )
            
            # Parse the LLM's response
            # Extract the confidence score if provided
            confidence_match = re.search(r'(\d+\.\d+)', result)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            # Extract the reason if provided
            reason = result.split("reason:", 1)[-1].strip() if "reason:" in result.lower() else result
            
            # Check for contradictions in the response
            # If the reason suggests it should respond despite a NO, override to YES
            trading_keywords = ["trading", "market", "stock", "crypto", "invest", "finance", "question", "requires", "assistance"]
            has_trading_keywords = any(keyword in reason.lower() for keyword in trading_keywords)
            
            # Determine if we should respond based on YES/NO and the reason content
            if "YES" in result.upper() or has_trading_keywords:
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
                    confidence=0.7,
                    reason="Ollama model determined no response needed"
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