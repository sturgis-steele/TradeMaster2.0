# TradeMaster Discord Bot - LLM Engine
# This file implements the core LLM engine that powers the bot's intelligence.
# It uses Crew AI for agent workflows and the Groq API for fast, efficient responses.
#
# File Interactions:
# - bot/client.py: Imports and uses LLMEngine for processing user messages
# - main.py: Indirectly connected through bot/client.py
# - Groq API: Connects to Groq's LLM services for AI responses
# - Crew AI: Uses Crew AI library for agent-based workflows

import logging
import os
from typing import Dict, Any, Tuple, List, Optional
import asyncio

# LangChain and Crew AI imports
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage
from langchain.tools import BaseTool

# Crew AI imports
from crewai import Agent, Task, Crew, Process

# Set up logging for this module
logger = logging.getLogger("TradeMaster.LLMEngine")

class LLMEngine:
    """
    The core LLM engine that powers the TradeMaster bot.
    This class handles the processing of messages, tool selection, and response generation.
    It uses Crew AI to create a flexible agent workflow.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM engine with the Groq API.
        
        Args:
            api_key: The Groq API key (defaults to GROQ_API_KEY environment variable)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("No Groq API key provided. LLM functionality will be limited.")
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="llama3-70b-8192",  # Can be configured based on needs
            temperature=0.7,
            max_tokens=1024
        )
        logger.info("Initialized Groq LLM")
        
        # Initialize tools registry - will be populated with available tools
        self.tools: List[BaseTool] = []
        
        # Initialize the agent and crew
        self._initialize_crew()
        logger.info("Built agent workflow with Crew AI")
    
    def _initialize_crew(self) -> None:
        """
        Initialize the Crew AI agent and workflow.
        This defines the decision-making process of the agent.
        """
        # Define the system prompt for the agent
        system_prompt = "You are TradeMaster, an AI assistant specialized in trading, investing, and financial markets. You help users with trading strategies, market analysis, and financial education. When responding to users: 1. Be concise, accurate, and helpful 2. Use data and facts from your tools to support your statements 3. Use appropriate tools when needed to provide better assistance"
        
        # Create the main trading assistant agent
        # Use a role without format placeholders to avoid string interpolation issues
        self.trading_agent = Agent(
            role="Trading Assistant",  # Simple role without format placeholders
            goal="Provide helpful, accurate information about trading cryptocurrency and financial markets",
            backstory="You are an expert in trading, investing, and financial markets with years of experience helping traders make informed decisions.",
            verbose=True,
            llm=self.llm,
            tools=self.tools
        )
        
        # Create the crew with our agent
        self.crew = Crew(
            agents=[self.trading_agent],
            process=Process.sequential,  # Use sequential process for single agent
            verbose=True
        )
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a new tool with the LLM engine.
        
        Args:
            tool: The tool to register
        """
        self.tools.append(tool)
        logger.info(f"Registered tool: {tool.name}")
        
        # Rebuild the crew with the new tool
        self._initialize_crew()
    
    async def process_message(self, message: str, user_id: Any, context: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        # Changed parameter type to Any to handle both string and integer user_ids
        """
        Process a message from a user and generate a response.
        This is the main entry point for the LLM engine.
        
        Args:
            message: The user's message content
            user_id: The Discord user ID
            context: The user's conversation context
            
        Returns:
            A tuple of (response_text, tool_used)
        """
        try:
            # Log the incoming parameters for debugging
            logger.debug(f"Processing message for user_id: {user_id} (type: {type(user_id).__name__})")
            logger.debug(f"Context keys: {list(context.keys())}")
            
            # Create a task for the agent to process the message
            # Crew AI expects inputs as a dictionary for string interpolation, not a list
            # Convert the context dictionary to a format Crew AI can use
            
            # Create a dictionary for Crew AI inputs
            crew_inputs = {
                "user_id": str(user_id),  # Convert user_id to string to avoid validation errors
                "message": message,
                "conversation_history": str(context.get("conversation_history", []))
            }
            
            # Add any additional context items from the context dictionary
            for key, value in context.items():
                if key not in ["conversation_history"]:
                    # Ensure all values are properly converted to strings
                    crew_inputs[key] = str(value)
                    logger.debug(f"Added context item: {key} (type: {type(value).__name__})")
            
            logger.debug(f"Created crew_inputs with {len(crew_inputs)} items")
            
            # Create a context list for the Task object
            # This is different from the inputs used for string interpolation
            context_list = [
                {
                    "description": "User ID",
                    "expected_output": str(user_id)
                },
                {
                    "description": "User message",
                    "expected_output": message
                },
                {
                    "description": "Conversation history",
                    "expected_output": str(context.get("conversation_history", []))
                }
            ]
            
            logger.debug(f"Created context_list with {len(context_list)} initial items")
            
            # Create the task with proper logging
            try:
                task = Task(
                    description=f"Respond to the following message: {message}",
                    expected_output="A helpful response to the user's query",
                    agent=self.trading_agent,
                    context=context_list
                )
                logger.debug("Successfully created Task object")
            except Exception as task_error:
                logger.error(f"Error creating Task object: {task_error}")
                # Log detailed information about the context_list for debugging
                for i, ctx_item in enumerate(context_list):
                    logger.error(f"Context item {i}: description={ctx_item['description']}, expected_output type={type(ctx_item['expected_output']).__name__}")
                raise
            
            # Log that we're about to process the message with the LLM
            logger.info(f"Processing message with LLM: {message[:50]}...")
            
            # Run the crew asynchronously
            loop = asyncio.get_event_loop()
            try:
                logger.debug("About to execute Crew AI task")
                result = await loop.run_in_executor(
                    None,
                    lambda: self.crew.kickoff(inputs=crew_inputs, tasks=[task])
                )
                logger.info("Successfully received response from Crew AI")
            except Exception as e:
                logger.error(f"Error running Crew AI task: {e}")
                # Log more details about the exception
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception args: {e.args}")
                raise
            
            # Extract the response from the task output
            response = result[0].output if isinstance(result, list) else result.output
            logger.info(f"Generated response: {response[:50]}...")
            tool_used = None  # In a real implementation, you would track which tool was used
            
            # Update conversation history in the context
            if "conversation_history" not in context:
                context["conversation_history"] = []
            
            # Add this exchange to the history
            context["conversation_history"].append(HumanMessage(content=message))
            context["conversation_history"].append(AIMessage(content=response))
            
            # Trim history if it gets too long
            if len(context["conversation_history"]) > 10:  # Keep last 10 exchanges
                context["conversation_history"] = context["conversation_history"][-10:]
            
            return response, tool_used
            
        except Exception as e:
            logger.error(f"Error processing message with LLM: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            # Include traceback for more detailed debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I'm having trouble processing your request right now. Please try again later.", None

# Example of how to create and register a custom tool:
"""
# 1. Define your tool class
from langchain.tools import BaseTool

class MarketDataTool(BaseTool):
    name = "market_data"
    description = "Get current market data for a specific symbol"
    
    def _run(self, symbol: str) -> str:
        # Implementation to fetch market data
        return f"Market data for {symbol}: Price $100, Change +2%"
    
    async def _arun(self, symbol: str) -> str:
        # Async implementation
        return self._run(symbol)

# 2. Register the tool with the LLM engine
llm_engine = LLMEngine()
llm_engine.register_tool(MarketDataTool())
"""