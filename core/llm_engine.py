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
from typing import Dict, Any, Tuple, List, Optional, Union
import asyncio
import json
import traceback

# LangChain and Crew AI imports
try:
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import SystemMessage, HumanMessage
    from langchain_core.messages import AIMessage
    from langchain.tools import BaseTool
    
    # Crew AI imports
    from crewai import Agent, Task, Crew, Process
    
    LANGCHAIN_AVAILABLE = True
    CREW_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    CREW_AVAILABLE = False
    logging.error(f"Required packages not installed: {e}")

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
        self.llm = None
        self.trading_agent = None
        self.crew = None
        self.task = None
        
        # Initialize tools registry - will be populated with available tools
        self.tools: List[BaseTool] = []
        
        # Check if required packages are available
        if not LANGCHAIN_AVAILABLE or not CREW_AVAILABLE:
            logger.error("LangChain or CrewAI not available. LLM functionality will be limited.")
            return
        
        # Check if API key is available
        if not self.api_key:
            logger.warning("No Groq API key provided. LLM functionality will be limited.")
            return
        
        try:
            # Initialize the Groq LLM
            model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=model_name,
                temperature=0.7,
                max_tokens=1024
            )
            logger.info(f"Initialized Groq LLM with model {model_name}")
            
            # Initialize the agent and crew
            self._initialize_crew()
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _initialize_crew(self) -> None:
        """
        Initialize the Crew AI agent and workflow.
        This defines the decision-making process of the agent.
        """
        try:
            # Define the system prompt for the agent
            system_prompt = "You are TradeMaster, an AI assistant specialized in trading, investing, and financial markets. You help users with trading strategies, market analysis, and financial education. When responding to users: 1. Be concise, accurate, and helpful 2. Use data and facts from your tools to support your statements 3. Use appropriate tools when needed to provide better assistance"
            
            # Create the main trading assistant agent
            self.trading_agent = Agent(
                role="Trading Assistant",
                goal="Provide helpful, accurate information about trading cryptocurrency and financial markets",
                backstory="You are an expert in trading, investing, and financial markets with years of experience helping traders make informed decisions.",
                verbose=True,
                llm=self.llm,
                tools=self.tools
            )
            
            # Create a default task that will be used as a template
            self.task = Task(
                description="Respond to user messages about trading and finance",
                expected_output="A helpful response to the user's query",
                agent=self.trading_agent
            )
            
            # Create the crew with our agent
            self.crew = Crew(
                agents=[self.trading_agent],
                process=Process.sequential,  # Use sequential process for single agent
                verbose=True,
                tasks=[self.task]  # Add the template task to the crew
            )
            
            logger.info("Built agent workflow with Crew AI")
            
        except Exception as e:
            logger.error(f"Failed to initialize Crew AI workflow: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.trading_agent = None
            self.crew = None
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a new tool with the LLM engine.
        
        Args:
            tool: The tool to register
        """
        try:
            self.tools.append(tool)
            logger.info(f"Registered tool: {tool.name}")
            
            # Rebuild the crew with the new tool
            if LANGCHAIN_AVAILABLE and CREW_AVAILABLE and self.llm:
                self._initialize_crew()
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
    
    async def process_message(self, message: str, user_id: Union[str, int], context: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """
        Process a message from a user and generate a response.
        This is the main entry point for the LLM engine.
        
        Args:
            message: The user's message content
            user_id: The Discord user ID (string or integer)
            context: The user's conversation context
            
        Returns:
            A tuple of (response_text, tool_used)
        """
        # Default fallback response if things go wrong
        fallback_response = "I'm having trouble processing your request right now. Please try again later."
        
        # If LangChain or CrewAI are not available, provide a fallback response
        if not LANGCHAIN_AVAILABLE or not CREW_AVAILABLE:
            logger.error("LangChain or CrewAI not available. Cannot process message.")
            return fallback_response, None
        
        # If LLM or Crew is not initialized, provide a fallback response
        if not self.llm or not self.crew or not self.trading_agent:
            logger.error("LLM engine not properly initialized. Cannot process message.")
            return fallback_response, None
            
        try:
            # Log the incoming parameters for debugging
            logger.debug(f"Processing message for user_id: {user_id} (type: {type(user_id).__name__})")
            logger.debug(f"Context keys: {list(context.keys())}")
            
            # Create a simplified task for this specific message
            task_description = f"Respond to the following message from user {user_id}: {message}"
            
            # Convert user_id to string (CrewAI sometimes has issues with non-string values)
            safe_user_id = str(user_id)
            
            # Create crew inputs dictionary with safe string conversions
            crew_inputs = {
                "user_id": safe_user_id,
                "message": message,
                "task_description": task_description
            }
            
            # Add safe context items
            for key, value in context.items():
                if key != "conversation_history":
                    try:
                        if isinstance(value, (dict, list)):
                            crew_inputs[key] = json.dumps(value)
                        else:
                            crew_inputs[key] = str(value)
                    except:
                        # Skip problematic values
                        pass
            
            # Add a simplified form of conversation history if it exists
            if "conversation_history" in context and context["conversation_history"]:
                try:
                    # Create a simplified version of the conversation history
                    history = []
                    for msg in context["conversation_history"][-6:]:  # Only use last 3 exchanges (6 messages)
                        if hasattr(msg, "content"):
                            role = "user" if isinstance(msg, HumanMessage) else "assistant"
                            history.append(f"{role}: {msg.content[:100]}...")
                    
                    crew_inputs["conversation_history"] = "\n".join(history)
                except Exception as e:
                    logger.warning(f"Could not process conversation history: {e}")
            
            # Run the crew asynchronously
            loop = asyncio.get_event_loop()
            try:
                logger.debug("About to execute Crew AI task")
                
                # Use a timeout to prevent hanging
                # The key fix here is removing 'tasks' parameter from kickoff()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.crew.kickoff(inputs=crew_inputs)
                    ),
                    timeout=60  # 60 second timeout
                )
                logger.info("Successfully received response from Crew AI")
            except asyncio.TimeoutError:
                logger.error("Timeout while waiting for Crew AI response")
                return "I'm sorry, but it's taking me too long to process your request. Could you try again with a simpler question?", None
            except Exception as e:
                logger.error(f"Error running Crew AI task: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception args: {e.args}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return fallback_response, None
            
            # Extract the response from the result
            try:
                # Different versions of CrewAI return result in different formats
                # Try several approaches to extract the response
                if isinstance(result, str):
                    response = result
                elif hasattr(result, 'output'):
                    response = result.output
                elif isinstance(result, list) and len(result) > 0:
                    if hasattr(result[0], 'output'):
                        response = result[0].output
                    else:
                        response = str(result[0])
                else:
                    response = str(result)
                
                # Ensure the response is a string
                if not isinstance(response, str):
                    response = str(response)
                
                logger.info(f"Generated response: {response[:50]}...")
                tool_used = None  # In a real implementation, you would track which tool was used
            except Exception as e:
                logger.error(f"Error extracting response from result: {e}")
                logger.error(f"Result type: {type(result).__name__}")
                logger.error(f"Result: {str(result)[:100]}...")
                return fallback_response, None
            
            # Update conversation history in the context
            try:
                if "conversation_history" not in context:
                    context["conversation_history"] = []
                
                # Add this exchange to the history
                context["conversation_history"].append(HumanMessage(content=message))
                context["conversation_history"].append(AIMessage(content=response))
                
                # Trim history if it gets too long
                if len(context["conversation_history"]) > 10:  # Keep last 10 exchanges
                    context["conversation_history"] = context["conversation_history"][-10:]
            except Exception as e:
                logger.warning(f"Error updating conversation history: {e}")
                # Continue anyway, this isn't critical
            
            return response, tool_used
            
        except Exception as e:
            logger.error(f"Error processing message with LLM: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            # Include traceback for more detailed debugging
            logger.error(f"Traceback: {traceback.format_exc()}")
            return fallback_response, None