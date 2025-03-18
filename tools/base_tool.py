"""
Base Tool Interface for TradeMaster 2.0

This module defines the base interface that all tools must implement.
Tools extend the bot's capabilities by providing specialized functions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

logger = logging.getLogger("TradeMaster.Tools")

class BaseTool(ABC):
    """
    Abstract base class for all TradeMaster tools.
    All tools must extend this class and implement its methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the tool. Must be unique across all tools.
        Used as the command name when invoking the tool.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        A description of what the tool does. Should clearly explain
        the tool's purpose and when it should be used.
        """
        pass
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        """
        List of parameters the tool accepts.
        
        Returns:
            A list of parameter dictionaries with these keys:
            - name: The parameter name
            - type: The parameter type (string, number, boolean)
            - description: A description of the parameter
            - required: Whether the parameter is required
            - default: The default value (if not required)
        """
        return []
    
    @property
    def examples(self) -> List[str]:
        """Example invocations of the tool."""
        return []
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the provided parameters.
        
        Args:
            **kwargs: The parameters for the tool
            
        Returns:
            A dictionary containing the tool's output and any relevant data
        """
        pass