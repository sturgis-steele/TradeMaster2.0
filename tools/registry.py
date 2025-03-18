"""
Tool Registry for TradeMaster 2.0

This module manages the registration and retrieval of tools.
It maintains a central registry of all available tools.
"""

import logging
from typing import Dict, List, Type, Optional
from .base_tool import BaseTool

logger = logging.getLogger("TradeMaster.Tools.Registry")

class ToolRegistry:
    """
    Registry for TradeMaster tools.
    
    This class manages the registration and retrieval of tools,
    ensuring that tool names are unique and providing a central
    access point for available tools.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        logger.info("Tool Registry initialized")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool: An instance of BaseTool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"A tool with the name '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: The name of the tool to retrieve
            
        Returns:
            The tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[BaseTool]:
        """
        Get a list of all registered tools.
        
        Returns:
            A list of all registered tool instances
        """
        return list(self._tools.values())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get a dictionary of tool names and descriptions.
        
        Returns:
            A dictionary mapping tool names to descriptions
        """
        return {tool.name: tool.description for tool in self._tools.values()}
    
    def get_tool_info(self) -> List[Dict]:
        """
        Get detailed information about all tools.
        
        Returns:
            A list of dictionaries containing tool information
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "examples": tool.examples
            }
            for tool in self._tools.values()
        ]
    
    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        logger.info("Tool Registry cleared")


# Create a singleton instance
registry = ToolRegistry()