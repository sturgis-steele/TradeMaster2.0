import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger("TradeMaster.Context")

class ContextManager:
    """Manages user conversation contexts."""
    
    def __init__(self):
        # Simple dict to store user contexts
        self.contexts: Dict[str, Dict[str, Any]] = {}
        logger.info("Context Manager initialized")
    
    def update_last_message(self, user_id: str, message: str):
        if user_id not in self.contexts:
            self.contexts[user_id] = {}
        
        self.contexts[user_id]['last_message'] = message
        self.contexts[user_id]['last_active'] = datetime.now().isoformat()
    
    def get_context(self, user_id: str) -> Dict[str, Any]:
        return self.contexts.get(user_id, {})