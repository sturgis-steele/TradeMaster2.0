import logging

logger = logging.getLogger("TradeMaster.Gatekeeper")

class Gatekeeper:
    """Determines which messages warrant a response."""
    
    def __init__(self):
        logger.info("Gatekeeper initialized (placeholder)")
    
    async def should_respond(self, message: str, user_id: str, context=None) -> bool:
        # In Phase 1, use simple logic (e.g., only respond to direct mentions)
        return "trademaster" in message.lower()