# NEW: Conversation Memory and Context Manager
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation"""
    timestamp: datetime
    transcription: str
    response: str
    mode: str

class ConversationManager:
    """Manages conversation memory and context for longer interactions"""
    
    def __init__(self, max_turns: int = 10, context_window_minutes: int = 5):
        self.max_turns = max_turns
        self.context_window = timedelta(minutes=context_window_minutes)
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        logger.info(f"✅ ConversationManager initialized: max_turns={max_turns}, context_window={context_window_minutes}min")
    
    def get_connection_id(self, websocket) -> str:
        return f"ws_{id(websocket)}"
    
    def start_conversation(self, websocket) -> str:
        conn_id = self.get_connection_id(websocket)
        self.conversations[conn_id] = []
        logger.info(f"🆕 Started conversation for connection: {conn_id}")
        return conn_id
    
    def add_turn(self, websocket, transcription: str, response: str = "", mode: str = "transcribe"):
        # Don't add empty turns to the history
        if not transcription.strip() and not response.strip():
            return
            
        conn_id = self.get_connection_id(websocket)
        if conn_id not in self.conversations:
            self.start_conversation(websocket)
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            transcription=transcription,
            response=response,
            mode=mode
        )
        self.conversations[conn_id].append(turn)
        self._cleanup_old_turns(conn_id)
    
    def get_conversation_context(self, websocket, max_context_length: int = 800) -> str:
        conn_id = self.get_connection_id(websocket)
        if conn_id not in self.conversations: return ""
        
        recent_turns = [
            turn for turn in self.conversations[conn_id] 
            if (datetime.now() - turn.timestamp) < self.context_window
        ]
        
        context_parts = []
        for turn in recent_turns[-self.max_turns:]:
            if turn.transcription: context_parts.append(f"User said: {turn.transcription}")
            if turn.response and turn.mode == "understand": context_parts.append(f"You replied: {turn.response}")
        
        full_context = " | ".join(context_parts)
        
        if len(full_context) > max_context_length:
            return "..." + full_context[-max_context_length:]
        return full_context
    
    def cleanup_conversation(self, websocket):
        conn_id = self.get_connection_id(websocket)
        if self.conversations.pop(conn_id, None):
            logger.info(f"🧹 Cleaned up conversation: {conn_id}")
