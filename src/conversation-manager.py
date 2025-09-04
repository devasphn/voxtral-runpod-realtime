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
    audio_duration: float
    transcription: str
    response: str
    language: Optional[str] = None
    speech_ratio: float = 0.0
    mode: str = "transcribe"  # transcribe or understand

class ConversationManager:
    """Manages conversation memory and context for longer interactions"""
    
    def __init__(self, max_turns: int = 20, context_window_minutes: int = 10):
        self.max_turns = max_turns
        self.context_window = timedelta(minutes=context_window_minutes)
        
        # Conversation storage per WebSocket connection
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        
        # Language tracking for code-switching detection
        self.language_patterns: Dict[str, List[str]] = {}
        
        # Audio context for continuous processing
        self.audio_context: Dict[str, bytes] = {}
        
        logger.info(f"✅ ConversationManager initialized: max_turns={max_turns}, context_window={context_window_minutes}min")
    
    def get_connection_id(self, websocket) -> str:
        """Generate unique connection ID"""
        return f"ws_{id(websocket)}"
    
    def start_conversation(self, websocket) -> str:
        """Start a new conversation session"""
        conn_id = self.get_connection_id(websocket)
        self.conversations[conn_id] = []
        self.language_patterns[conn_id] = []
        self.audio_context[conn_id] = b""
        
        logger.info(f"🆕 Started conversation for connection: {conn_id}")
        return conn_id
    
    def add_turn(self, websocket, transcription: str, response: str = "", 
                 audio_duration: float = 0.0, speech_ratio: float = 0.0,
                 mode: str = "transcribe", language: str = None):
        """Add a conversation turn"""
        conn_id = self.get_connection_id(websocket)
        
        if conn_id not in self.conversations:
            self.start_conversation(websocket)
        
        # Detect language for code-switching
        detected_lang = self._detect_language(transcription, conn_id)
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            audio_duration=audio_duration,
            transcription=transcription,
            response=response,
            language=language or detected_lang,
            speech_ratio=speech_ratio,
            mode=mode
        )
        
        # Add turn to conversation
        self.conversations[conn_id].append(turn)
        
        # Update language patterns
        if detected_lang:
            self.language_patterns[conn_id].append(detected_lang)
        
        # Cleanup old turns
        self._cleanup_old_turns(conn_id)
        
        logger.info(f"➕ Added conversation turn: {conn_id} - '{transcription[:50]}...' (lang: {detected_lang})")
    
    def get_conversation_context(self, websocket, max_context_length: int = 1000) -> str:
        """Get recent conversation context as a formatted string"""
        conn_id = self.get_connection_id(websocket)
        
        if conn_id not in self.conversations:
            return ""
        
        turns = self.conversations[conn_id]
        if not turns:
            return ""
        
        # Get recent turns within context window
        cutoff_time = datetime.now() - self.context_window
        recent_turns = [turn for turn in turns if turn.timestamp > cutoff_time]
        
        # Format context
        context_parts = []
        for i, turn in enumerate(recent_turns[-5:]):  # Last 5 turns
            if turn.transcription:
                context_parts.append(f"User: {turn.transcription}")
            if turn.response and turn.mode == "understand":
                context_parts.append(f"Assistant: {turn.response}")
        
        full_context = "\\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > max_context_length:
            full_context = "..." + full_context[-max_context_length:]
        
        return full_context
    
    def get_conversation_stats(self, websocket) -> Dict[str, Any]:
        """Get conversation statistics"""
        conn_id = self.get_connection_id(websocket)
        
        if conn_id not in self.conversations:
            return {"turns": 0, "languages": [], "total_duration": 0.0}
        
        turns = self.conversations[conn_id]
        languages = list(set(turn.language for turn in turns if turn.language))
        total_duration = sum(turn.audio_duration for turn in turns)
        
        return {
            "turns": len(turns),
            "languages": languages,
            "total_duration": total_duration,
            "last_activity": turns[-1].timestamp.isoformat() if turns else None,
            "code_switching": self._detect_code_switching(conn_id)
        }
    
    def cleanup_conversation(self, websocket):
        """Clean up conversation data when WebSocket disconnects"""
        conn_id = self.get_connection_id(websocket)
        
        # Store final stats before cleanup
        stats = self.get_conversation_stats(websocket)
        
        # Clean up data
        self.conversations.pop(conn_id, None)
        self.language_patterns.pop(conn_id, None)
        self.audio_context.pop(conn_id, None)
        
        logger.info(f"🧹 Cleaned up conversation: {conn_id} - Final stats: {stats}")
    
    def _detect_language(self, text: str, conn_id: str) -> Optional[str]:
        """Simple language detection for code-switching"""
        if not text:
            return None
        
        # Simple heuristics for Hindi-English detection
        hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहा़ि्ीुूेैोौंःऽ')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        text_chars = set(text)
        has_hindi = bool(text_chars.intersection(hindi_chars))
        has_english = bool(text_chars.intersection(english_chars))
        
        if has_hindi and has_english:
            return "hi-en"  # Code-switched
        elif has_hindi:
            return "hi"
        elif has_english:
            return "en"
        else:
            # Fallback based on recent patterns
            recent_patterns = self.language_patterns.get(conn_id, [])[-3:]
            if recent_patterns:
                return recent_patterns[-1]
            return "en"  # Default to English
    
    def _detect_code_switching(self, conn_id: str) -> bool:
        """Detect if conversation involves code-switching"""
        languages = self.language_patterns.get(conn_id, [])
        if len(languages) < 2:
            return False
        
        # Check for language alternation
        unique_langs = set(languages)
        has_mixed = "hi-en" in unique_langs
        has_alternation = len(unique_langs) > 1
        
        return has_mixed or has_alternation
    
    def _cleanup_old_turns(self, conn_id: str):
        """Remove old conversation turns to prevent memory bloat"""
        if conn_id not in self.conversations:
            return
        
        turns = self.conversations[conn_id]
        
        # Remove turns older than context window
        cutoff_time = datetime.now() - self.context_window
        self.conversations[conn_id] = [
            turn for turn in turns 
            if turn.timestamp > cutoff_time
        ]
        
        # Also limit by max_turns
        if len(self.conversations[conn_id]) > self.max_turns:
            self.conversations[conn_id] = self.conversations[conn_id][-self.max_turns:]
        
        # Cleanup language patterns too
        if conn_id in self.language_patterns:
            pattern_count = len(self.conversations[conn_id])
            self.language_patterns[conn_id] = self.language_patterns[conn_id][-pattern_count:]
    
    def get_suggested_prompt(self, websocket) -> str:
        """Get context-aware prompt for understanding mode"""
        context = self.get_conversation_context(websocket)
        stats = self.get_conversation_stats(websocket)
        
        base_prompt = "Listen to the audio and provide a helpful response."
        
        if context:
            base_prompt = f"Previous conversation:\\n{context}\\n\\nListen to the current audio and provide a helpful response that considers the conversation history."
        
        # Add language-specific instructions for code-switching
        if stats.get("code_switching", False):
            base_prompt += "\\n\\nNote: This conversation involves mixing Hindi and English. Please respond appropriately to any language switches."
        
        return base_prompt
