import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for PURE UNDERSTANDING-ONLY streaming"""
    
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        # PURE UNDERSTANDING-ONLY: Only one connection type
        self.understanding_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, connection_type: str = "understand"):
        """Accept new WebSocket connection - PURE UNDERSTANDING-ONLY"""
        await websocket.accept()
        
        # Force understanding-only mode
        if connection_type != "understand":
            logger.warning(f"Attempted connection type '{connection_type}' forced to 'understand' (PURE UNDERSTANDING-ONLY)")
            connection_type = "understand"
        
        async with self._lock:
            # Store connection info
            self.active_connections[websocket] = {
                "type": "understand",  # Always understanding
                "connected_at": asyncio.get_event_loop().time(),
                "messages_sent": 0,
                "messages_received": 0,
                "transcription_disabled": True,
                "understanding_only": True
            }
            
            # Add to understanding connections
            self.understanding_connections.add(websocket)
            
            logger.info(f"WebSocket connected: UNDERSTANDING-ONLY (Total: {len(self.active_connections)})")
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "connection_type": "understand",
                "mode": "PURE UNDERSTANDING-ONLY",
                "transcription_disabled": True,
                "message": "Connected to Voxtral PURE UNDERSTANDING-ONLY service",
                "features": [
                    "Conversational AI responses",
                    "0.3-second gap detection",
                    "Context-aware conversation",
                    "No transcription capability"
                ]
            })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            connection_info = self.active_connections[websocket]
            
            # Remove from active connections
            del self.active_connections[websocket]
            
            # Remove from understanding connections
            self.understanding_connections.discard(websocket)
            
            logger.info(f"WebSocket disconnected: UNDERSTANDING-ONLY (Total: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            # Add understanding-only markers
            message.update({
                "understanding_only": True,
                "transcription_disabled": True
            })
            
            await websocket.send_json(message)
            
            # Update message count
            if websocket in self.active_connections:
                self.active_connections[websocket]["messages_sent"] += 1
                
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_understanding(self, message: Dict[str, Any]):
        """Broadcast message to all understanding connections"""
        disconnected = set()
        
        # Add understanding-only markers
        message.update({
            "understanding_only": True,
            "transcription_disabled": True
        })
        
        for websocket in self.understanding_connections:
            try:
                await websocket.send_json(message)
                if websocket in self.active_connections:
                    self.active_connections[websocket]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to understanding: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections (UNDERSTANDING-ONLY)"""
        # Since we only have understanding connections, this is the same as broadcast_to_understanding
        await self.broadcast_to_understanding(message)
    
    @property
    def connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_connections_by_type(self) -> Dict[str, int]:
        """Get connection count by type - PURE UNDERSTANDING-ONLY"""
        return {
            "understand": len(self.understanding_connections),
            "transcribe": 0,  # Always 0 - transcription disabled
            "total": len(self.active_connections)
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        total_sent = sum(info["messages_sent"] for info in self.active_connections.values())
        total_received = sum(info["messages_received"] for info in self.active_connections.values())
        
        return {
            "mode": "PURE UNDERSTANDING-ONLY",
            "transcription_disabled": True,
            "total_connections": len(self.active_connections),
            "understanding_connections": len(self.understanding_connections),
            "transcription_connections": 0,  # Always 0
            "connections_by_type": self.get_connections_by_type(),
            "total_messages_sent": total_sent,
            "total_messages_received": total_received,
            "average_messages_per_connection": total_sent / len(self.active_connections) if self.active_connections else 0,
            "connection_features": [
                "Understanding-only mode",
                "Conversational AI responses",
                "0.3s gap detection",
                "Context memory",
                "No transcription capability"
            ]
        }
