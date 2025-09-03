import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time streaming"""
    
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self.connections_by_type: Dict[str, Set[WebSocket]] = {
            "transcribe": set(),
            "understand": set()
        }
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, connection_type: str = "transcribe"):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        async with self._lock:
            # Store connection info
            self.active_connections[websocket] = {
                "type": connection_type,
                "connected_at": asyncio.get_event_loop().time(),
                "messages_sent": 0,
                "messages_received": 0
            }
            
            # Add to type-specific set
            if connection_type in self.connections_by_type:
                self.connections_by_type[connection_type].add(websocket)
            
            logger.info(f"WebSocket connected: {connection_type} (Total: {len(self.active_connections)})")
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "connection_type": connection_type,
                "message": f"Connected to Voxtral {connection_type} service"
            })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            connection_info = self.active_connections[websocket]
            connection_type = connection_info["type"]
            
            # Remove from active connections
            del self.active_connections[websocket]
            
            # Remove from type-specific set
            if connection_type in self.connections_by_type:
                self.connections_by_type[connection_type].discard(websocket)
            
            logger.info(f"WebSocket disconnected: {connection_type} (Total: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
            
            # Update message count
            if websocket in self.active_connections:
                self.active_connections[websocket]["messages_sent"] += 1
                
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_type(self, message: Dict[str, Any], connection_type: str):
        """Broadcast message to all connections of specific type"""
        if connection_type not in self.connections_by_type:
            return
        
        disconnected = set()
        
        for websocket in self.connections_by_type[connection_type]:
            try:
                await websocket.send_json(message)
                if websocket in self.active_connections:
                    self.active_connections[websocket]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_type}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all active connections"""
        disconnected = set()
        
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
                self.active_connections[websocket]["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to all: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)
    
    @property
    def connection_count(self) -> int:
        """Get total number of active connections"""
        return len(self.active_connections)
    
    def get_connections_by_type(self) -> Dict[str, int]:
        """Get connection count by type"""
        return {
            conn_type: len(connections) 
            for conn_type, connections in self.connections_by_type.items()
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        total_sent = sum(info["messages_sent"] for info in self.active_connections.values())
        total_received = sum(info["messages_received"] for info in self.active_connections.values())
        
        return {
            "total_connections": len(self.active_connections),
            "connections_by_type": self.get_connections_by_type(),
            "total_messages_sent": total_sent,
            "total_messages_received": total_received,
            "average_messages_per_connection": total_sent / len(self.active_connections) if self.active_connections else 0
        }
