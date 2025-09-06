import asyncio
import logging
from fastapi import WebSocket
from typing import Dict, List, Optional, Any
import time
import json

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}
        self.total_messages_sent = 0
        self.total_messages_received = 0

    async def connect(self, websocket: WebSocket, connection_type: str = "understand"):
        """Accept and register a new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Initialize connection metadata
            self.active_connections[websocket] = {
                "type": connection_type,
                "connected_at": time.time(),
                "messages_sent": 0,
                "messages_received": 0,
                "last_activity": time.time()
            }
            
            # Send welcome message
            welcome_message = {
                "type": "connection",
                "status": "connected",
                "connection_type": connection_type,
                "message": f"Connected to {connection_type} service",
                "understanding_only": True,
                "server_time": time.time(),
                "features": [
                    "Real-time audio understanding",
                    "300ms gap detection",
                    "Context memory",
                    "Sub-200ms response target",
                    "WebRTC VAD",
                    "No transcription (understanding only)"
                ]
            }
            
            await self.send_personal_message(welcome_message, websocket)
            
            logger.info(f"✅ WebSocket connected: {connection_type} (total: {len(self.active_connections)})")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection and log statistics"""
        if websocket in self.active_connections:
            connection_info = self.active_connections[websocket]
            connection_type = connection_info.get("type", "unknown")
            messages_sent = connection_info.get("messages_sent", 0)
            messages_received = connection_info.get("messages_received", 0)
            duration = time.time() - connection_info.get("connected_at", time.time())
            
            # Remove from active connections
            del self.active_connections[websocket]
            
            logger.info(f"🔌 WebSocket disconnected: {connection_type}, "
                       f"duration: {duration:.1f}s, sent: {messages_sent}, received: {messages_received} "
                       f"(remaining: {len(self.active_connections)})")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            if websocket in self.active_connections:
                # Update activity time
                self.active_connections[websocket]["last_activity"] = time.time()
                
                # Ensure message has timestamp
                if "timestamp" not in message:
                    message["timestamp"] = time.time()
                
                # Send message
                await websocket.send_json(message)
                
                # Update counters
                self.active_connections[websocket]["messages_sent"] += 1
                self.total_messages_sent += 1
                
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            # Remove broken connection
            if websocket in self.active_connections:
                self.disconnect(websocket)

    async def broadcast_to_type(self, message: dict, connection_type: str):
        """Broadcast message to all connections of specific type"""
        if not self.active_connections:
            return

        # Add timestamp to message
        if "timestamp" not in message:
            message["timestamp"] = time.time()

        # Send to all connections of the specified type
        tasks = []
        for websocket, info in self.active_connections.items():
            if info.get("type") == connection_type:
                tasks.append(self.send_personal_message(message, websocket))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Broadcasted message to {len(tasks)} {connection_type} connections")

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all active connections"""
        if not self.active_connections:
            return

        # Add timestamp to message
        if "timestamp" not in message:
            message["timestamp"] = time.time()

        # Send to all connections
        tasks = []
        for websocket in self.active_connections.keys():
            tasks.append(self.send_personal_message(message, websocket))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Broadcasted message to {len(tasks)} connections")

    def increment_received(self, websocket: WebSocket):
        """Increment received message counter for connection"""
        if websocket in self.active_connections:
            self.active_connections[websocket]["messages_received"] += 1
            self.active_connections[websocket]["last_activity"] = time.time()
            self.total_messages_received += 1

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        if not self.active_connections:
            return {
                "total_connections": 0,
                "connections_by_type": {},
                "total_messages_sent": self.total_messages_sent,
                "total_messages_received": self.total_messages_received,
                "average_messages_per_connection": 0,
                "uptime_stats": {}
            }

        # Count connections by type
        connections_by_type = {}
        total_sent = 0
        total_received = 0
        uptimes = []
        current_time = time.time()

        for websocket, info in self.active_connections.items():
            conn_type = info.get("type", "unknown")
            connections_by_type[conn_type] = connections_by_type.get(conn_type, 0) + 1
            
            total_sent += info.get("messages_sent", 0)
            total_received += info.get("messages_received", 0)
            
            uptime = current_time - info.get("connected_at", current_time)
            uptimes.append(uptime)

        # Calculate averages
        total_connections = len(self.active_connections)
        avg_messages_per_connection = (total_sent + total_received) / total_connections if total_connections > 0 else 0
        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0

        return {
            "total_connections": total_connections,
            "connections_by_type": connections_by_type,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "session_messages_sent": total_sent,
            "session_messages_received": total_received,
            "average_messages_per_connection": round(avg_messages_per_connection, 2),
            "uptime_stats": {
                "average_uptime_seconds": round(avg_uptime, 2),
                "max_uptime_seconds": round(max(uptimes, default=0), 2),
                "min_uptime_seconds": round(min(uptimes, default=0), 2)
            },
            "server_uptime": time.time()  # Could be enhanced to track actual server start time
        }

    def get_connection_info(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get information about specific connection"""
        return self.active_connections.get(websocket)

    async def send_health_check(self):
        """Send health check to all connections"""
        health_message = {
            "type": "health_check",
            "status": "healthy",
            "server_time": time.time(),
            "active_connections": len(self.active_connections),
            "understanding_only": True
        }
        
        await self.broadcast_to_all(health_message)

    async def cleanup_stale_connections(self, max_idle_seconds: int = 300):
        """Remove connections that have been idle for too long"""
        current_time = time.time()
        stale_connections = []
        
        for websocket, info in self.active_connections.items():
            last_activity = info.get("last_activity", current_time)
            idle_time = current_time - last_activity
            
            if idle_time > max_idle_seconds:
                stale_connections.append(websocket)
        
        # Close and remove stale connections
        for websocket in stale_connections:
            try:
                await websocket.close(code=1000, reason="Connection idle timeout")
            except Exception as e:
                logger.debug(f"Error closing stale connection: {e}")
            finally:
                self.disconnect(websocket)
        
        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")

    @property
    def connection_count(self) -> int:
        """Get current connection count"""
        return len(self.active_connections)

    def is_connected(self, websocket: WebSocket) -> bool:
        """Check if WebSocket is still connected"""
        return websocket in self.active_connections
