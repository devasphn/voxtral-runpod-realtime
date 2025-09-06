import asyncio, logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections = {}

    async def connect(self, ws: WebSocket, _type="understand"):
        await ws.accept()
        self.connections[ws] = {"sent":0,"recv":0}
        await ws.send_json({"type":"connected","message":"Ready for real-time understanding"})

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            del self.connections[ws]

    async def send_personal_message(self, msg: dict, ws: WebSocket):
        await ws.send_json(msg)
        self.connections[ws]["sent"] += 1

    def increment_received(self, ws: WebSocket):
        self.connections[ws]["recv"] += 1

    def get_connection_stats(self):
        total = len(self.connections)
        return {"total_connections":total}
