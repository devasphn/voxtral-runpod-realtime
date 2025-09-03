import asyncio
import pytest
import json
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from src.main import app
from src.websocket_handler import WebSocketManager
import base64

# Test client
client = TestClient(app)

@pytest.fixture
def websocket_manager():
    """Create WebSocket manager for testing"""
    return WebSocketManager()

class MockWebSocket:
    """Mock WebSocket for testing"""
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.is_closed = False
        
    async def accept(self):
        pass
        
    async def send_json(self, data):
        self.messages_sent.append(data)
        
    async def send_bytes(self, data):
        self.messages_sent.append(data)
        
    async def receive_json(self):
        if self.messages_received:
            return self.messages_received.pop(0)
        await asyncio.sleep(0.1)  # Simulate waiting
        
    async def receive_bytes(self):
        return b"test_audio_data"
        
    async def close(self):
        self.is_closed = True
        
    def add_message(self, message):
        """Add message to receive queue"""
        self.messages_received.append(message)

@pytest.mark.asyncio
async def test_websocket_connection(websocket_manager):
    """Test WebSocket connection management"""
    mock_ws = MockWebSocket()
    
    # Test connection
    await websocket_manager.connect(mock_ws, "transcribe")
    
    assert websocket_manager.connection_count == 1
    assert mock_ws in websocket_manager.active_connections
    assert len(mock_ws.messages_sent) == 1  # Welcome message
    
    # Check welcome message
    welcome_msg = mock_ws.messages_sent[0]
    assert welcome_msg["type"] == "connection"
    assert welcome_msg["status"] == "connected"
    assert welcome_msg["connection_type"] == "transcribe"

@pytest.mark.asyncio
async def test_websocket_disconnect(websocket_manager):
    """Test WebSocket disconnection"""
    mock_ws = MockWebSocket()
    
    # Connect then disconnect
    await websocket_manager.connect(mock_ws, "transcribe")
    initial_count = websocket_manager.connection_count
    
    websocket_manager.disconnect(mock_ws)
    
    assert websocket_manager.connection_count == initial_count - 1
    assert mock_ws not in websocket_manager.active_connections

@pytest.mark.asyncio
async def test_websocket_broadcast(websocket_manager):
    """Test message broadcasting"""
    mock_ws1 = MockWebSocket()
    mock_ws2 = MockWebSocket()
    
    # Connect multiple websockets
    await websocket_manager.connect(mock_ws1, "transcribe")
    await websocket_manager.connect(mock_ws2, "understand")
    
    # Clear welcome messages
    mock_ws1.messages_sent.clear()
    mock_ws2.messages_sent.clear()
    
    # Test type-specific broadcast
    test_message = {"type": "test", "data": "hello"}
    await websocket_manager.broadcast_to_type(test_message, "transcribe")
    
    assert len(mock_ws1.messages_sent) == 1
    assert len(mock_ws2.messages_sent) == 0  # Different type
    assert mock_ws1.messages_sent[0] == test_message

@pytest.mark.asyncio
async def test_websocket_broadcast_all(websocket_manager):
    """Test broadcast to all connections"""
    mock_ws1 = MockWebSocket()
    mock_ws2 = MockWebSocket()
    
    # Connect multiple websockets
    await websocket_manager.connect(mock_ws1, "transcribe")
    await websocket_manager.connect(mock_ws2, "understand")
    
    # Clear welcome messages
    mock_ws1.messages_sent.clear()
    mock_ws2.messages_sent.clear()
    
    # Test broadcast to all
    test_message = {"type": "global", "data": "broadcast"}
    await websocket_manager.broadcast_to_all(test_message)
    
    assert len(mock_ws1.messages_sent) == 1
    assert len(mock_ws2.messages_sent) == 1
    assert mock_ws1.messages_sent[0] == test_message
    assert mock_ws2.messages_sent[0] == test_message

def test_websocket_connection_stats(websocket_manager):
    """Test connection statistics"""
    stats = websocket_manager.get_connection_stats()
    
    assert "total_connections" in stats
    assert "connections_by_type" in stats
    assert "total_messages_sent" in stats
    assert "total_messages_received" in stats
    assert "average_messages_per_connection" in stats

@pytest.mark.asyncio
async def test_personal_message(websocket_manager):
    """Test sending personal message"""
    mock_ws = MockWebSocket()
    await websocket_manager.connect(mock_ws, "transcribe")
    
    # Clear welcome message
    mock_ws.messages_sent.clear()
    
    # Send personal message
    personal_msg = {"type": "personal", "content": "Hello user"}
    await websocket_manager.send_personal_message(personal_msg, mock_ws)
    
    assert len(mock_ws.messages_sent) == 1
    assert mock_ws.messages_sent[0] == personal_msg
    
    # Check message count updated
    connection_info = websocket_manager.active_connections[mock_ws]
    assert connection_info["messages_sent"] == 1

def test_websocket_health_endpoint():
    """Test WebSocket health via HTTP endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "active_connections" in data
    assert "timestamp" in data

def test_websocket_connections_endpoint():
    """Test connections info endpoint"""
    response = client.get("/connections")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_connections" in data
    assert "connections_by_type" in data
    assert "max_connections" in data

# Integration test with actual WebSocket
def test_websocket_transcribe_endpoint():
    """Test WebSocket transcribe endpoint (basic connection)"""
    with client.websocket_connect("/ws/transcribe") as websocket:
        # Should receive welcome message
        data = websocket.receive_json()
        assert data["type"] == "connection"
        assert data["status"] == "connected"
        assert data["connection_type"] == "transcribe"

def test_websocket_understand_endpoint():
    """Test WebSocket understand endpoint (basic connection)"""
    with client.websocket_connect("/ws/understand") as websocket:
        # Should receive welcome message  
        data = websocket.receive_json()
        assert data["type"] == "connection"
        assert data["status"] == "connected"
        assert data["connection_type"] == "understand"

@pytest.mark.asyncio
async def test_audio_data_handling():
    """Test audio data processing"""
    # Test audio data validation
    valid_wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt '
    invalid_data = b'invalid_audio_data'
    
    # These would be tested with actual audio processor
    # For now, just test data types
    assert isinstance(valid_wav_header, bytes)
    assert len(valid_wav_header) > 0
    
@pytest.mark.asyncio 
async def test_base64_audio_handling():
    """Test base64 encoded audio handling"""
    # Create test audio data
    test_audio = b'test_audio_data_123456789'
    base64_audio = base64.b64encode(test_audio).decode()
    
    # Test decode
    decoded = base64.b64decode(base64_audio)
    assert decoded == test_audio
    
    # Test message format for understanding
    message = {
        "audio": base64_audio,
        "text": "What can you hear?"
    }
    
    assert "audio" in message
    assert "text" in message
    assert isinstance(message["audio"], str)  # Base64 string

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
