import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch

from config.settings import Settings
from config.logging_config import setup_logging
from websocket_handler import WebSocketManager
from model_loader import VoxtralModelManager
from utils import get_system_info

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global model manager
model_manager = None
ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting Voxtral Real-Time Server...")
    logger.info(f"Settings: {settings.dict()}")
    
    # Initialize model
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Voxtral Real-Time Server...")
    if model_manager:
        await model_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B Real-Time API",
    description="Real-time audio processing with Mistral's Voxtral Mini 3B",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main test client page"""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "system": system_info,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters",
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "capabilities": [
            "Speech transcription",
            "Audio understanding", 
            "Multi-turn conversations",
            "Function calling"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription"""
    await ws_manager.connect(websocket, "transcribe")
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Process with model
            if model_manager and model_manager.is_loaded:
                result = await model_manager.transcribe_audio(data)
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Model not loaded"})
                
    except WebSocketDisconnect:
        logger.info("Transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """WebSocket endpoint for real-time audio understanding"""
    await ws_manager.connect(websocket, "understand")
    try:
        while True:
            # Receive audio data or text query
            message = await websocket.receive_json()
            
            # Process with model
            if model_manager and model_manager.is_loaded:
                result = await model_manager.understand_audio(message)
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Model not loaded"})
                
    except WebSocketDisconnect:
        logger.info("Understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket understanding error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        ws_manager.disconnect(websocket)

@app.get("/connections")
async def get_connections():
    """Get active WebSocket connections info"""
    return {
        "total_connections": ws_manager.connection_count,
        "connections_by_type": ws_manager.get_connections_by_type(),
        "max_connections": settings.MAX_CONCURRENT_CONNECTIONS
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
