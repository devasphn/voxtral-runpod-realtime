# FINAL PERFECTED SOLUTION - main.py - VAD-FIXED & ROBUST STATE MANAGEMENT
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
import json
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from config.settings import Settings
from config.logging_config import setup_logging
from src.websocket_handler import WebSocketManager
from src.conversation_manager import ConversationManager
from src.model_loader import VoxtralModelManager
from src.audio_processor import PerfectAudioProcessor

# --- Initialization ---
setup_logging()
logger = logging.getLogger(__name__)

settings = Settings()
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager()
audio_processor = PerfectAudioProcessor()

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    logger.info("🚀 Starting FINAL PERFECTED Voxtral Real-Time Server...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        model_manager = VoxtralModelManager()
        await model_manager.load_model()
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to load model on startup: {e}", exc_info=True)
    yield
    logger.info("🛑 Shutting down FINAL PERFECTED server...")
    if model_manager: await model_manager.cleanup()
    logger.info("✅ FINAL PERFECTED graceful shutdown completed")

app = FastAPI(title="Voxtral Mini 3B - FINAL PERFECTED API", version="5.0.0-GOLD", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HTTP Endpoints ---
@app.get("/")
async def root():
    with open("static/index.html", "r") as f: return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_manager and model_manager.is_loaded else "error_not_loaded"
    status_code = 200 if model_status == "loaded" else 503
    return JSONResponse(status_code=status_code, content={"status": "healthy" if model_status == "loaded" else "unhealthy", "model_status": model_status})

@app.get("/model/info")
async def model_info():
    """THE FIX: Re-added the missing /model/info endpoint to prevent 404 errors from the frontend."""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or in error state.")
    return {
        "model_name": settings.MODEL_NAME, "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE), "context_length": "32K tokens",
        "supported_languages": list(model_manager.supported_languages),
        "architecture": "Voxtral-Mini-3B with Flash Attention 2"
    }

# --- WebSocket Endpoint: Understanding (with Gap Detection) ---
@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "unders
