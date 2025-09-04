# PERFECT COMPLETE SOLUTION - main.py - PERFECT 300MS GAP DETECTION
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
import json
import time
import io
import wave

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config.settings import Settings
from config.logging_config import setup_logging
from src.websocket_handler import WebSocketManager
from src.conversation_manager import ConversationManager
from src.utils import get_system_info
from src.model_loader import VoxtralModelManager
from src.audio_processor import PerfectAudioProcessor

# Initialize
setup_logging()
logger = logging.getLogger(__name__)

settings = Settings()
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager()
audio_processor = PerfectAudioProcessor(conversation_manager=conversation_manager)

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    logger.info("🚀 Starting PERFECT Voxtral Real-Time Server...")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load model on startup: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down PERFECT server...")
    if model_manager:
        await model_manager.cleanup()
    await audio_processor.cleanup()
    logger.info("✅ PERFECT graceful shutdown completed")

app = FastAPI(
    title="Voxtral Mini 3B - PERFECT Real-Time API",
    description="Real-time transcription and understanding with PERFECT 300ms gap detection",
    version="2.0.0-PERFECT",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if model_manager and model_manager.is_loaded else "not_loaded",
        "active_connections": ws_manager.connection_count,
        "perfect_architecture": True
    }

@app.get("/model/info")
async def model_info():
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": settings.MODEL_NAME,
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "supported_languages": list(model_manager.supported_languages.keys()),
        "api_methods": {
            "transcription": "apply_transcription_request (Correct & Perfected)",
            "understanding": "apply_chat_template (Correct & Perfected)"
        },
        "perfect": True
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await ws_manager.connect(websocket, "transcribe")
    conversation_manager.start_conversation(websocket)
    try:
        while not shutdown_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            
            if "bytes" in message:
                result = await audio_processor.process_webm_chunk_transcribe(message["bytes"], websocket)
                if result and "audio_data" in result:
                    transcription = await model_manager.transcribe_audio_pure(result["audio_data"])
                    if "error" not in transcription:
                        conversation_manager.add_turn(websocket, transcription=transcription.get("text", ""))
                        await websocket.send_json(transcription)
    except WebSocketDisconnect:
        logger.info("Transcription client disconnected.")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    # PERFECT: Context for this specific connection
    conn_context = {
        "pcm_buffer": bytearray(),
        "last_speech_time": time.time(),
        "processing": False,
        "user_query": "Please respond naturally to what I said."
    }

    async def process_gaps():
        """Background task for PERFECT 300ms gap detection"""
        while not shutdown_event.is_set() and websocket.client_state.name == 'CONNECTED':
            await asyncio.sleep(0.1) # Check every 100ms
            
            is_silent = (time.time() - conn_context["last_speech_time"]) > 0.3
            has_audio = len(conn_context["pcm_buffer"]) > (16000 * 2 * 0.5) # at least 0.5s of audio

            if not conn_context["processing"] and has_audio and is_silent:
                conn_context["processing"] = True
                
                audio_to_process = bytes(conn_context["pcm_buffer"])
                conn_context["pcm_buffer"].clear()
                
                logger.info(f"🧠 PERFECT GAP DETECTED: Processing {len(audio_to_process)} bytes of audio.")

                # 1. Transcribe the audio chunk
                transcription_result = await model_manager.transcribe_audio_pure(audio_to_process)
                
                if "error" not in transcription_result and transcription_result.get("text"):
                    transcribed_text = transcription_result["text"]
                    
                    # 2. Generate understanding response
                    context_str = conversation_manager.get_conversation_context(websocket)
                    understanding_result = await model_manager.generate_understanding_response(
                        transcribed_text=transcribed_text,
                        user_query=conn_context["user_query"],
                        context=context_str
                    )

                    if "error" not in understanding_result:
                        # Add to conversation history
                        conversation_manager.add_turn(
                            websocket,
                            transcription=transcribed_text,
                            response=understanding_result.get("response", "")
                        )
                        # Send result to client
                        await websocket.send_json(understanding_result)
                
                conn_context["processing"] = False

    gap_task = asyncio.create_task(process_gaps())

    try:
        while not shutdown_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message: # Handle user query updates
                try:
                    data = json.loads(message["text"])
                    if "query" in data:
                        conn_context["user_query"] = data["query"]
                        logger.info(f"User query updated to: '{data['query']}'")
                except json.JSONDecodeError:
                    pass

            elif "bytes" in message:
                result = await audio_processor.process_webm_chunk_understand(message["bytes"])
                if result and "pcm_data" in result:
                    conn_context["pcm_buffer"].extend(result["pcm_data"])
                    if result.get("speech_detected"):
                        conn_context["last_speech_time"] = time.time()

    except WebSocketDisconnect:
        logger.info("Understanding client disconnected.")
    finally:
        gap_task.cancel()
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)
