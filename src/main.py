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

# Initialize
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

app = FastAPI(title="Voxtral Mini 3B - FINAL PERFECTED API", version="3.0.0-FINAL", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html", "r") as f: return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_manager and model_manager.is_loaded else "error_not_loaded"
    status_code = 200 if model_status == "loaded" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if model_status == "loaded" else "unhealthy", "model_status": model_status}
    )

@app.get("/model/info")
async def model_info():
    """THE FIX: This endpoint was missing, causing a 404 error on the frontend."""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": settings.MODEL_NAME, "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE), "context_length": "32K tokens",
        "supported_languages": list(model_manager.supported_languages),
        "architecture": "Voxtral-Mini-3B with Flash Attention 2"
    }

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    conn_context = {"pcm_buffer": bytearray(), "last_speech_time": time.time(), "processing": False, "user_query": "Please respond naturally."}

    async def process_audio_chunk(audio_chunk: bytes):
        if len(audio_chunk) < 1600: return # Ignore tiny audio fragments
        logger.info(f"🧠 Processing {len(audio_chunk)} bytes of audio.")
        
        transcription_result = await model_manager.transcribe_audio_pure(audio_chunk)
        if "error" in transcription_result or not transcription_result.get("text", "").strip():
            logger.warning(f"Skipping empty or failed transcription: {transcription_result}")
            return

        transcribed_text = transcription_result["text"]
        logger.info(f"🎤 Transcribed: '{transcribed_text}'")
        context_str = conversation_manager.get_conversation_context(websocket)
        understanding_result = await model_manager.generate_understanding_response(
            transcribed_text=transcribed_text, user_query=conn_context["user_query"], context=context_str
        )

        if "error" not in understanding_result:
            conversation_manager.add_turn(
                websocket, transcription=transcribed_text, response=understanding_result.get("response", ""), mode="understand"
            )
            try:
                await websocket.send_json(understanding_result)
            except WebSocketDisconnect: logger.warning("Client disconnected during message send.")

    async def gap_detection_task():
        while websocket.client_state.name == 'CONNECTED' and not shutdown_event.is_set():
            await asyncio.sleep(0.1)
            
            is_silent = (time.time() - conn_context["last_speech_time"]) > 0.3
            has_enough_audio = len(conn_context["pcm_buffer"]) > 4800 # ~0.15s

            if not conn_context["processing"] and has_enough_audio and is_silent:
                conn_context["processing"] = True
                audio_to_process = conn_context["pcm_buffer"]
                conn_context["pcm_buffer"] = bytearray() # Clear buffer immediately
                try:
                    await process_audio_chunk(bytes(audio_to_process))
                finally:
                    conn_context["processing"] = False

    gap_task = asyncio.create_task(gap_detection_task())

    try:
        while not shutdown_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect": break
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if "query" in data: conn_context["user_query"] = data["query"]
                except json.JSONDecodeError: pass
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
        if not conn_context["processing"] and len(conn_context["pcm_buffer"]) > 1600:
             logger.info("Client disconnected, performing final flush of audio buffer.")
             await process_audio_chunk(bytes(conn_context["pcm_buffer"]))
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)
