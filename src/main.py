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
from fastapi.responses import HTMLResponse

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
audio_processor = PerfectAudioProcessor(conversation_manager=conversation_manager)

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
        # In a real production system, you might want to exit here.
        # For now, we'll let it run so the health check can report the model error.
    yield
    logger.info("🛑 Shutting down FINAL PERFECTED server...")
    if model_manager: await model_manager.cleanup()
    logger.info("✅ FINAL PERFECTED graceful shutdown completed")

app = FastAPI(
    title="Voxtral Mini 3B - FINAL PERFECTED API",
    version="3.0.0-FINAL",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html", "r") as f: return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_manager and model_manager.is_loaded else "error_not_loaded"
    status_code = 200 if model_status == "loaded" else 503
    return {"status": "healthy" if model_status == "loaded" else "unhealthy", "model_status": model_status}, status_code

# Unchanged endpoints: /model/info, /ws/transcribe

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    conn_context = {"pcm_buffer": bytearray(), "last_speech_time": time.time(), "processing": False, "user_query": "Please respond naturally."}

    async def process_audio_chunk(audio_chunk: bytes, final_flush: bool = False):
        """Encapsulated processing logic for reuse."""
        logger.info(f"🧠 Processing {len(audio_chunk)} bytes of audio. Final flush: {final_flush}")
        
        transcription_result = await model_manager.transcribe_audio_pure(audio_chunk)
        if "error" in transcription_result or not transcription_result.get("text", "").strip():
            logger.warning(f"Skipping empty or failed transcription: {transcription_result}")
            return

        transcribed_text = transcription_result["text"]
        context_str = conversation_manager.get_conversation_context(websocket)
        understanding_result = await model_manager.generate_understanding_response(
            transcribed_text=transcribed_text, user_query=conn_context["user_query"], context=context_str
        )

        if "error" not in understanding_result:
            conversation_manager.add_turn(
                websocket, transcription=transcribed_text, response=understanding_result.get("response", "")
            )
            try:
                await websocket.send_json(understanding_result)
            except WebSocketDisconnect:
                logger.warning("Client disconnected during message send.")

    async def gap_detection_task():
        """FINAL PERFECTED: Background task with robust state management."""
        while not shutdown_event.is_set() and websocket.client_state.name == 'CONNECTED':
            await asyncio.sleep(0.15) # Check every 150ms
            
            is_silent = (time.time() - conn_context["last_speech_time"]) > 0.35 # 350ms gap
            has_enough_audio = len(conn_context["pcm_buffer"]) > 8000 # ~0.25s of audio

            if not conn_context["processing"] and has_enough_audio and is_silent:
                conn_context["processing"] = True
                audio_to_process = bytes(conn_context["pcm_buffer"])
                conn_context["pcm_buffer"].clear()
                try:
                    await process_audio_chunk(audio_to_process)
                finally:
                    conn_context["processing"] = False # CRITICAL: Ensure flag is always reset

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
        # FINAL PERFECTED: Process any remaining audio in the buffer on disconnect.
        if not conn_context["processing"] and len(conn_context["pcm_buffer"]) > 4800:
             logger.info("Client disconnected, performing final flush of audio buffer.")
             await process_audio_chunk(bytes(conn_context["pcm_buffer"]), final_flush=True)

        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)
