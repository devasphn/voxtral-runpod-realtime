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

# --- Initialization ---
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
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to load model on startup: {e}", exc_info=True)
    
    yield
    
    logger.info("🛑 Shutting down FINAL PERFECTED server...")
    if model_manager:
        await model_manager.cleanup()
    logger.info("✅ FINAL PERFECTED graceful shutdown completed")

# --- FastAPI App Definition ---
app = FastAPI(
    title="Voxtral Mini 3B - FINAL PERFECTED API",
    description="Real-time transcription and understanding with <300ms gap detection and corrected VAD logic.",
    version="4.0.0-STABLE",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HTTP Endpoints ---
@app.get("/")
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_manager and model_manager.is_loaded else "error_not_loaded"
    status_code = 200 if model_status == "loaded" else 503
    return HTTPException(status_code=status_code, detail={"status": "healthy" if model_status == "loaded" else "unhealthy", "model_status": model_status})

@app.get("/model/info")
async def model_info():
    """Provides model information to the frontend."""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or in error state.")
    return {
        "model_name": settings.MODEL_NAME,
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "supported_languages": list(model_manager.supported_languages),
        "api_methods": {
            "transcription": "apply_transcription_request (Correct & Perfected)",
            "understanding": "apply_chat_template (Correct & Perfected)"
        },
        "perfect": True
    }

# --- WebSocket Endpoint: Transcription ---
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
                result = await audio_processor.process_webm_chunk_transcribe(message["bytes"])
                if result and "audio_data" in result:
                    transcription = await model_manager.transcribe_audio_pure(result["audio_data"])
                    if "error" not in transcription and transcription.get("text", "").strip():
                        conversation_manager.add_turn(websocket, transcription=transcription["text"])
                        await websocket.send_json(transcription)
    except WebSocketDisconnect:
        logger.info("Transcription client disconnected.")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

# --- WebSocket Endpoint: Understanding (with Gap Detection) ---
@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    conn_context = {
        "pcm_buffer": bytearray(),
        "last_speech_time": time.time(),
        "processing": False,
        "user_query": "Please respond naturally to what I said.",
        "min_buffer_size": int(settings.AUDIO_SAMPLE_RATE * 2 * 0.25) # Minimum 0.25 seconds of audio
    }

    async def process_audio_chunk(audio_chunk: bytes, reason: str):
        """Encapsulated processing logic for reuse."""
        logger.info(f"🧠 Processing {len(audio_chunk)} bytes of audio. Reason: {reason}.")
        
        # Step 1: Transcribe the audio chunk
        transcription_result = await model_manager.transcribe_audio_pure(audio_chunk)
        if "error" in transcription_result or not transcription_result.get("text", "").strip():
            logger.warning(f"Skipping empty or failed transcription: {transcription_result}")
            return

        # Step 2: Generate understanding response
        transcribed_text = transcription_result["text"]
        context_str = conversation_manager.get_conversation_context(websocket)
        understanding_result = await model_manager.generate_understanding_response(
            transcribed_text=transcribed_text, user_query=conn_context["user_query"], context=context_str
        )

        # Step 3: Send result and save history
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
            await asyncio.sleep(0.1) # Check every 100ms for silence gap
            
            silence_duration_ms = (time.time() - conn_context["last_speech_time"]) * 1000
            is_silent_gap = silence_duration_ms > 300 # User-defined gap threshold
            has_enough_audio = len(conn_context["pcm_buffer"]) > conn_context["min_buffer_size"]

            if not conn_context["processing"] and has_enough_audio and is_silent_gap:
                conn_context["processing"] = True
                audio_to_process = bytes(conn_context["pcm_buffer"])
                conn_context["pcm_buffer"].clear()
                try:
                    await process_audio_chunk(audio_to_process, reason=f"Gap detected ({silence_duration_ms:.0f}ms)")
                except Exception as e:
                    logger.error(f"Error during gap processing: {e}", exc_info=True)
                finally:
                    conn_context["processing"] = False # CRITICAL: Ensure flag is always reset

    gap_task = asyncio.create_task(gap_detection_task())

    try:
        while not shutdown_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if "query" in data:
                        conn_context["user_query"] = data["query"]
                        logger.info(f"User query updated to: '{data['query']}'")
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
        # FINAL PERFECTED: Process any remaining audio in the buffer on disconnect (final flush)
        if not conn_context["processing"] and len(conn_context["pcm_buffer"]) > conn_context["min_buffer_size"]:
             logger.info("Client disconnected, performing final flush of audio buffer.")
             await process_audio_chunk(bytes(conn_context["pcm_buffer"]), reason="Final flush on disconnect")

        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)
