import asyncio
import logging
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import Settings
from config.logging_config import setup_logging
from src.model_loader import VoxtralUnderstandingManager
from src.audio_processor import UnderstandingAudioProcessor
from src.conversation_manager import ConversationManager
from src.websocket_handler import WebSocketManager
from src.utils import get_system_info, PerformanceMonitor

settings = Settings()
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE, enable_json_logging=False)
logger = logging.getLogger(__name__)

model_manager: VoxtralUnderstandingManager = None
audio_processor: UnderstandingAudioProcessor = None
conversation_manager: ConversationManager = None
ws_manager: WebSocketManager = None
performance_monitor: PerformanceMonitor = None
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Voxtral Understanding-Only Server...")
    global model_manager, audio_processor, conversation_manager, ws_manager, performance_monitor

    performance_monitor = PerformanceMonitor()
    ws_manager = WebSocketManager()
    conversation_manager = ConversationManager(
        max_turns=settings.MAX_CONVERSATION_TURNS,
        context_window_minutes=settings.CONTEXT_WINDOW_MINUTES
    )
    audio_processor = UnderstandingAudioProcessor(
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        channels=settings.AUDIO_CHANNELS,
        gap_threshold_ms=settings.GAP_THRESHOLD_MS,
        conversation_manager=conversation_manager
    )
    model_manager = VoxtralUnderstandingManager(
        model_name=settings.MODEL_NAME,
        device=settings.DEVICE,
        torch_dtype=settings.TORCH_DTYPE
    )

    logger.info("📥 Loading Voxtral model...")
    await model_manager.load_model()
    logger.info("✅ Server startup completed!")
    yield

    logger.info("🛑 Shutting down server...")
    shutdown_event.set()
    if audio_processor:
        await audio_processor.cleanup()
    if model_manager:
        await model_manager.cleanup()
    logger.info("✅ Server shutdown completed!")

app = FastAPI(
    title="Voxtral Understanding-Only Real-Time API",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        conn_stats = ws_manager.get_connection_stats()
        audio_stats = audio_processor.get_stats()
        return {
            "status": "healthy",
            "timestamp": system_info["timestamp"],
            "model_status": model_status,
            "active_connections": conn_stats["total_connections"],
            "audio_processing": audio_stats
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})

@app.get("/model/info")
async def model_info():
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = model_manager.model_info.copy()
    info.update({"target_response_ms": settings.TARGET_RESPONSE_MS})
    return info

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)

    try:
        while not shutdown_event.is_set():
            data = await websocket.receive_bytes()
            ws_manager.increment_received(websocket)
            if len(data) < 50:
                # explicit no-speech feedback after repeated small packets
                await websocket.send_json({"type":"no_speech","message":"No speech detected","understanding_only":True})
                continue

            result = await audio_processor.process_audio_understanding(data, websocket)
            if result.get("speech_complete"):
                # Process through model and send final response
                response = await model_manager.understand_audio({
                    "audio": result["audio_data"],
                    "text": result.get("transcribed_text_request", "")
                })
                await ws_manager.send_personal_message(response, websocket)
            else:
                # intermediate feedback
                await ws_manager.send_personal_message(result, websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        audio_processor.cleanup_connection(websocket)
        ws_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower(), access_log=True)
