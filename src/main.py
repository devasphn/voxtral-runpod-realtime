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
    """Get model information for the frontend"""
    try:
        if not model_manager or not model_manager.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        info = {
            "model_name": settings.MODEL_NAME,
            "device": str(settings.DEVICE),
            "model_size": "3B parameters",
            "flash_attention_status": "DISABLED (compatibility fix)" if not settings.USE_FLASH_ATTENTION else "ENABLED",
            "gap_detection_ms": settings.GAP_THRESHOLD_MS,
            "target_response_ms": settings.TARGET_RESPONSE_MS,
            "supported_languages": ["English", "Spanish", "French", "Portuguese", "Hindi", "German", "Dutch", "Italian"],
            "understanding_only": True,
            "transcription_disabled": True,
            "features": ["Conversational AI", "Context Memory", "WebRTC VAD", "Gap Detection"],
            "audio_config": {
                "sample_rate": settings.AUDIO_SAMPLE_RATE,
                "channels": settings.AUDIO_CHANNELS,
                "gap_threshold_ms": settings.GAP_THRESHOLD_MS,
                "min_speech_duration_ms": settings.MIN_SPEECH_DURATION_MS,
                "max_speech_duration_ms": settings.MAX_SPEECH_DURATION_MS
            }
        }
        return info
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/connections")
async def connections_info():
    """Get WebSocket connections information"""
    try:
        stats = ws_manager.get_connection_stats()
        return {
            "total_connections": stats["total_connections"],
            "connections_by_type": stats.get("connections_by_type", {}),
            "max_connections": settings.MAX_CONCURRENT_CONNECTIONS,
            "timestamp": get_system_info()["timestamp"]
        }
    except Exception as e:
        logger.error(f"Connections info error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)

    try:
        while not shutdown_event.is_set():
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                ws_manager.increment_received(websocket)
                
                if len(data) < 50:
                    # Send explicit feedback for very small packets
                    await websocket.send_json({
                        "type": "audio_feedback",
                        "message": "Very small audio packet received",
                        "understanding_only": True,
                        "packet_size": len(data)
                    })
                    continue

                # Process audio through the understanding pipeline
                result = await audio_processor.process_audio_understanding(data, websocket)
                
                if result.get("speech_complete"):
                    # Process through model and send final response
                    logger.info(f"Processing complete speech segment ({len(result.get('audio_data', b''))} bytes)")
                    
                    try:
                        response = await model_manager.understand_audio({
                            "audio": result["audio_data"],
                            "text": result.get("text", "Listen to this audio and provide a helpful response.")
                        })
                        
                        # Add conversation turn
                        if "response" in response:
                            conversation_manager.add_turn(
                                websocket=websocket,
                                transcription="[Audio input]",
                                response=response["response"],
                                audio_duration=result.get("duration_ms", 0) / 1000.0,
                                mode="understand"
                            )
                        
                        await ws_manager.send_personal_message(response, websocket)
                        
                    except Exception as e:
                        logger.error(f"Model inference error: {e}")
                        error_response = {
                            "type": "understanding",
                            "error": f"Model inference failed: {str(e)}",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                        await ws_manager.send_personal_message(error_response, websocket)
                else:
                    # Send intermediate feedback
                    if result.get("audio_received"):
                        feedback = {
                            "type": "audio_feedback",
                            "speech_detected": result.get("speech_detected", False),
                            "duration_ms": result.get("duration_ms", 0),
                            "silence_duration_ms": result.get("silence_duration_ms", 0),
                            "remaining_to_gap_ms": max(0, settings.GAP_THRESHOLD_MS - result.get("silence_duration_ms", 0)),
                            "understanding_only": True
                        }
                        await ws_manager.send_personal_message(feedback, websocket)

            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive",
                    "message": "WebSocket connection active",
                    "timestamp": asyncio.get_event_loop().time()
                })
                continue
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "understanding_only": True
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        audio_processor.cleanup_connection(websocket)
        ws_manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower(), access_log=True)
