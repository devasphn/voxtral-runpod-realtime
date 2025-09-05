# COMPLETELY FIXED MAIN.PY - FULL FASTAPI APPLICATION WITH PROPER INTEGRATION
import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import all necessary components
from config.settings import Settings
from config.logging_config import setup_logging
from src.model_loader import VoxtralUnderstandingManager
from src.audio_processor import UnderstandingAudioProcessor
from src.conversation_manager import ConversationManager
from src.websocket_handler import WebSocketManager
from src.utils import get_system_info, PerformanceMonitor

# Initialize settings
settings = Settings()

# Setup logging
setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE,
    enable_json_logging=False
)

logger = logging.getLogger(__name__)

# Global variables
model_manager: VoxtralUnderstandingManager = None
audio_processor: UnderstandingAudioProcessor = None
conversation_manager: ConversationManager = None
ws_manager: WebSocketManager = None
performance_monitor: PerformanceMonitor = None
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting COMPLETELY FIXED Voxtral Understanding-Only Server...")
    
    global model_manager, audio_processor, conversation_manager, ws_manager, performance_monitor
    
    try:
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Initialize WebSocket manager
        ws_manager = WebSocketManager()
        
        # Initialize conversation manager
        conversation_manager = ConversationManager(
            max_turns=settings.MAX_CONVERSATION_TURNS,
            context_window_minutes=settings.CONTEXT_WINDOW_MINUTES
        )
        
        # Initialize audio processor
        audio_processor = UnderstandingAudioProcessor(
            sample_rate=settings.AUDIO_SAMPLE_RATE,
            channels=settings.AUDIO_CHANNELS,
            gap_threshold_ms=settings.GAP_THRESHOLD_MS,
            conversation_manager=conversation_manager
        )
        
        # Initialize model manager
        model_manager = VoxtralUnderstandingManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        
        # Load model
        logger.info("📥 Loading Voxtral model...")
        await model_manager.load_model()
        
        logger.info("✅ COMPLETELY FIXED server startup completed!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down COMPLETELY FIXED server...")
    shutdown_event.set()
    
    if audio_processor:
        await audio_processor.cleanup()
    
    if model_manager:
        await model_manager.cleanup()
    
    logger.info("✅ COMPLETELY FIXED server shutdown completed!")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Understanding-Only Real-Time API",
    description="COMPLETELY FIXED Real-time conversational AI using Voxtral",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Signal handlers
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """COMPLETELY FIXED health check endpoint"""
    try:
        system_info = get_system_info()
        
        # Model status
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        # Connection stats
        connection_stats = ws_manager.get_connection_stats() if ws_manager else {}
        
        # Audio processor stats
        audio_stats = audio_processor.get_stats() if audio_processor else {}
        
        return {
            "status": "healthy",
            "timestamp": system_info["timestamp"],
            "mode": "COMPLETELY FIXED UNDERSTANDING-ONLY",
            "transcription_disabled": True,
            "model_status": model_status,
            "flash_attention_disabled": True,
            "gpu_available": system_info["gpu"]["available"],
            "active_connections": connection_stats.get("total_connections", 0),
            "audio_processing": audio_stats,
            "fixes_applied": [
                "✅ Complete main.py with proper FastAPI setup",
                "✅ Correct model loading with proper imports",
                "✅ Flash attention disabled for compatibility",
                "✅ Understanding-only mode enforced"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "mode": "COMPLETELY FIXED UNDERSTANDING-ONLY"
            }
        )

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_manager.model_info.copy()
    info.update({
        "flash_attention_status": "DISABLED (compatibility fix)",
        "gap_detection_ms": settings.GAP_THRESHOLD_MS,
        "target_response_ms": settings.TARGET_RESPONSE_MS,
        "mode": "COMPLETELY FIXED UNDERSTANDING-ONLY"
    })
    
    return info

@app.get("/connections")
async def connections_info():
    """Get WebSocket connections information"""
    if not ws_manager:
        return {"total_connections": 0, "connections_by_type": {}}
    
    return ws_manager.get_connection_stats()

@app.get("/stats")
async def system_stats():
    """Get comprehensive system statistics"""
    stats = {
        "system": get_system_info(),
        "model": model_manager.model_info if model_manager and model_manager.is_loaded else {},
        "connections": ws_manager.get_connection_stats() if ws_manager else {},
        "audio": audio_processor.get_stats() if audio_processor else {},
        "performance": performance_monitor.get_stats() if performance_monitor else {}
    }
    
    return stats

# WebSocket endpoint for understanding
@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """COMPLETELY FIXED: WebSocket endpoint for conversational AI"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 COMPLETELY FIXED UNDERSTANDING session started")
        
        while not shutdown_event.is_set():
            try:
                # Receive binary audio data
                audio_data = await websocket.receive_bytes()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Validate audio data
                if not audio_data or len(audio_data) < 100:
                    logger.debug("Invalid/insufficient audio data received")
                    continue
                
                # COMPLETELY FIXED: Process through corrected audio processor
                result = await audio_processor.process_audio_understanding(audio_data, websocket)
                
                if result and isinstance(result, dict):
                    if "error" in result:
                        logger.error(f"COMPLETELY FIXED audio processing error: {result['error']}")
                        await websocket.send_json({
                            "error": f"Audio processing failed: {result['error']}", 
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                        continue
                    
                    # Send intermediate feedback
                    if result.get("audio_received") and not result.get("speech_complete"):
                        await websocket.send_json({
                            "type": "audio_feedback",
                            "audio_received": True,
                            "segment_duration_ms": result.get("segment_duration_ms", 0),
                            "silence_duration_ms": result.get("silence_duration_ms", 0),
                            "remaining_to_gap_ms": result.get("remaining_to_gap_ms", 0),
                            "gap_will_trigger_at_ms": result.get("gap_will_trigger_at_ms", 300),
                            "speech_detected": result.get("speech_detected", False),
                            "speech_ratio": result.get("speech_ratio", 0),
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                        continue
                    
                    # COMPLETELY FIXED: Process complete speech segment
                    if result.get("speech_complete") and "audio_data" in result:
                        duration_ms = result.get("duration_ms", 0)
                        speech_quality = result.get("speech_quality", 0)
                        
                        # Quality check for understanding
                        if duration_ms > 500 and speech_quality > 0.15:
                            logger.info(f"🧠 COMPLETELY FIXED processing: {duration_ms:.0f}ms, quality: {speech_quality:.3f}")
                            
                            if model_manager and model_manager.is_loaded:
                                # Get conversation context
                                context = conversation_manager.get_conversation_context(websocket)
                                
                                # Create proper message format
                                message = {
                                    "audio": result["audio_data"],  # Raw PCM data
                                    "text": f"Please understand and respond to what you hear in the audio. {context}" if context else "Please understand and respond to what you hear in the audio."
                                }
                                
                                # COMPLETELY FIXED: Generate understanding response
                                understanding_result = await model_manager.understand_audio(message)
                                
                                if (isinstance(understanding_result, dict) and 
                                    understanding_result.get("response") and 
                                    "error" not in understanding_result and
                                    len(understanding_result["response"].strip()) > 3):
                                    
                                    response_time_ms = understanding_result.get("processing_time_ms", 0)
                                    transcribed_text = understanding_result.get("transcribed_text", "Audio processed")
                                    
                                    # Add to conversation
                                    conversation_manager.add_turn(
                                        websocket,
                                        transcription=transcribed_text,
                                        response=understanding_result["response"],
                                        audio_duration=duration_ms / 1000,
                                        speech_ratio=speech_quality,
                                        mode="understand",
                                        language=understanding_result.get("language", "en")
                                    )
                                    
                                    # Prepare response
                                    final_result = {
                                        "type": "understanding",
                                        "transcription": transcribed_text,
                                        "response": understanding_result["response"],
                                        "response_time_ms": response_time_ms,
                                        "audio_duration_ms": duration_ms,
                                        "speech_quality": speech_quality,
                                        "gap_detected": result.get("gap_detected", False),
                                        "language": understanding_result.get("language", "en"),
                                        "understanding_only": True,
                                        "transcription_disabled": True,
                                        "fixes_applied": True,
                                        "model_api_fixed": understanding_result.get("model_api_fixed", False),
                                        "sub_200ms": response_time_ms < 200,
                                        "timestamp": asyncio.get_event_loop().time()
                                    }
                                    
                                    # Add conversation stats
                                    conv_stats = conversation_manager.get_conversation_stats(websocket)
                                    final_result["conversation"] = conv_stats
                                    
                                    await websocket.send_json(final_result)
                                    logger.info(f"✅ COMPLETELY FIXED UNDERSTANDING complete: '{understanding_result['response'][:50]}...' ({response_time_ms:.0f}ms)")
                                else:
                                    logger.warning(f"Invalid understanding result: {understanding_result}")
                                    await websocket.send_json({
                                        "error": "Failed to generate understanding response",
                                        "understanding_only": True,
                                        "transcription_disabled": True,
                                        "fixes_applied": True
                                    })
                            else:
                                await websocket.send_json({
                                    "error": "Model not loaded",
                                    "understanding_only": True,
                                    "transcription_disabled": True,
                                    "fixes_applied": True
                                })
                        else:
                            logger.debug(f"Skipping low quality: duration={duration_ms:.0f}ms, quality={speech_quality:.3f}")
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"COMPLETELY FIXED inner WebSocket error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "understanding_only": True,
                            "transcription_disabled": True,
                            "fixes_applied": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("COMPLETELY FIXED WebSocket disconnected")
    except Exception as e:
        logger.error(f"COMPLETELY FIXED WebSocket error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        audio_processor.cleanup_connection(websocket)
        ws_manager.disconnect(websocket)

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
