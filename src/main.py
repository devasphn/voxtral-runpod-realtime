# ULTIMATE REAL-TIME VOXTRAL STREAMING - FIXED IMPLEMENTATION
import asyncio
import logging
import signal
import sys
import os
import json
import base64
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch

from config.settings import Settings
from config.logging_config import setup_logging
from src.websocket_handler import WebSocketManager
from src.conversation_manager import ConversationManager
from src.utils import get_system_info

# Import the FINAL FIXED model manager
from src.model_loader import VoxtralModelManager

try:
    from src.audio_processor import FixedAudioProcessor as AudioProcessor
except ImportError:
    from src.audio_processor import AudioProcessor

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global managers
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager(max_turns=50, context_window_minutes=30)

# REAL-TIME AUDIO PROCESSOR WITH ENHANCED VAD
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    chunk_duration_ms=30,
    conversation_manager=conversation_manager,
    vad_threshold_ms=300,  # 300ms silence threshold
    real_time_mode=True
)

# Shutdown flag
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ULTIMATE FIX: Application lifespan with optimized model loading"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting ULTIMATE REAL-TIME Voxtral Server...")
    
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE,
            optimize_for_realtime=True  # NEW: Real-time optimization
        )
        await model_manager.load_model_optimized()  # NEW: Optimized loading
        logger.info("✅ ULTIMATE model loaded for real-time streaming!")
    except Exception as e:
        logger.error(f"❌ Failed to load ULTIMATE model: {e}")
        raise RuntimeError(f"ULTIMATE model loading failed: {e}")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ULTIMATE server...")
    shutdown_event.set()
    
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup
    if model_manager:
        await model_manager.cleanup()
    await audio_processor.cleanup()
    
    logger.info("✅ ULTIMATE graceful shutdown completed")

async def background_cleanup():
    """Background maintenance for real-time performance"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(60)  # Every minute
            
            # Clean up old VAD data
            audio_processor.cleanup_old_vad_data()
            
            # Model cache optimization
            if model_manager:
                await model_manager.optimize_cache()
                
            logger.debug(f"🔄 Background cleanup: {ws_manager.connection_count} active")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Real-Time Streaming - ULTIMATE FIXED",
    description="Real-time voice streaming with 300ms VAD and <200ms response",
    version="1.0.0-ULTIMATE",
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
    """ULTIMATE: Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "ultimate_fixes": [
                "✅ Unified WebSocket protocol handling",
                "✅ Real-time VAD with 300ms silence detection", 
                "✅ Optimized audio pipeline for streaming",
                "✅ Model API optimization for <200ms response",
                "✅ Continuous recording architecture"
            ],
            "vad_stats": audio_processor.get_vad_stats(),
            "system": system_info,
            "timestamp": time.time(),
            "ultimate_realtime": True
        }
    except Exception as e:
        logger.error(f"ULTIMATE health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_realtime_unified(websocket: WebSocket):
    """ULTIMATE: Unified WebSocket for real-time streaming with proper protocol handling"""
    await ws_manager.connect(websocket, "realtime")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🎤 ULTIMATE real-time session started")
        
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected", 
            "mode": "realtime",
            "vad_threshold_ms": 300,
            "target_latency_ms": 200,
            "continuous_recording": True
        })
        
        while not shutdown_event.is_set():
            try:
                # ULTIMATE FIX: Unified message handling
                raw_message = await websocket.receive()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Handle different message types properly
                message_type = raw_message.get("type")
                
                if message_type == "websocket.receive":
                    # Handle binary audio data (continuous recording)
                    if "bytes" in raw_message:
                        audio_data = raw_message["bytes"]
                        await handle_audio_stream(websocket, audio_data)
                    
                    # Handle JSON text messages (commands/queries)
                    elif "text" in raw_message:
                        try:
                            json_message = json.loads(raw_message["text"])
                            await handle_json_message(websocket, json_message)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON received")
                    
                elif message_type == "websocket.disconnect":
                    logger.info("Client disconnected")
                    break
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"ULTIMATE WebSocket error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "ultimate_realtime": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("ULTIMATE WebSocket disconnected")
    except Exception as e:
        logger.error(f"ULTIMATE WebSocket error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

async def handle_audio_stream(websocket: WebSocket, audio_data: bytes):
    """ULTIMATE: Handle continuous audio stream with real-time VAD"""
    try:
        if not audio_data or len(audio_data) < 200:
            return
        
        # Process through real-time VAD audio processor
        vad_result = await audio_processor.process_realtime_audio(audio_data, websocket)
        
        if vad_result and vad_result.get("speech_ended"):
            # 300ms silence detected - process accumulated audio
            accumulated_audio = vad_result.get("accumulated_audio")
            if accumulated_audio and len(accumulated_audio) > 1000:
                
                logger.info(f"🎤 Processing speech segment: {len(accumulated_audio)} bytes")
                start_time = time.time()
                
                if model_manager and model_manager.is_loaded:
                    # ULTIMATE: Real-time optimized transcription
                    result = await model_manager.transcribe_realtime_optimized(
                        accumulated_audio,
                        language="auto"
                    )
                    
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    if result and result.get("text") and len(result["text"].strip()) > 1:
                        # Add conversation context
                        conversation_manager.add_turn(
                            websocket,
                            transcription=result["text"],
                            audio_duration=vad_result.get("duration_ms", 0) / 1000,
                            speech_ratio=vad_result.get("speech_ratio", 0),
                            mode="realtime"
                        )
                        
                        # Send real-time response
                        response = {
                            "type": "transcription",
                            "text": result["text"],
                            "processing_time_ms": processing_time,
                            "vad_triggered": True,
                            "speech_ratio": vad_result.get("speech_ratio", 0),
                            "timestamp": time.time(),
                            "ultimate_realtime": True
                        }
                        
                        await websocket.send_json(response)
                        logger.info(f"✅ Real-time response: {processing_time:.1f}ms - '{result['text']}'")
                    
    except Exception as e:
        logger.error(f"Audio stream handling error: {e}")

async def handle_json_message(websocket: WebSocket, message: dict):
    """ULTIMATE: Handle JSON commands and queries"""
    try:
        command = message.get("command")
        
        if command == "get_status":
            status = {
                "type": "status",
                "vad_active": audio_processor.is_vad_active(),
                "model_loaded": model_manager.is_loaded if model_manager else False,
                "conversation_turns": len(conversation_manager.get_conversation_stats(websocket)),
                "ultimate_realtime": True
            }
            await websocket.send_json(status)
            
        elif command == "reset_conversation":
            conversation_manager.cleanup_conversation(websocket)
            conversation_manager.start_conversation(websocket)
            await websocket.send_json({
                "type": "reset_complete",
                "ultimate_realtime": True
            })
            
        elif command == "understanding_query":
            # Handle understanding queries with context
            query_text = message.get("text", "")
            audio_b64 = message.get("audio")
            
            if audio_b64:
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    context = conversation_manager.get_conversation_context(websocket)
                    
                    if model_manager and model_manager.is_loaded:
                        # Two-step: transcribe then understand
                        start_time = time.time()
                        
                        transcription = await model_manager.transcribe_realtime_optimized(
                            audio_bytes, language="auto"
                        )
                        
                        if transcription and transcription.get("text"):
                            understanding = await model_manager.generate_understanding_optimized(
                                transcribed_text=transcription["text"],
                                user_query=query_text,
                                context=context
                            )
                            
                            processing_time = (time.time() - start_time) * 1000
                            
                            response = {
                                "type": "understanding",
                                "transcription": transcription["text"],
                                "response": understanding.get("response", ""),
                                "processing_time_ms": processing_time,
                                "ultimate_realtime": True
                            }
                            
                            await websocket.send_json(response)
                            
                except Exception as e:
                    await websocket.send_json({
                        "error": f"Understanding query failed: {e}",
                        "ultimate_realtime": True
                    })
                    
    except Exception as e:
        logger.error(f"JSON message handling error: {e}")

if __name__ == "__main__":
    try:
        uvicorn.run(
            "fixed_main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        logger.info("🛑 ULTIMATE server stopped by user")
    except Exception as e:
        logger.error(f"❌ ULTIMATE server error: {e}")
        sys.exit(1)
