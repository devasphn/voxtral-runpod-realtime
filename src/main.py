# FIXED MAIN.PY - UNIFIED APPROACH WITH BULLETPROOF ERROR HANDLING
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import json
import base64

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import torch

from config.settings import Settings
from config.logging_config import setup_logging
from src.websocket_handler import WebSocketManager
from src.model_loader import VoxtralModelManager
from src.audio_processor import AudioProcessor
from src.utils import get_system_info

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global model manager and audio processor
model_manager = None
ws_manager = WebSocketManager()
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    chunk_duration_ms=30
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with robust error handling"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting UNIFIED Voxtral Real-Time Server...")
    
    # Initialize model
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    if model_manager:
        await model_manager.cleanup()
    await audio_processor.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - UNIFIED Real-Time API",
    description="UNIFIED: Transcription & Understanding with bulletproof WebSocket handling",
    version="3.0.0",
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
            "unified_processing": "FFmpeg streaming for both modes",
            "transcription_mode": "ASR only - Speech to Text",
            "understanding_mode": "ASR + LLM - Speech to Intelligent Response",
            "audio_processing": "Unified FFmpeg WebM streaming",
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
        "audio_processing": "Unified FFmpeg WebM -> PCM streaming",
        "transcription_mode": "ASR only - converts speech to text",
        "understanding_mode": "ASR + LLM - speech to intelligent response",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "capabilities": [
            "✅ Pure ASR transcription with unified processing",
            "✅ Audio understanding (speak -> get intelligent reply)",
            "✅ Real-time unified FFmpeg WebM processing",
            "✅ Bulletproof WebSocket error handling",
            "✅ Browser MediaRecorder support",
            "✅ Multi-language support"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """TRANSCRIPTION MODE: Speech -> Text (ASR only) - UNIFIED processing"""
    await ws_manager.connect(websocket, "transcribe")
    
    try:
        while True:
            try:
                # Receive WebM audio from browser with validation
                data = await websocket.receive_bytes()
                
                # Validate data
                if not data or len(data) < 10:
                    logger.warning("Received invalid/empty audio data")
                    continue
                
                # Process WebM chunk through unified FFmpeg streaming
                result = await audio_processor.process_webm_chunk_transcribe(data)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # Only process if we have significant speech
                    if duration_ms > 500 and speech_ratio > 0.3:
                        logger.info(f"🎤 TRANSCRIBING unified processed audio ({duration_ms:.0f}ms, speech: {speech_ratio:.2f})")
                        
                        if model_manager and model_manager.is_loaded:
                            # Use TRANSCRIPTION mode - ASR only
                            transcription_result = await model_manager.transcribe_audio(result["audio_data"])
                            
                            # Send transcription if meaningful and no errors
                            if (transcription_result.get("text") and 
                                "error" not in transcription_result and
                                len(transcription_result["text"].strip()) > 2):
                                
                                await websocket.send_json(transcription_result)
                                logger.info(f"✅ TRANSCRIBED: '{transcription_result['text'][:50]}...'")
                        else:
                            await websocket.send_json({"error": "Model not loaded"})
                elif result and "error" in result:
                    logger.error(f"Audio processing error: {result['error']}")
                    
            except Exception as inner_e:
                logger.error(f"Inner WebSocket transcription error: {inner_e}")
                try:
                    await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break  # Connection likely broken
                        
    except WebSocketDisconnect:
        logger.info("Transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"})
        except:
            pass
    finally:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """UNDERSTANDING MODE: Speech -> Intelligent Response (ASR + LLM) - UNIFIED processing"""
    await ws_manager.connect(websocket, "understand")
    
    try:
        while True:
            try:
                # Receive JSON message with audio and optional query
                message = await websocket.receive_json()
                
                # Validate message format
                if not isinstance(message, dict) or "audio" not in message:
                    await websocket.send_json({"error": "Invalid message format"})
                    continue
                
                audio_data = message.get("audio")
                query = message.get("text", "What can you hear in this audio?")
                
                if not audio_data:
                    await websocket.send_json({"error": "No audio data provided"})
                    continue
                
                # Handle base64 encoded audio from browser
                try:
                    if isinstance(audio_data, str):
                        audio_bytes = base64.b64decode(audio_data)
                    else:
                        audio_bytes = audio_data
                        
                    if len(audio_bytes) < 10:
                        logger.warning("Decoded audio data too small")
                        continue
                        
                except Exception as decode_e:
                    logger.error(f"Failed to decode audio: {decode_e}")
                    await websocket.send_json({"error": f"Audio decode error: {str(decode_e)}"})
                    continue
                
                # Process through unified FFmpeg approach
                result = await audio_processor.process_webm_chunk_understand(audio_bytes)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # Only process if we have significant speech
                    if duration_ms > 1000 and speech_ratio > 0.2:
                        logger.info(f"🎤 UNDERSTANDING unified processed audio ({duration_ms:.0f}ms, speech: {speech_ratio:.2f})")
                        
                        if model_manager and model_manager.is_loaded:
                            # Use UNDERSTANDING mode - ASR + LLM
                            understanding_result = await model_manager.understand_audio(result["audio_data"], query)
                            
                            # Send response if meaningful and no errors
                            if ("response" in understanding_result and 
                                "error" not in understanding_result and
                                len(understanding_result["response"].strip()) > 2):
                                
                                await websocket.send_json(understanding_result)
                                logger.info(f"✅ UNDERSTOOD: '{understanding_result['response'][:50]}...'")
                            else:
                                await websocket.send_json({"error": "No meaningful response generated"})
                        else:
                            await websocket.send_json({"error": "Model not loaded"})
                elif result and "error" in result:
                    logger.error(f"Audio understanding processing error: {result['error']}")
                    await websocket.send_json({"error": result['error']})
                    
            except json.JSONDecodeError as json_e:
                logger.error(f"JSON decode error: {json_e}")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as inner_e:
                logger.error(f"Inner WebSocket understanding error: {inner_e}")
                try:
                    await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break  # Connection likely broken
                
    except WebSocketDisconnect:
        logger.info("Understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket understanding error: {e}")
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(e)}"})
        except:
            pass
    finally:
        ws_manager.disconnect(websocket)

@app.get("/connections")
async def get_connections():
    """Get connection statistics"""
    return {
        "total_connections": ws_manager.connection_count,
        "connections_by_type": ws_manager.get_connections_by_type(),
        "max_connections": settings.MAX_CONCURRENT_CONNECTIONS,
        "unified_audio_stats": audio_processor.get_stats()
    }

@app.post("/audio/reset")
async def reset_audio():
    """Reset audio processor"""
    audio_processor.reset()
    return {"status": "Unified audio processor reset successfully"}

@app.get("/debug/audio")
async def debug_audio():
    """Debug audio processing statistics"""
    return {
        "audio_stats": audio_processor.get_stats(),
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "active_connections": ws_manager.connection_count,
        "memory_usage": model_manager._get_memory_usage() if model_manager and model_manager.is_loaded else {}
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
