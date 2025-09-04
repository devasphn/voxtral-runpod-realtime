import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

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
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting PERFECT Voxtral Real-Time Server...")
    
    # Initialize model
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ Model loaded with CORRECT processor!")
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
    title="Voxtral Mini 3B - PERFECT Real-Time API",
    description="TRANSCRIPTION = ASR | UNDERSTANDING = ASR + LLM",
    version="2.0.0",
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
            "transcription_mode": "ASR only - Speech to Text",
            "understanding_mode": "ASR + LLM - Speech to Intelligent Response",
            "audio_processing": "FFmpeg WebM streaming",
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
        "audio_processing": "FFmpeg WebM -> PCM streaming",
        "transcription_mode": "ASR only - converts speech to text",
        "understanding_mode": "ASR + LLM - speech to intelligent response",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "capabilities": [
            "✅ Pure ASR transcription",
            "✅ Audio understanding (speak -> get intelligent reply)",
            "✅ Real-time FFmpeg WebM processing",
            "✅ Browser MediaRecorder support",
            "✅ Multi-language support"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """TRANSCRIPTION MODE: Speech -> Text (ASR only) - FFmpeg streaming"""
    await ws_manager.connect(websocket, "transcribe")
    
    try:
        while True:
            # Receive WebM audio from browser
            data = await websocket.receive_bytes()
            
            # Process WebM chunk through FFmpeg streaming
            result = await audio_processor.process_webm_chunk(data)
            
            if result and "audio_data" in result:
                duration_ms = result.get("duration_ms", 0)
                if duration_ms > 500:  # Process audio longer than 500ms
                    logger.info(f"🎤 TRANSCRIBING FFmpeg processed audio ({duration_ms:.0f}ms)")
                    
                    if model_manager and model_manager.is_loaded:
                        # Use TRANSCRIPTION mode - ASR only
                        transcription_result = await model_manager.transcribe_audio(result["audio_data"])
                        
                        # Send transcription if meaningful
                        if (transcription_result.get("text") and 
                            transcription_result["text"].strip() and 
                            not any(phrase in transcription_result["text"].lower() for phrase in 
                                   ["i'm sorry", "could not", "didn't catch", "can't understand"])):
                            
                            await websocket.send_json(transcription_result)
                            logger.info(f"✅ TRANSCRIBED: '{transcription_result['text'][:50]}...'")
                    else:
                        await websocket.send_json({"error": "Model not loaded"})
                        
    except WebSocketDisconnect:
        logger.info("Transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """UNDERSTANDING MODE: Speech -> Intelligent Response (ASR + LLM) - FFmpeg streaming"""
    await ws_manager.connect(websocket, "understand")
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            # Handle direct audio + text format
            if model_manager and model_manager.is_loaded:
                result = await model_manager.understand_audio(message)
                await websocket.send_json(result)
            else:
                await websocket.send_json({"error": "Model not loaded"})
                
    except WebSocketDisconnect:
        logger.info("Understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket understanding error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
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
        "audio_stats": audio_processor.get_stats()
    }

@app.post("/audio/reset")
async def reset_audio():
    """Reset audio processor"""
    audio_processor.reset()
    return {"status": "Audio processor reset successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
