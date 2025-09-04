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
    chunk_duration_ms=30,
    vad_mode=3
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting Voxtral Real-Time Server with FIXED audio processing...")
    
    # Initialize model
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ Model loaded successfully with mistral_common integration")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Voxtral Real-Time Server...")
    if model_manager:
        await model_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - FIXED Real-Time API",
    description="Fixed real-time audio processing with proper Voxtral integration",
    version="1.1.0",
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
            "vad_enabled": True,
            "mistral_common_integration": True,
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
        "vad_enabled": True,
        "vad_mode": 3,
        "audio_format": "WebM/Opus -> WAV conversion",
        "mistral_common": "1.8.1+",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "capabilities": [
            "Real-time speech transcription",
            "Audio understanding with Q&A", 
            "Voice Activity Detection",
            "WebM/Opus audio processing",
            "Multi-language support"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """FIXED: WebSocket endpoint for real-time transcription"""
    await ws_manager.connect(websocket, "transcribe")
    try:
        while True:
            # Receive audio data from browser (WebM/Opus format)
            data = await websocket.receive_bytes()
            
            # IMPROVED: Better VAD processing for WebM audio
            try:
                vad_result = audio_processor.process_chunk(data)
                
                if vad_result and "audio_data" in vad_result and vad_result.get("speech_ratio", 0) > 0.3:
                    # Only process if significant speech detected
                    logger.info(f"🎤 Speech detected (ratio: {vad_result.get('speech_ratio', 0):.2f}), processing...")
                    
                    if model_manager and model_manager.is_loaded:
                        # Use the fixed model with proper WebM -> WAV conversion
                        result = await model_manager.transcribe_audio(vad_result["audio_data"])
                        
                        # Send result if transcription is meaningful
                        if result.get("text") and result["text"].strip() and not result["text"].startswith("I'm sorry"):
                            await websocket.send_json(result)
                            logger.info(f"✅ Real transcription: {result['text'][:50]}...")
                        else:
                            logger.info("⚠️ Generic response, audio quality may be low")
                    else:
                        await websocket.send_json({"error": "Model not loaded"})
                        
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                # Continue processing, don't break connection
                
    except WebSocketDisconnect:
        logger.info("Transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """FIXED: WebSocket endpoint for audio understanding"""
    await ws_manager.connect(websocket, "understand")
    try:
        while True:
            # Receive JSON message with audio and question
            message = await websocket.receive_json()
            
            if model_manager and model_manager.is_loaded:
                # Use the fixed model with proper audio understanding
                result = await model_manager.understand_audio(message)
                await websocket.send_json(result)
                
                if not result.get("error"):
                    logger.info(f"✅ Understanding response: {result.get('response', '')[:50]}...")
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
    """Get active WebSocket connections info"""
    return {
        "total_connections": ws_manager.connection_count,
        "connections_by_type": ws_manager.get_connections_by_type(),
        "max_connections": settings.MAX_CONCURRENT_CONNECTIONS,
        "vad_stats": audio_processor.get_stats()
    }

@app.post("/vad/reset")
async def reset_vad():
    """Reset VAD processor"""
    audio_processor.reset()
    return {"status": "VAD reset successfully"}

@app.get("/debug/audio-test")
async def test_audio_processing():
    """Debug endpoint to test audio processing"""
    if not model_manager or not model_manager.is_loaded:
        return {"error": "Model not loaded"}
    
    try:
        # Create test audio (sine wave)
        import numpy as np
        duration = 3  # 3 seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_int16 = (audio_data * 16383).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Test transcription
        result = await model_manager.transcribe_audio(audio_bytes)
        
        return {
            "test_status": "completed",
            "audio_duration": duration,
            "audio_bytes": len(audio_bytes),
            "result": result
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
