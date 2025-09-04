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
    logger.info("🚀 Starting FIXED Voxtral Real-Time Server...")
    
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

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - PERFECT Real-Time API",
    description="TRANSCRIPTION = ASR | UNDERSTANDING = ASR + LLM",
    version="1.2.0",
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
            "vad_enabled": True,
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
        "transcription_mode": "ASR only - converts speech to text",
        "understanding_mode": "ASR + LLM - speech to intelligent response (no text needed!)",
        "audio_processing": "WebM/Opus -> 16kHz WAV",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "capabilities": [
            "✅ Pure ASR transcription",
            "✅ Audio understanding (speak -> get intelligent reply)",
            "✅ Real-time Voice Activity Detection",
            "✅ WebM/Opus browser audio support",
            "✅ Multi-language support"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """TRANSCRIPTION MODE: Speech -> Text (ASR only)"""
    await ws_manager.connect(websocket, "transcribe")
    try:
        while True:
            # Receive WebM/Opus audio from browser
            data = await websocket.receive_bytes()
            
            # VAD processing
            vad_result = audio_processor.process_chunk(data)
            
            if vad_result and "audio_data" in vad_result:
                speech_ratio = vad_result.get("speech_ratio", 0)
                if speech_ratio > 0.2:  # Speech detected
                    logger.info(f"🎤 TRANSCRIBING (speech ratio: {speech_ratio:.2f})")
                    
                    if model_manager and model_manager.is_loaded:
                        # Use TRANSCRIPTION mode - ASR only
                        result = await model_manager.transcribe_audio(vad_result["audio_data"])
                        
                        # Send transcription if meaningful
                        if (result.get("text") and 
                            result["text"].strip() and 
                            not any(phrase in result["text"].lower() for phrase in 
                                   ["i'm sorry", "could not", "didn't catch"])):
                            
                            await websocket.send_json(result)
                            logger.info(f"✅ TRANSCRIBED: '{result['text'][:50]}...'")
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
    """UNDERSTANDING MODE: Speech -> Intelligent Response (ASR + LLM)"""
    await ws_manager.connect(websocket, "understand")
    
    # Audio buffer for understanding mode
    audio_buffer = []
    
    try:
        while True:
            # Receive message
            message = await websocket.receive_json()
            
            # Handle different message types
            if message.get("type") == "audio_chunk":
                # Store audio chunks
                audio_data = message.get("audio")
                if audio_data:
                    audio_buffer.append(audio_data)
                
            elif message.get("type") == "process":
                # Process accumulated audio for understanding
                if audio_buffer:
                    # Combine all audio chunks
                    combined_audio = "".join(audio_buffer)
                    
                    # UNDERSTANDING MODE: Just audio, no text needed!
                    understanding_message = {
                        "audio": combined_audio
                        # No text field needed - the model will understand and respond intelligently!
                    }
                    
                    if model_manager and model_manager.is_loaded:
                        logger.info(f"🧠 UNDERSTANDING audio (ASR + LLM processing)...")
                        result = await model_manager.understand_audio(understanding_message)
                        await websocket.send_json(result)
                        
                        if not result.get("error"):
                            logger.info(f"✅ INTELLIGENT RESPONSE: '{result.get('response', '')[:50]}...'")
                    else:
                        await websocket.send_json({"error": "Model not loaded"})
                    
                    # Clear buffer
                    audio_buffer.clear()
                else:
                    await websocket.send_json({"error": "No audio data to process"})
            
            else:
                # Handle direct audio + text (legacy format)
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
        "vad_stats": audio_processor.get_stats()
    }

@app.post("/vad/reset")
async def reset_vad():
    """Reset VAD processor"""
    audio_processor.reset()
    return {"status": "VAD reset successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )
