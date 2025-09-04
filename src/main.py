# FIXED MAIN.PY - WITH PROPER WEBSOCKET HANDLING AND ERROR RECOVERY
import asyncio
import logging
import signal
import sys
import os
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
from src.conversation_manager import ConversationManager
from src.utils import get_system_info

# FIXED: Import the corrected classes
try:
    from src.fixed_model_loader import VoxtralModelManager  # Use fixed version
except ImportError:
    from src.model_loader import VoxtralModelManager  # Fallback

try:
    from src.fixed_audio_processor import FixedAudioProcessor as AudioProcessor  # Use fixed version
except ImportError:
    from src.audio_processor import AudioProcessor  # Fallback

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global managers
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager(max_turns=30, context_window_minutes=15)
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    chunk_duration_ms=30,
    conversation_manager=conversation_manager
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
    """FIXED: Application lifespan with proper cleanup"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting FIXED Voxtral Real-Time Server...")
    
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ FIXED model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load FIXED model: {e}")
        raise RuntimeError(f"FIXED model loading failed: {e}")
    
    # Start background cleanup
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FIXED server...")
    shutdown_event.set()
    
    # Cancel background tasks
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup managers
    if model_manager:
        await model_manager.cleanup()
    await audio_processor.cleanup()
    
    logger.info("✅ FIXED graceful shutdown completed")

async def background_cleanup():
    """Background maintenance tasks"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            active_connections = ws_manager.connection_count
            logger.debug(f"🔄 FIXED background cleanup: {active_connections} active")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"FIXED background cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - FIXED Real-Time API",
    description="FIXED system with proper error handling and language support",
    version="4.0.1-FIXED",
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
    """FIXED: Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "fixes_applied": [
                "✅ Correct Language Codes (ISO 639-1)",
                "✅ Fixed ThreadPool Shutdown",
                "✅ Improved WebSocket Handling",
                "✅ Better FFmpeg Streaming",
                "✅ Enhanced Error Recovery"
            ],
            "system": system_info,
            "shutdown_requested": shutdown_event.is_set(),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"FIXED health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """FIXED: Model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters", 
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "fixes_applied": "FIXED version with proper language handling",
        "supported_languages": [
            "English (en)", "Spanish (es)", "French (fr)", "Portuguese (pt)", 
            "Hindi (hi)", "German (de)", "Dutch (nl)", "Italian (it)"
        ],
        "fixed_capabilities": [
            "✅ Valid ISO 639-1 language codes",
            "✅ Proper error handling and recovery",
            "✅ Improved audio processing pipeline",
            "✅ Better WebSocket streaming",
            "✅ Enhanced conversation memory"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """FIXED: WebSocket transcription with proper error handling"""
    await ws_manager.connect(websocket, "transcribe")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🎤 FIXED transcription session started")
        
        while not shutdown_event.is_set():
            try:
                # FIXED: Receive audio data with validation
                data = await websocket.receive_bytes()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # FIXED: Better data validation
                if not data or len(data) < 50:  # Minimum WebM chunk size
                    logger.debug("Invalid/insufficient audio data received")
                    continue
                
                # FIXED: Process through corrected audio processor
                result = await audio_processor.process_webm_chunk_transcribe(data, websocket)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # FIXED: More permissive thresholds
                    if duration_ms > 200 and speech_ratio > 0.05:
                        logger.info(f"🎤 FIXED TRANSCRIBING: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                        
                        if model_manager and model_manager.is_loaded:
                            # Get context
                            context = conversation_manager.get_conversation_context(websocket)
                            
                            # FIXED: Use corrected transcription
                            transcription_result = await model_manager.transcribe_audio(
                                result["audio_data"], 
                                context=context,
                                language=None  # FIXED: Let model auto-detect
                            )
                            
                            # FIXED: Better result validation
                            if (transcription_result.get("text") and 
                                "error" not in transcription_result and
                                len(transcription_result["text"].strip()) > 0 and
                                transcription_result["text"].strip() not in ["", ".", "...", "..."]):
                                
                                # Add to conversation
                                conversation_manager.add_turn(
                                    websocket,
                                    transcription=transcription_result["text"],
                                    audio_duration=duration_ms / 1000,
                                    speech_ratio=speech_ratio,
                                    mode="transcribe",
                                    language=transcription_result.get("language")
                                )
                                
                                # Add stats
                                conv_stats = conversation_manager.get_conversation_stats(websocket)
                                transcription_result["conversation"] = conv_stats
                                transcription_result["fixed"] = True
                                
                                await websocket.send_json(transcription_result)
                                logger.info(f"✅ FIXED TRANSCRIBED: '{transcription_result['text']}'")
                            else:
                                logger.debug(f"Skipping empty/invalid transcription: {transcription_result}")
                        else:
                            await websocket.send_json({"error": "Model not loaded"})
                    else:
                        logger.debug(f"Skipping: duration={duration_ms:.0f}ms, speech={speech_ratio:.3f}")
                elif result and "error" in result:
                    logger.error(f"FIXED audio processing error: {result['error']}")
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"FIXED inner WebSocket transcription error: {inner_e}")
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"FIXED processing error: {str(inner_e)}",
                            "fixed": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("FIXED transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"FIXED WebSocket transcription error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """FIXED: WebSocket understanding with proper message handling"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 FIXED understanding session started")
        
        while not shutdown_event.is_set():
            try:
                # FIXED: Receive JSON message
                message = await websocket.receive_json()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # FIXED: Better message validation
                if not isinstance(message, dict) or "audio" not in message:
                    await websocket.send_json({
                        "error": "Invalid message format. Expected: {\"audio\": \"base64_data\", \"text\": \"optional_query\"}",
                        "fixed": True
                    })
                    continue
                
                audio_data = message.get("audio")
                query = message.get("text", "What can you hear in this audio?")
                
                if not audio_data:
                    await websocket.send_json({"error": "No audio data provided", "fixed": True})
                    continue
                
                # FIXED: Handle base64 audio data
                try:
                    if isinstance(audio_data, str):
                        # FIXED: Better base64 decoding
                        if audio_data.startswith("data:"):
                            # Remove data URL prefix if present
                            audio_data = audio_data.split(",")[1] if "," in audio_data else audio_data
                        audio_bytes = base64.b64decode(audio_data)
                    else:
                        audio_bytes = audio_data
                        
                    if len(audio_bytes) < 50:
                        logger.warning("Decoded audio data too small")
                        continue
                        
                except Exception as decode_e:
                    logger.error(f"FIXED audio decode error: {decode_e}")
                    await websocket.send_json({
                        "error": f"FIXED audio decode error: {str(decode_e)}",
                        "fixed": True
                    })
                    continue
                
                # FIXED: Process through corrected processor
                result = await audio_processor.process_webm_chunk_understand(audio_bytes, websocket)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # FIXED: More permissive thresholds
                    if duration_ms > 300 and speech_ratio > 0.05:
                        logger.info(f"🧠 FIXED UNDERSTANDING: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                        
                        if model_manager and model_manager.is_loaded:
                            # Get context
                            context = conversation_manager.get_conversation_context(websocket)
                            
                            # FIXED: Use corrected understanding
                            understanding_result = await model_manager.understand_audio(
                                result["audio_data"], 
                                query=query,
                                context=context
                            )
                            
                            # FIXED: Better result validation
                            if ("response" in understanding_result and 
                                "error" not in understanding_result and
                                len(understanding_result["response"].strip()) > 0):
                                
                                # Add to conversation
                                transcription = understanding_result.get("transcription", "")
                                conversation_manager.add_turn(
                                    websocket,
                                    transcription=transcription,
                                    response=understanding_result["response"],
                                    audio_duration=duration_ms / 1000,
                                    speech_ratio=speech_ratio,
                                    mode="understand",
                                    language=understanding_result.get("language")
                                )
                                
                                # Add stats
                                conv_stats = conversation_manager.get_conversation_stats(websocket)
                                understanding_result["conversation"] = conv_stats
                                understanding_result["fixed"] = True
                                
                                await websocket.send_json(understanding_result)
                                logger.info(f"✅ FIXED UNDERSTOOD: '{understanding_result['response'][:100]}...'")
                            else:
                                logger.warning(f"Invalid understanding result: {understanding_result}")
                                await websocket.send_json({
                                    "error": "No valid understanding generated",
                                    "fixed": True
                                })
                        else:
                            await websocket.send_json({"error": "Model not loaded", "fixed": True})
                    else:
                        logger.debug(f"Skipping understanding: duration={duration_ms:.0f}ms, speech={speech_ratio:.3f}")
                elif result and "error" in result:
                    logger.error(f"FIXED understanding processing error: {result['error']}")
                    await websocket.send_json({"error": result["error"], "fixed": True})
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as json_e:
                logger.error(f"FIXED JSON decode error: {json_e}")
                await websocket.send_json({
                    "error": "Invalid JSON format",
                    "fixed": True,
                    "example": {
                        "audio": "base64_encoded_audio_data",
                        "text": "What can you hear?"
                    }
                })
            except Exception as inner_e:
                logger.error(f"FIXED inner WebSocket understanding error: {inner_e}")
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"FIXED processing error: {str(inner_e)}",
                            "fixed": True
                        })
                except:
                    break
                
    except WebSocketDisconnect:
        logger.info("FIXED understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"FIXED WebSocket understanding error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.get("/conversations")
async def get_conversations():
    """Get conversation statistics"""
    active_conversations = {}
    for conn_id, turns in conversation_manager.conversations.items():
        if turns:
            active_conversations[conn_id] = {
                "turns": len(turns),
                "last_activity": turns[-1].timestamp.isoformat(),
                "languages": list(set(turn.language for turn in turns if turn.language)),
                "total_duration": sum(turn.audio_duration for turn in turns),
                "modes": list(set(turn.mode for turn in turns))
            }
    
    return {
        "active_conversations": len(active_conversations),
        "total_ws_connections": ws_manager.connection_count,
        "conversation_details": active_conversations,
        "system_stats": {
            "max_turns_per_conversation": conversation_manager.max_turns,
            "context_window_minutes": conversation_manager.context_window.total_seconds() / 60,
            "audio_processor_stats": audio_processor.get_stats()
        },
        "fixed": True
    }

@app.post("/conversations/reset")
async def reset_conversations():
    """Reset all conversation data"""
    conversation_manager.conversations.clear()
    conversation_manager.language_patterns.clear()
    conversation_manager.audio_context.clear()
    audio_processor.reset()
    
    return {
        "status": "All conversations and audio processor reset successfully",
        "fixed": True
    }

@app.get("/debug/fixed")
async def debug_fixed():
    """FIXED: Enhanced debug information"""
    return {
        "conversation_manager": {
            "active_sessions": len(conversation_manager.conversations),
            "language_patterns": {k: v[-3:] for k, v in conversation_manager.language_patterns.items()},
        },
        "audio_processor": audio_processor.get_stats(),
        "websocket_manager": ws_manager.get_connection_stats(),
        "model_status": {
            "loaded": model_manager.is_loaded if model_manager else False,
            "memory_usage": model_manager._get_memory_usage() if model_manager and model_manager.is_loaded else {}
        },
        "system_status": {
            "shutdown_requested": shutdown_event.is_set(),
            "background_tasks_active": not shutdown_event.is_set()
        },
        "fixes_applied": [
            "✅ Language code validation (ISO 639-1)",
            "✅ ThreadPool cleanup without timeout", 
            "✅ WebSocket error handling",
            "✅ FFmpeg process management",
            "✅ Audio processing pipeline"
        ],
        "fixed": True
    }

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        logger.info("🛑 FIXED server stopped by user")
    except Exception as e:
        logger.error(f"❌ FIXED server error: {e}")
        sys.exit(1)
