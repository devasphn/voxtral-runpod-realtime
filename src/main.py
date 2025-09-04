# ENHANCED MAIN.PY - WITH CONVERSATION MEMORY AND ROBUST SHUTDOWN
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
from src.model_loader import VoxtralModelManager
from src.audio_processor import AudioProcessor
from src.conversation_manager import ConversationManager  # NEW
from src.utils import get_system_info

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global managers
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager(max_turns=30, context_window_minutes=15)  # NEW
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    chunk_duration_ms=30
)

# Shutdown flag for graceful termination
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with robust error handling and cleanup"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting ENHANCED Voxtral Real-Time Server with Conversation Memory...")
    
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
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    
    # Set shutdown flag
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
    
    logger.info("✅ Graceful shutdown completed")

async def background_cleanup():
    """Background task for periodic cleanup"""
    while not shutdown_event.is_set():
        try:
            # Cleanup old conversation data
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Get connection count for health monitoring
            active_connections = ws_manager.connection_count
            logger.debug(f"🔄 Background cleanup: {active_connections} active connections")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - ENHANCED Real-Time API",
    description="Enhanced system with conversation memory, robust shutdown, and multilingual support",
    version="4.0.0",
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
    """Enhanced health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "enhancements": [
                "✅ Conversation Memory System",
                "✅ Robust Shutdown Handling", 
                "✅ Multilingual Code-Switching Support",
                "✅ Long Conversation Context",
                "✅ Enhanced Speech Detection",
                "✅ WebSocket Health Monitoring"
            ],
            "system": system_info,
            "shutdown_requested": shutdown_event.is_set(),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get enhanced model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters", 
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "conversation_support": "Enhanced with memory and context preservation",
        "multilingual_features": [
            "🌍 Code-switching detection (Hindi-English)",
            "🔄 Language pattern tracking",
            "🗣️ Mixed language conversation handling",
            "🧠 Context-aware responses"
        ],
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian", "Mixed (Hindi-English)"
        ],
        "enhanced_capabilities": [
            "✅ Long conversation memory (up to 15 minutes context)",
            "✅ Conversation turn tracking and analysis", 
            "✅ Code-switching detection and handling",
            "✅ Context-aware responses in understanding mode",
            "✅ Improved speech detection with lower thresholds",
            "✅ Robust WebSocket connection management"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """ENHANCED TRANSCRIPTION: Speech → Text with conversation memory"""
    await ws_manager.connect(websocket, "transcribe")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🎤 Enhanced transcription session started with conversation memory")
        
        while not shutdown_event.is_set():
            try:
                # Receive WebM audio from browser with validation
                data = await websocket.receive_bytes()
                
                # Check for shutdown
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Validate data
                if not data or len(data) < 10:
                    logger.warning("Received invalid/empty audio data")
                    continue
                
                # Process WebM chunk through unified FFmpeg streaming
                result = await audio_processor.process_webm_chunk_transcribe(data)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # ENHANCED: Much more sensitive thresholds for better detection
                    if duration_ms > 200 and speech_ratio > 0.05:  # Even more lenient
                        logger.info(f"🎤 TRANSCRIBING with conversation context ({duration_ms:.0f}ms, speech: {speech_ratio:.3f})")
                        
                        if model_manager and model_manager.is_loaded:
                            # Get conversation context for better transcription
                            context = conversation_manager.get_conversation_context(websocket)
                            
                            # Use TRANSCRIPTION mode - ASR only
                            transcription_result = await model_manager.transcribe_audio(
                                result["audio_data"], 
                                context=context  # Pass conversation context
                            )
                            
                            # Send transcription if meaningful and no errors
                            if (transcription_result.get("text") and 
                                "error" not in transcription_result and
                                len(transcription_result["text"].strip()) > 0):
                                
                                # Add to conversation memory
                                conversation_manager.add_turn(
                                    websocket,
                                    transcription=transcription_result["text"],
                                    audio_duration=duration_ms / 1000,
                                    speech_ratio=speech_ratio,
                                    mode="transcribe",
                                    language=transcription_result.get("language")
                                )
                                
                                # Add conversation stats to response
                                conv_stats = conversation_manager.get_conversation_stats(websocket)
                                transcription_result["conversation"] = conv_stats
                                
                                await websocket.send_json(transcription_result)
                                logger.info(f"✅ TRANSCRIBED with context: '{transcription_result['text']}' (lang: {conv_stats.get('languages', [])})")
                            else:
                                logger.debug(f"Skipping empty transcription: {transcription_result}")
                        else:
                            await websocket.send_json({"error": "Model not loaded"})
                    else:
                        logger.debug(f"Skipping transcription: duration={duration_ms:.0f}ms, speech_ratio={speech_ratio:.3f}")
                elif result and "error" in result:
                    logger.error(f"Audio processing error: {result['error']}")
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"Inner WebSocket transcription error: {inner_e}")
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("Transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """ENHANCED UNDERSTANDING: Speech → Intelligent Response with conversation memory"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 Enhanced understanding session started with conversation memory")
        
        while not shutdown_event.is_set():
            try:
                # Receive JSON message with audio and optional query
                message = await websocket.receive_json()
                
                # Check for shutdown
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Validate message format
                if not isinstance(message, dict) or "audio" not in message:
                    await websocket.send_json({"error": "Invalid message format"})
                    continue
                
                audio_data = message.get("audio")
                query = message.get("text")
                
                # Use conversation-aware query if none provided
                if not query:
                    query = conversation_manager.get_suggested_prompt(websocket)
                
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
                    
                    # ENHANCED: More sensitive thresholds
                    if duration_ms > 300 and speech_ratio > 0.05:
                        logger.info(f"🧠 UNDERSTANDING with conversation context ({duration_ms:.0f}ms, speech: {speech_ratio:.3f})")
                        
                        if model_manager and model_manager.is_loaded:
                            # Get conversation context
                            context = conversation_manager.get_conversation_context(websocket)
                            
                            # Use UNDERSTANDING mode - ASR + LLM with context
                            understanding_result = await model_manager.understand_audio(
                                result["audio_data"], 
                                query=query,
                                context=context  # Pass conversation context
                            )
                            
                            # Send response if meaningful and no errors
                            if ("response" in understanding_result and 
                                "error" not in understanding_result and
                                len(understanding_result["response"].strip()) > 0):
                                
                                # Extract transcription from understanding if available
                                transcription = understanding_result.get("transcription", "")
                                
                                # Add to conversation memory
                                conversation_manager.add_turn(
                                    websocket,
                                    transcription=transcription,
                                    response=understanding_result["response"],
                                    audio_duration=duration_ms / 1000,
                                    speech_ratio=speech_ratio,
                                    mode="understand",
                                    language=understanding_result.get("language")
                                )
                                
                                # Add conversation stats to response
                                conv_stats = conversation_manager.get_conversation_stats(websocket)
                                understanding_result["conversation"] = conv_stats
                                
                                await websocket.send_json(understanding_result)
                                logger.info(f"✅ UNDERSTOOD with context: '{understanding_result['response'][:100]}...' (turns: {conv_stats['turns']})")
                            else:
                                logger.warning(f"No valid understanding: {understanding_result}")
                        else:
                            await websocket.send_json({"error": "Model not loaded"})
                    else:
                        logger.debug(f"Skipping understanding: duration={duration_ms:.0f}ms, speech_ratio={speech_ratio:.3f}")
                elif result and "error" in result:
                    logger.error(f"Audio understanding processing error: {result['error']}")
                    await websocket.send_json({"error": result['error']})
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as json_e:
                logger.error(f"JSON decode error: {json_e}")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as inner_e:
                logger.error(f"Inner WebSocket understanding error: {inner_e}")
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break
                
    except WebSocketDisconnect:
        logger.info("Understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket understanding error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.get("/conversations")
async def get_conversations():
    """Get conversation statistics and active sessions"""
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
        }
    }

@app.post("/conversations/reset")
async def reset_conversations():
    """Reset all conversation data"""
    conversation_manager.conversations.clear()
    conversation_manager.language_patterns.clear()
    conversation_manager.audio_context.clear()
    audio_processor.reset()
    
    return {"status": "All conversations and audio processor reset successfully"}

@app.get("/debug/enhanced")
async def debug_enhanced():
    """Enhanced debug information"""
    return {
        "conversation_manager": {
            "active_sessions": len(conversation_manager.conversations),
            "language_patterns": {k: v[-3:] for k, v in conversation_manager.language_patterns.items()},  # Last 3 languages per session
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
        }
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
            timeout_graceful_shutdown=30  # 30 seconds for graceful shutdown
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
