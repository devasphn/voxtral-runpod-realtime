# VOXTRAL UNDERSTANDING-ONLY REAL-TIME STREAMING - COMPLETE SOLUTION
import asyncio
import logging
import signal
import sys
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import json
import base64
import time

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

# Import the UNDERSTANDING-ONLY model manager
from src.model_loader import VoxtralUnderstandingManager

# Import UNDERSTANDING-ONLY audio processor
try:
    from src.audio_processor import UnderstandingAudioProcessor as AudioProcessor
except ImportError:
    from src.audio_processor import AudioProcessor

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global managers - UNDERSTANDING ONLY
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager(max_turns=30, context_window_minutes=15)
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    gap_threshold_ms=300,  # 0.3 second gap detection
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
    """Application lifespan - UNDERSTANDING ONLY"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting Voxtral UNDERSTANDING-ONLY Real-Time Server...")
    
    try:
        model_manager = VoxtralUnderstandingManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ UNDERSTANDING-ONLY model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load UNDERSTANDING-ONLY model: {e}")
        raise RuntimeError(f"UNDERSTANDING-ONLY model loading failed: {e}")
    
    # Start background cleanup
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down UNDERSTANDING-ONLY server...")
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
    
    logger.info("✅ UNDERSTANDING-ONLY graceful shutdown completed")

async def background_cleanup():
    """Background maintenance tasks"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            active_connections = ws_manager.connection_count
            logger.debug(f"🔄 UNDERSTANDING-ONLY background cleanup: {active_connections} active")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"UNDERSTANDING-ONLY background cleanup error: {e}")

# Create FastAPI app - UNDERSTANDING ONLY
app = FastAPI(
    title="Voxtral Mini 3B - UNDERSTANDING-ONLY Real-Time API",
    description="UNDERSTANDING-ONLY system with 0.3s gap detection and sub-200ms response",
    version="7.0.0-UNDERSTANDING-ONLY",
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
    """UNDERSTANDING-ONLY: Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "mode": "UNDERSTANDING_ONLY",
            "features": [
                "✅ UNDERSTANDING-ONLY: Pure conversational AI responses to audio",
                "✅ 0.3-second gap detection for natural speech boundaries", 
                "✅ Sub-200ms response time optimization",
                "✅ Enhanced audio processing for human speech",
                "✅ Fixed WebSocket message handling",
                "✅ Proper conversation context management"
            ],
            "system": system_info,
            "shutdown_requested": shutdown_event.is_set(),
            "timestamp": asyncio.get_event_loop().time(),
            "understanding_only": True
        }
    except Exception as e:
        logger.error(f"UNDERSTANDING-ONLY health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """UNDERSTANDING-ONLY: Model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters", 
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "understanding_only": True,
        "supported_languages": [
            "English (en)", "Spanish (es)", "French (fr)", "Portuguese (pt)", 
            "Hindi (hi)", "German (de)", "Dutch (nl)", "Italian (it)"
        ],
        "mode_details": {
            "understanding": {
                "purpose": "Conversational AI responses to audio with context",
                "output": "AI assistant responses to user speech with conversation history",
                "temperature": 0.3,
                "api_method": "apply_chat_template with conversation context",
                "gap_detection": "0.3 seconds for natural speech boundaries",
                "response_target": "Sub-200ms for real-time interaction"
            }
        },
        "optimizations": [
            "✅ 0.3-second gap detection using WebRTC VAD",
            "✅ Sub-200ms response time optimization",
            "✅ Enhanced audio processing pipeline for human speech",
            "✅ Conversation context management for better responses",
            "✅ Proper WebSocket message handling and error recovery"
        ]
    }

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """UNDERSTANDING-ONLY: WebSocket for conversational AI responses to audio"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 UNDERSTANDING-ONLY session started")
        
        while not shutdown_event.is_set():
            try:
                # Receive binary audio data directly
                audio_data = await websocket.receive_bytes()
                
                if shutdown_event.is_set():
                    await websocket.send_json({"info": "Server shutting down"})
                    break
                
                # Process audio with 0.3s gap detection
                if not audio_data or len(audio_data) < 1000:  # Minimum audio size
                    logger.debug("Insufficient audio data received")
                    continue
                
                # Process through UNDERSTANDING-ONLY audio processor
                result = await audio_processor.process_audio_understanding(audio_data, websocket)
                
                if result and isinstance(result, dict):
                    if "error" in result:
                        logger.error(f"UNDERSTANDING-ONLY audio processing error: {result['error']}")
                        await websocket.send_json({
                            "error": f"Audio processing failed: {result['error']}", 
                            "understanding_only": True
                        })
                        continue
                    
                    # Check if we have a complete speech segment (0.3s gap detected)
                    if "speech_complete" in result and result["speech_complete"]:
                        duration_ms = result.get("duration_ms", 0)
                        speech_ratio = result.get("speech_ratio", 0)
                        
                        # Quality thresholds for understanding
                        if duration_ms > 500 and speech_ratio > 0.3:  # At least 0.5 second, good speech quality
                            logger.info(f"🧠 UNDERSTANDING: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                            
                            if model_manager and model_manager.is_loaded:
                                # Get conversation context for better responses
                                context = conversation_manager.get_conversation_context(websocket)
                                
                                # Generate understanding response using conversation context
                                understanding_result = await model_manager.generate_understanding_response(
                                    audio_data=result["audio_data"],
                                    context=context,
                                    optimize_for_speed=True  # Sub-200ms target
                                )
                                
                                if (isinstance(understanding_result, dict) and 
                                    understanding_result.get("response") and 
                                    "error" not in understanding_result and
                                    len(understanding_result["response"].strip()) > 5):
                                    
                                    transcribed_text = understanding_result.get("transcribed_text", "")
                                    response = understanding_result["response"]
                                    
                                    # Create final result
                                    final_result = {
                                        "type": "understanding",
                                        "transcription": transcribed_text,
                                        "response": response,
                                        "timestamp": asyncio.get_event_loop().time(),
                                        "language": understanding_result.get("language", "en"),
                                        "response_time_ms": understanding_result.get("processing_time_ms", 0),
                                        "understanding_only": True,
                                        "gap_detected": True
                                    }
                                    
                                    # Add to conversation for context
                                    conversation_manager.add_turn(
                                        websocket,
                                        transcription=transcribed_text,
                                        response=response,
                                        audio_duration=duration_ms / 1000,
                                        speech_ratio=speech_ratio,
                                        mode="understand",
                                        language=understanding_result.get("language", "en")
                                    )
                                    
                                    # Add stats and send response
                                    conv_stats = conversation_manager.get_conversation_stats(websocket)
                                    final_result["conversation"] = conv_stats
                                    
                                    await websocket.send_json(final_result)
                                    logger.info(f"✅ UNDERSTANDING RESPONSE: '{transcribed_text}' → '{response[:50]}...' ({understanding_result.get('processing_time_ms', 0)}ms)")
                                else:
                                    logger.warning(f"Invalid understanding result: {understanding_result}")
                                    await websocket.send_json({
                                        "error": "Failed to generate understanding response",
                                        "understanding_only": True
                                    })
                            else:
                                await websocket.send_json({
                                    "error": "Model not loaded", 
                                    "understanding_only": True
                                })
                        else:
                            logger.debug(f"Skipping: duration={duration_ms:.0f}ms, speech={speech_ratio:.3f}")
                    
                    # Send intermediate feedback for continuous audio
                    elif "audio_received" in result:
                        await websocket.send_json({
                            "type": "audio_received",
                            "duration_ms": result.get("duration_ms", 0),
                            "speech_ratio": result.get("speech_ratio", 0),
                            "gap_detected": False,
                            "understanding_only": True
                        })
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"UNDERSTANDING-ONLY inner WebSocket error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "understanding_only": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("UNDERSTANDING-ONLY WebSocket disconnected")
    except Exception as e:
        logger.error(f"UNDERSTANDING-ONLY WebSocket error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.get("/conversations")
async def get_conversations():
    """Get conversation statistics - UNDERSTANDING ONLY"""
    active_conversations = {}
    for conn_id, turns in conversation_manager.conversations.items():
        if turns:
            active_conversations[conn_id] = {
                "turns": len(turns),
                "last_activity": turns[-1].timestamp.isoformat(),
                "languages": list(set(turn.language for turn in turns if turn.language)),
                "total_duration": sum(turn.audio_duration for turn in turns),
                "modes": ["understand"]  # Only understanding mode
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
        "understanding_only": True,
        "gap_detection_ms": 300
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
        "understanding_only": True
    }

@app.get("/debug/understanding-only")
async def debug_understanding_only():
    """UNDERSTANDING-ONLY: Enhanced debug information"""
    return {
        "conversation_manager": {
            "active_sessions": len(conversation_manager.conversations),
            "language_patterns": {k: v[-3:] for k, v in conversation_manager.language_patterns.items()},
        },
        "audio_processor": audio_processor.get_stats(),
        "websocket_manager": ws_manager.get_connection_stats(),
        "model_status": {
            "loaded": model_manager.is_loaded if model_manager else False,
            "memory_usage": model_manager._get_memory_usage() if model_manager and model_manager.is_loaded else {},
            "supported_languages": model_manager.supported_languages if model_manager else {},
            "default_language": getattr(model_manager, 'default_language', 'en') if model_manager else 'en'
        },
        "system_status": {
            "shutdown_requested": shutdown_event.is_set(),
            "background_tasks_active": not shutdown_event.is_set()
        },
        "understanding_only_features": [
            "✅ UNDERSTANDING: Conversational AI responses to audio input",
            "✅ 0.3-SECOND GAP: Natural speech boundary detection",
            "✅ SUB-200MS: Optimized response time for real-time interaction",
            "✅ CONTEXT: Conversation history for better responses",
            "✅ AUDIO: Enhanced processing pipeline for human speech",
            "✅ WEBSOCKET: Fixed message handling and error recovery",
            "✅ MULTILINGUAL: Support for 8+ languages with auto-detection"
        ],
        "understanding_only": True,
        "gap_detection_ms": 300,
        "target_response_ms": 200
    }

if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",  # Updated to match this file
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        logger.info("🛑 UNDERSTANDING-ONLY server stopped by user")
    except Exception as e:
        logger.error(f"❌ UNDERSTANDING-ONLY server error: {e}")
        sys.exit(1)
