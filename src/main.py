# PURE UNDERSTANDING-ONLY MAIN APPLICATION - NO TRANSCRIPTION
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

# FIXED IMPORT - No circular import, PURE UNDERSTANDING-ONLY
from src.model_loader import VoxtralUnderstandingManager

try:
    from src.audio_processor import UnderstandingAudioProcessor as AudioProcessor
except ImportError:
    from src.audio_processor import AudioProcessor

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global managers - PURE UNDERSTANDING-ONLY (NO TRANSCRIPTION)
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
    """PURE UNDERSTANDING-ONLY: Application lifespan with proper cleanup"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting PURE UNDERSTANDING-ONLY Voxtral Real-Time Server...")
    logger.info("🚫 Transcription functionality: COMPLETELY DISABLED")
    
    try:
        model_manager = VoxtralUnderstandingManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ PURE UNDERSTANDING-ONLY model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load PURE UNDERSTANDING-ONLY model: {e}")
        raise RuntimeError(f"PURE UNDERSTANDING-ONLY model loading failed: {e}")
    
    # Start background cleanup
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PURE UNDERSTANDING-ONLY server...")
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
    
    logger.info("✅ PURE UNDERSTANDING-ONLY graceful shutdown completed")

async def background_cleanup():
    """Background maintenance tasks"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            active_connections = ws_manager.connection_count
            logger.debug(f"🔄 PURE UNDERSTANDING-ONLY background cleanup: {active_connections} active")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"PURE UNDERSTANDING-ONLY background cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - PURE UNDERSTANDING-ONLY Real-Time API",
    description="PURE UNDERSTANDING-ONLY system with 0.3s gap detection and sub-200ms response",
    version="2.0.0-PURE-UNDERSTANDING-ONLY",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the main PURE UNDERSTANDING-ONLY client page"""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """PURE UNDERSTANDING-ONLY: Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "mode": "PURE UNDERSTANDING-ONLY",
            "transcription_status": "COMPLETELY DISABLED",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "gap_detection_ms": 300,
            "target_response_ms": 200,
            "flash_attention": "DISABLED (compatibility fix)",
            "features": [
                "✅ PURE UNDERSTANDING-ONLY mode with conversational AI responses",
                "🚫 Transcription functionality: COMPLETELY DISABLED",
                "✅ 0.3-second gap detection using WebRTC VAD",
                "✅ Sub-200ms response time optimization",
                "✅ Continuous audio processing with speech boundaries",
                "✅ Context-aware conversation memory",
                "✅ Enhanced audio processing for human speech",
                "✅ Flash Attention compatibility fix applied"
            ],
            "system": system_info,
            "shutdown_requested": shutdown_event.is_set(),
            "timestamp": asyncio.get_event_loop().time(),
            "understanding_only": True,
            "transcription_disabled": True
        }
    except Exception as e:
        logger.error(f"PURE UNDERSTANDING-ONLY health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """PURE UNDERSTANDING-ONLY: Model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters", 
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "mode": "PURE UNDERSTANDING-ONLY",
        "transcription_status": "COMPLETELY DISABLED",
        "gap_detection_ms": 300,
        "target_response_ms": 200,
        "flash_attention_status": "DISABLED (compatibility fix)",
        "understanding_only": True,
        "transcription_disabled": True,
        "supported_languages": [
            "English (en)", "Spanish (es)", "French (fr)", "Portuguese (pt)", 
            "Hindi (hi)", "German (de)", "Dutch (nl)", "Italian (it)"
        ],
        "features": {
            "gap_detection": {
                "threshold_ms": 300,
                "method": "WebRTC VAD",
                "automatic": True
            },
            "understanding": {
                "purpose": "Conversational AI responses to audio input",
                "temperature": 0.3,
                "max_tokens": 200,
                "context_memory": True
            },
            "performance": {
                "target_response_ms": 200,
                "optimization": "Eager Attention (Flash Attention disabled)",
                "caching": True
            },
            "transcription": {
                "status": "COMPLETELY DISABLED",
                "reason": "Pure understanding-only implementation"
            }
        },
        "audio_processing": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "WebM → PCM → WAV",
            "vad_enabled": True
        }
    }

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """PURE UNDERSTANDING-ONLY: Single WebSocket endpoint for conversational AI"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🧠 PURE UNDERSTANDING-ONLY session started")
        
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
                
                # Process through PURE UNDERSTANDING-ONLY audio processor
                result = await audio_processor.process_audio_understanding(audio_data, websocket)
                
                if result and isinstance(result, dict):
                    if "error" in result:
                        logger.error(f"PURE UNDERSTANDING-ONLY audio processing error: {result['error']}")
                        await websocket.send_json({
                            "error": f"Audio processing failed: {result['error']}", 
                            "understanding_only": True,
                            "transcription_disabled": True
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
                            "transcription_disabled": True
                        })
                        continue
                    
                    # Process complete speech segment
                    if result.get("speech_complete") and "audio_data" in result:
                        duration_ms = result.get("duration_ms", 0)
                        speech_quality = result.get("speech_quality", 0)
                        
                        # Quality check for understanding
                        if duration_ms > 500 and speech_quality > 0.3:  # At least 0.5s, decent quality
                            logger.info(f"🧠 PURE UNDERSTANDING-ONLY processing: {duration_ms:.0f}ms, quality: {speech_quality:.3f}")
                            
                            if model_manager and model_manager.is_loaded:
                                # Get conversation context
                                context = conversation_manager.get_conversation_context(websocket)
                                
                                # Generate understanding response
                                understanding_result = await model_manager.generate_understanding_response(
                                    result["audio_data"], 
                                    context=context,
                                    optimize_for_speed=True
                                )
                                
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
                                        "flash_attention_disabled": True,
                                        "sub_200ms": response_time_ms < 200,
                                        "timestamp": asyncio.get_event_loop().time()
                                    }
                                    
                                    # Add conversation stats
                                    conv_stats = conversation_manager.get_conversation_stats(websocket)
                                    final_result["conversation"] = conv_stats
                                    
                                    await websocket.send_json(final_result)
                                    logger.info(f"✅ PURE UNDERSTANDING-ONLY complete: '{understanding_result['response'][:50]}...' ({response_time_ms:.0f}ms)")
                                else:
                                    logger.warning(f"Invalid understanding result: {understanding_result}")
                                    await websocket.send_json({
                                        "error": "Failed to generate understanding response",
                                        "understanding_only": True,
                                        "transcription_disabled": True
                                    })
                            else:
                                await websocket.send_json({
                                    "error": "Model not loaded",
                                    "understanding_only": True,
                                    "transcription_disabled": True
                                })
                        else:
                            logger.debug(f"Skipping low quality: duration={duration_ms:.0f}ms, quality={speech_quality:.3f}")
                    
            except WebSocketDisconnect:
                break
            except Exception as inner_e:
                logger.error(f"PURE UNDERSTANDING-ONLY inner WebSocket error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "understanding_only": True,
                            "transcription_disabled": True
                        })
                except:
                    break
                        
    except WebSocketDisconnect:
        logger.info("PURE UNDERSTANDING-ONLY WebSocket disconnected")
    except Exception as e:
        logger.error(f"PURE UNDERSTANDING-ONLY WebSocket error: {e}")
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
        "mode": "PURE UNDERSTANDING-ONLY",
        "transcription_status": "COMPLETELY DISABLED",
        "active_conversations": len(active_conversations),
        "total_ws_connections": ws_manager.connection_count,
        "conversation_details": active_conversations,
        "system_stats": {
            "max_turns_per_conversation": conversation_manager.max_turns,
            "context_window_minutes": conversation_manager.context_window.total_seconds() / 60,
            "audio_processor_stats": audio_processor.get_stats(),
            "gap_detection_ms": 300,
            "target_response_ms": 200,
            "flash_attention_disabled": True
        },
        "understanding_only": True,
        "transcription_disabled": True
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
        "understanding_only": True,
        "transcription_disabled": True
    }

@app.get("/debug/understanding-only")
async def debug_understanding_only():
    """PURE UNDERSTANDING-ONLY: Enhanced debug information"""
    return {
        "mode": "PURE UNDERSTANDING-ONLY",
        "transcription_status": "COMPLETELY DISABLED",
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
            "default_language": getattr(model_manager, 'default_language', 'en') if model_manager else 'en',
            "flash_attention_disabled": True
        },
        "system_status": {
            "shutdown_requested": shutdown_event.is_set(),
            "background_tasks_active": not shutdown_event.is_set(),
            "gap_detection_ms": 300,
            "target_response_ms": 200,
            "flash_attention_fix_applied": True
        },
        "pure_understanding_only_features": [
            "✅ Single WebSocket endpoint: /ws/understand",
            "🚫 Transcription endpoints: COMPLETELY REMOVED",
            "✅ 0.3-second gap detection with WebRTC VAD",
            "✅ Sub-200ms response time optimization", 
            "✅ Conversational AI responses with context memory",
            "✅ Enhanced audio processing for human speech",
            "✅ Continuous recording with automatic speech boundaries",
            "✅ Real-time feedback with gap detection status",
            "✅ Flash Attention compatibility fix applied"
        ],
        "understanding_only": True,
        "transcription_disabled": True
    }

if __name__ == "__main__":
    try:
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            access_log=True,
            timeout_graceful_shutdown=30
        )
    except KeyboardInterrupt:
        logger.info("🛑 PURE UNDERSTANDING-ONLY server stopped by user")
    except Exception as e:
        logger.error(f"❌ PURE UNDERSTANDING-ONLY server error: {e}")
        sys.exit(1)
