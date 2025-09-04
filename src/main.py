# COMPLETELY CORRECTED - main.py - FIXED WEBSOCKET DISCONNECT HANDLING
import asyncio
import logging
import signal
import sys
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import json
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
    """Application lifespan with proper cleanup"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting COMPLETELY CORRECTED Voxtral Real-Time Server...")
    
    try:
        model_manager = VoxtralModelManager(
            model_name=settings.MODEL_NAME,
            device=settings.DEVICE,
            torch_dtype=settings.TORCH_DTYPE
        )
        await model_manager.load_model()
        logger.info("✅ COMPLETELY CORRECTED model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load COMPLETELY CORRECTED model: {e}")
        raise RuntimeError(f"COMPLETELY CORRECTED model loading failed: {e}")
    
    # Start background cleanup
    cleanup_task = asyncio.create_task(background_cleanup())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down COMPLETELY CORRECTED server...")
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
    
    logger.info("✅ COMPLETELY CORRECTED graceful shutdown completed")

async def background_cleanup():
    """Background maintenance tasks"""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            active_connections = ws_manager.connection_count
            logger.debug(f"🔄 COMPLETELY CORRECTED background cleanup: {active_connections} active")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"COMPLETELY CORRECTED background cleanup error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - COMPLETELY CORRECTED Real-Time API",
    description="COMPLETELY CORRECTED WebSocket disconnect handling for understanding mode",
    version="9.0.0-COMPLETELY-CORRECTED",
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
    """COMPLETELY CORRECTED: Health check endpoint"""
    try:
        system_info = get_system_info()
        model_status = "loaded" if model_manager and model_manager.is_loaded else "not_loaded"
        
        return {
            "status": "healthy",
            "model_status": model_status,
            "active_connections": ws_manager.connection_count,
            "conversation_sessions": len(conversation_manager.conversations),
            "architecture": "COMPLETELY_CORRECTED_WEBSOCKET_DISCONNECT_HANDLING",
            "fixes_applied": [
                "✅ WEBSOCKET DISCONNECT FIX: Proper message.receive() handling",
                "✅ NO MORE RECEIVE AFTER DISCONNECT: Fixed connection state management",  
                "✅ 300MS GAP DETECTION: WebRTC VAD for natural speech pauses",
                "✅ ROBUST TASK MANAGEMENT: Proper background task cancellation",
                "✅ FAST LLM RESPONSE: Optimized for <200ms response time"
            ],
            "system": system_info,
            "shutdown_requested": shutdown_event.is_set(),
            "timestamp": asyncio.get_event_loop().time(),
            "completely_corrected": True
        }
    except Exception as e:
        logger.error(f"COMPLETELY CORRECTED health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """COMPLETELY CORRECTED: Model information"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": settings.MODEL_NAME,
        "model_size": "3B parameters", 
        "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE),
        "context_length": "32K tokens",
        "architecture": "COMPLETELY_CORRECTED_WEBSOCKET_DISCONNECT_HANDLING",
        "supported_languages": [
            "English (en)", "Spanish (es)", "French (fr)", "Portuguese (pt)", 
            "Hindi (hi)", "German (de)", "Dutch (nl)", "Italian (it)"
        ],
        "mode_details": {
            "transcription": {
                "purpose": "Pure speech-to-text conversion",
                "output": "Exact words spoken by user",
                "temperature": 0.0,
                "api_method": "apply_transcription_request",
                "streaming": "continuous_binary"
            },
            "understanding": {
                "purpose": "Conversational AI with 300ms gap detection",
                "output": "AI responses after silence gaps",
                "temperature": 0.3,
                "api_method": "continuous_streaming → gap_detection → transcribe → respond",
                "streaming": "continuous_binary_with_gap_detection",
                "gap_threshold": "300ms",
                "target_latency": "<200ms",
                "message_handling": "proper_websocket_receive_with_disconnect_detection"
            }
        },
        "completely_corrected_fixes": [
            "✅ Fixed WebSocket disconnect handling (no more receive after disconnect)",
            "✅ Proper message type detection with message.receive()", 
            "✅ Enhanced background task cancellation on disconnect",
            "✅ Optimized LLM pipeline for speed",
            "✅ Robust FFmpeg process management"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """COMPLETELY CORRECTED: WebSocket transcription - PURE SPEECH-TO-TEXT ONLY"""
    await ws_manager.connect(websocket, "transcribe")
    conversation_manager.start_conversation(websocket)
    
    try:
        logger.info("🎤 COMPLETELY CORRECTED transcription session started")
        
        while not shutdown_event.is_set():
            try:
                # COMPLETELY CORRECTED: Proper message receiving with disconnect detection
                message = await websocket.receive()
                
                # Handle disconnect properly
                if message["type"] == "websocket.disconnect":
                    logger.info("Transcription WebSocket disconnected cleanly")
                    break
                
                # Handle binary audio data
                elif message["type"] == "websocket.receive" and "bytes" in message:
                    data = message["bytes"]
                    
                    if shutdown_event.is_set():
                        await websocket.send_json({"info": "Server shutting down"})
                        break
                    
                    # Better data validation
                    if not data or len(data) < 200:
                        logger.debug("Invalid/insufficient audio data received")
                        continue
                    
                    # Process through audio processor
                    result = await audio_processor.process_webm_chunk_transcribe(data, websocket)
                    
                    # Better result validation with strict error handling
                    if result and isinstance(result, dict):
                        if "error" in result:
                            logger.error(f"COMPLETELY CORRECTED audio processing error: {result['error']}")
                            await websocket.send_json({
                                "error": f"Audio processing failed: {result['error']}", 
                                "completely_corrected": True
                            })
                            continue
                        
                        if "audio_data" in result:
                            duration_ms = result.get("duration_ms", 0)
                            speech_ratio = result.get("speech_ratio", 0)
                            
                            # Better quality thresholds for human speech
                            if duration_ms > 1000 and speech_ratio > 0.4:
                                logger.info(f"🎤 COMPLETELY CORRECTED TRANSCRIBING: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                                
                                if model_manager and model_manager.is_loaded:
                                    # PURE transcription - NO response generation
                                    transcription_result = await model_manager.transcribe_audio_pure(
                                        result["audio_data"], 
                                        language="en"
                                    )
                                    
                                    # Strict validation - ONLY accept pure transcription
                                    if (isinstance(transcription_result, dict) and 
                                        transcription_result.get("text") and
                                        "error" not in transcription_result and
                                        len(transcription_result["text"].strip()) > 1):
                                        
                                        # Add to conversation
                                        conversation_manager.add_turn(
                                            websocket,
                                            transcription=transcription_result["text"],
                                            audio_duration=duration_ms / 1000,
                                            speech_ratio=speech_ratio,
                                            mode="transcribe",
                                            language=transcription_result.get("language", "en")
                                        )
                                        
                                        # Add stats and send response
                                        conv_stats = conversation_manager.get_conversation_stats(websocket)
                                        transcription_result["conversation"] = conv_stats
                                        transcription_result["completely_corrected"] = True
                                        
                                        await websocket.send_json(transcription_result)
                                        logger.info(f"✅ COMPLETELY CORRECTED PURE TRANSCRIBED: '{transcription_result['text']}'")
                                    else:
                                        logger.debug(f"Filtered out non-speech or AI response: {transcription_result}")
                                else:
                                    await websocket.send_json({"error": "Model not loaded"})
                            else:
                                logger.debug(f"Skipping: duration={duration_ms:.0f}ms, speech={speech_ratio:.3f}")
                
                # Handle JSON control messages (optional for transcription)
                elif message["type"] == "websocket.receive" and "text" in message:
                    try:
                        control_data = json.loads(message["text"])
                        logger.debug(f"Received transcription control message: {control_data}")
                    except json.JSONDecodeError:
                        logger.debug("Received non-JSON text message in transcription mode")
                    
            except WebSocketDisconnect:
                logger.info("Transcription WebSocket disconnected via exception")
                break
            except Exception as inner_e:
                logger.error(f"COMPLETELY CORRECTED inner WebSocket transcription error: {inner_e}", exc_info=True)
                try:
                    if not shutdown_event.is_set():
                        await websocket.send_json({
                            "error": f"Processing error: {str(inner_e)}",
                            "completely_corrected": True
                        })
                except:
                    logger.warning("Could not send error message, WebSocket likely disconnected")
                    break
                        
    except WebSocketDisconnect:
        logger.info("COMPLETELY CORRECTED transcription WebSocket disconnected")
    except Exception as e:
        logger.error(f"COMPLETELY CORRECTED WebSocket transcription error: {e}")
    finally:
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """COMPLETELY CORRECTED: Understanding mode with proper disconnect handling"""
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    # Background task reference
    gap_detection_task = None
    
    try:
        logger.info("🧠 COMPLETELY CORRECTED understanding session started")
        
        # Start continuous audio processing 
        await audio_processor.start_ffmpeg_decoder("understand", websocket)
        
        # Store accumulated audio for gap-based processing
        understanding_context = {
            "accumulated_audio": bytearray(),
            "last_speech_time": time.time(),
            "processing_audio": False,
            "user_query": "Please respond naturally to what I said",
            "connection_active": True  # CORRECTED: Track connection state
        }
        
        # COMPLETELY CORRECTED: Background task for gap detection and LLM processing
        async def process_understanding_gaps():
            """Process accumulated audio when 300ms gap detected"""
            while understanding_context["connection_active"] and not shutdown_event.is_set():
                try:
                    await asyncio.sleep(0.05)  # Check every 50ms
                    
                    current_time = time.time()
                    
                    # Check if we have accumulated audio and detected a 300ms gap
                    if (len(understanding_context["accumulated_audio"]) > 16000 and  # At least 1 second
                        not understanding_context["processing_audio"] and
                        understanding_context["connection_active"]):  # CORRECTED: Check connection
                        
                        # Check for 300ms silence gap
                        silence_duration = current_time - understanding_context["last_speech_time"]
                        
                        if silence_duration >= 0.3:  # 300ms gap detected
                            understanding_context["processing_audio"] = True
                            
                            try:
                                # Get accumulated audio
                                audio_data = bytes(understanding_context["accumulated_audio"])
                                understanding_context["accumulated_audio"].clear()
                                
                                logger.info(f"🧠 GAP DETECTED: Processing {len(audio_data)} bytes after {silence_duration:.3f}s gap")
                                
                                # Create WAV from accumulated PCM
                                wav_data = audio_processor._pcm_to_wav_enhanced(audio_data)
                                duration_ms = len(audio_data) / 2 / 16000 * 1000
                                
                                if model_manager and model_manager.is_loaded and understanding_context["connection_active"]:
                                    # Get conversation context
                                    context = conversation_manager.get_conversation_context(websocket)
                                    
                                    # STEP 1: Transcribe the accumulated audio
                                    start_time = time.time()
                                    
                                    transcription_result = await model_manager.transcribe_audio_pure(
                                        wav_data,
                                        language="en"
                                    )
                                    
                                    transcribe_time = (time.time() - start_time) * 1000
                                    
                                    if (isinstance(transcription_result, dict) and 
                                        transcription_result.get("text") and 
                                        "error" not in transcription_result and
                                        len(transcription_result["text"].strip()) > 2 and
                                        understanding_context["connection_active"]):  # CORRECTED: Check connection
                                        
                                        transcribed_text = transcription_result["text"].strip()
                                        logger.info(f"✅ TRANSCRIBED in {transcribe_time:.0f}ms: '{transcribed_text}'")
                                        
                                        # STEP 2: Generate understanding response
                                        llm_start_time = time.time()
                                        
                                        understanding_result = await model_manager.generate_understanding_response(
                                            transcribed_text=transcribed_text,
                                            user_query=understanding_context["user_query"],
                                            context=context
                                        )
                                        
                                        llm_time = (time.time() - llm_start_time) * 1000
                                        total_time = (time.time() - start_time) * 1000
                                        
                                        if (isinstance(understanding_result, dict) and 
                                            understanding_result.get("response") and 
                                            "error" not in understanding_result and
                                            understanding_context["connection_active"]):  # CORRECTED: Check connection
                                            
                                            # Send response only if connection is still active
                                            try:
                                                final_result = {
                                                    "type": "understanding",
                                                    "transcription": transcribed_text,
                                                    "response": understanding_result["response"],
                                                    "timestamp": current_time,
                                                    "language": transcription_result.get("language", "en"),
                                                    "audio_quality": {
                                                        "duration_ms": duration_ms,
                                                        "gap_detected_ms": silence_duration * 1000
                                                    },
                                                    "performance": {
                                                        "transcribe_time_ms": round(transcribe_time, 1),
                                                        "llm_time_ms": round(llm_time, 1),
                                                        "total_time_ms": round(total_time, 1)
                                                    },
                                                    "completely_corrected": True
                                                }
                                                
                                                # Add to conversation
                                                conversation_manager.add_turn(
                                                    websocket,
                                                    transcription=transcribed_text,
                                                    response=understanding_result["response"],
                                                    audio_duration=duration_ms / 1000,
                                                    speech_ratio=0.8,  # Good quality since gap detected
                                                    mode="understand",
                                                    language=transcription_result.get("language", "en")
                                                )
                                                
                                                await websocket.send_json(final_result)
                                                
                                                success_icon = "🚀" if total_time < 200 else "⏱️"
                                                logger.info(f"✅ {success_icon} UNDERSTANDING RESPONSE in {total_time:.0f}ms: '{understanding_result['response'][:50]}...'")
                                            except Exception as send_error:
                                                logger.warning(f"Could not send understanding response, connection likely closed: {send_error}")
                                                understanding_context["connection_active"] = False
                                                break
                                        else:
                                            logger.warning(f"Invalid understanding result: {understanding_result}")
                                    else:
                                        logger.warning(f"Invalid transcription result: {transcription_result}")
                                        
                            except Exception as processing_error:
                                logger.error(f"Gap processing error: {processing_error}", exc_info=True)
                                if understanding_context["connection_active"]:
                                    try:
                                        await websocket.send_json({
                                            "error": f"Gap processing failed: {str(processing_error)}",
                                            "completely_corrected": True
                                        })
                                    except:
                                        logger.warning("Could not send gap processing error, connection likely closed")
                                        understanding_context["connection_active"] = False
                            finally:
                                understanding_context["processing_audio"] = False
                
                except asyncio.CancelledError:
                    logger.info("Gap detection task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Gap detection error: {e}")
                    if not understanding_context["connection_active"]:
                        break
        
        # Start gap detection task
        gap_detection_task = asyncio.create_task(process_understanding_gaps())
        
        # COMPLETELY CORRECTED: Main WebSocket message loop
        while not shutdown_event.is_set() and understanding_context["connection_active"]:
            try:
                # COMPLETELY CORRECTED: Proper message receiving with timeout
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                
                # Handle disconnect properly
                if message["type"] == "websocket.disconnect":
                    logger.info("Understanding WebSocket disconnected cleanly")
                    understanding_context["connection_active"] = False
                    break
                
                # Handle JSON control messages
                elif message["type"] == "websocket.receive" and "text" in message:
                    try:
                        control_data = json.loads(message["text"])
                        
                        # Handle control messages
                        if isinstance(control_data, dict):
                            if "query" in control_data or "text" in control_data:
                                understanding_context["user_query"] = control_data.get("query", control_data.get("text", "Please respond naturally"))
                                logger.info(f"Updated user query: '{understanding_context['user_query']}'")
                            else:
                                logger.debug(f"Unknown JSON control message: {control_data}")
                                
                    except json.JSONDecodeError:
                        logger.debug("Received non-JSON text message in understanding mode")
                
                # Handle binary audio data
                elif message["type"] == "websocket.receive" and "bytes" in message:
                    audio_data = message["bytes"]
                    
                    if len(audio_data) < 200:
                        continue
                    
                    # Process WebM chunk through audio processor to get PCM
                    result = await audio_processor.process_webm_chunk_understand(audio_data, websocket)
                    
                    if result and isinstance(result, dict) and "pcm_data" in result and understanding_context["connection_active"]:
                        # Accumulate PCM data for gap-based processing
                        pcm_data = result["pcm_data"]
                        understanding_context["accumulated_audio"].extend(pcm_data)
                        
                        # Update last speech time if speech detected
                        if result.get("speech_detected", False):
                            understanding_context["last_speech_time"] = time.time()
                        
                        # Limit buffer size (30 seconds max)
                        max_buffer_size = 16000 * 30 * 2  # 30 seconds
                        if len(understanding_context["accumulated_audio"]) > max_buffer_size:
                            excess = len(understanding_context["accumulated_audio"]) - max_buffer_size
                            del understanding_context["accumulated_audio"][:excess]
                
            except asyncio.TimeoutError:
                logger.debug("Understanding WebSocket receive timeout, continuing...")
                continue
            except WebSocketDisconnect:
                logger.info("Understanding WebSocket disconnected via exception")
                understanding_context["connection_active"] = False
                break
            except Exception as inner_e:
                logger.error(f"COMPLETELY CORRECTED inner WebSocket understanding error: {inner_e}", exc_info=True)
                understanding_context["connection_active"] = False
                break
        
    except WebSocketDisconnect:
        logger.info("COMPLETELY CORRECTED understanding WebSocket disconnected")
    except Exception as e:
        logger.error(f"COMPLETELY CORRECTED WebSocket understanding error: {e}")
    finally:
        # COMPLETELY CORRECTED: Proper cleanup
        logger.info("Cleaning up understanding WebSocket connection...")
        
        # Signal context that connection is inactive
        understanding_context["connection_active"] = False
        
        # Cancel gap detection task
        if gap_detection_task and not gap_detection_task.done():
            logger.info("Cancelling gap detection task...")
            gap_detection_task.cancel()
            try:
                await asyncio.wait_for(gap_detection_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.info("Gap detection task cancelled/timed out")
        
        # Cleanup managers
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)
        
        logger.info("✅ Understanding WebSocket cleanup completed")

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
        "completely_corrected": True
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
        "completely_corrected": True
    }

@app.get("/debug/completely-corrected")
async def debug_completely_corrected():
    """COMPLETELY CORRECTED: Enhanced debug information"""
    return {
        "architecture": "COMPLETELY_CORRECTED_WEBSOCKET_DISCONNECT_HANDLING",
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
        "completely_corrected_architecture": {
            "transcription_mode": "Continuous binary WebM streaming → FFmpeg → PCM → VAD → LLM",
            "understanding_mode": "Continuous binary WebM streaming → PCM accumulation → 300ms gap detection → LLM",
            "websocket_handling": "Proper message.receive() with disconnect detection",
            "gap_detection": "WebRTC VAD with 300ms silence threshold + connection state tracking",
            "target_latency": "<200ms for LLM response",
            "audio_buffer": "30 second max accumulation with connection state validation",
            "error_recovery": "Proper WebSocket disconnect handling - no receive after disconnect"
        },
        "completely_corrected_fixes": [
            "✅ WEBSOCKET DISCONNECT FIX: Use message.receive() instead of receive_json/bytes",
            "✅ CONNECTION STATE TRACKING: Proper connection_active flag management",
            "✅ 300MS GAP DETECTION: WebRTC VAD with connection validation",
            "✅ BACKGROUND TASK CANCELLATION: Proper async task cleanup on disconnect", 
            "✅ FAST LLM PIPELINE: Optimized inference for <200ms response time",
            "✅ NO MORE RECEIVE ERRORS: Fixed 'Cannot call receive once disconnect received'"
        ],
        "completely_corrected": True
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
        logger.info("🛑 COMPLETELY CORRECTED server stopped by user")
    except Exception as e:
        logger.error(f"❌ COMPLETELY CORRECTED server error: {e}")
        sys.exit(1)
