# COMPREHENSIVE FIXED MAIN.PY - ROBUST SHUTDOWN, CONVERSATION MEMORY & LONGER AUDIO SUPPORT
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import json
import base64
from datetime import datetime, timedelta
import uuid

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

# Global components
model_manager = None
ws_manager = WebSocketManager()
audio_processor = AudioProcessor(
    sample_rate=16000,
    channels=1,
    chunk_duration_ms=30
)

# CONVERSATION MEMORY SYSTEM
class ConversationMemory:
    """Manages conversation context and memory across interactions"""
    
    def __init__(self, max_history_length: int = 10, max_context_tokens: int = 2000):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.max_history_length = max_history_length
        self.max_context_tokens = max_context_tokens
        
    def add_interaction(self, session_id: str, user_input: str, ai_response: str, interaction_type: str = "understanding"):
        """Add a conversation interaction to memory"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "type": interaction_type
        }
        
        self.conversations[session_id].append(interaction)
        
        # Keep only recent interactions
        if len(self.conversations[session_id]) > self.max_history_length:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history_length:]
            
        logger.info(f"Added conversation interaction for session {session_id}")
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context for AI model"""
        if session_id not in self.conversations:
            return ""
            
        context_parts = []
        total_chars = 0
        
        # Build context from recent interactions
        for interaction in reversed(self.conversations[session_id]):
            interaction_text = f"User: {interaction['user_input']}\nAI: {interaction['ai_response']}\n"
            
            if total_chars + len(interaction_text) > self.max_context_tokens * 4:  # Rough token estimation
                break
                
            context_parts.insert(0, interaction_text)
            total_chars += len(interaction_text)
        
        if context_parts:
            context = "Previous conversation:\n" + "\n".join(context_parts) + "\nCurrent question: "
            logger.info(f"Built context for session {session_id}: {len(context_parts)} interactions, {total_chars} chars")
            return context
        
        return ""
    
    def update_user_profile(self, session_id: str, info: Dict[str, Any]):
        """Update user profile information"""
        if session_id not in self.user_profiles:
            self.user_profiles[session_id] = {}
            
        self.user_profiles[session_id].update(info)
        logger.info(f"Updated user profile for session {session_id}: {info}")
    
    def get_user_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        return self.user_profiles.get(session_id, {})
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old conversation sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        sessions_to_remove = []
        for session_id, conversations in self.conversations.items():
            if conversations:
                last_interaction_time = datetime.fromisoformat(conversations[-1]["timestamp"])
                if last_interaction_time < cutoff_time:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.conversations[session_id]
            if session_id in self.user_profiles:
                del self.user_profiles[session_id]
                
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old conversation sessions")

# Global conversation memory
conversation_memory = ConversationMemory()

# ENHANCED AUDIO BUFFER FOR LONGER CONVERSATIONS
class LongConversationBuffer:
    """Manages longer audio conversations with context"""
    
    def __init__(self, max_buffer_duration: int = 60):
        self.session_buffers: Dict[str, Dict[str, Any]] = {}
        self.max_buffer_duration = max_buffer_duration  # seconds
        
    def add_audio_chunk(self, session_id: str, audio_data: bytes, speech_ratio: float, mode: str):
        """Add audio chunk to session buffer"""
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = {
                "chunks": [],
                "total_duration": 0,
                "last_activity": datetime.now(),
                "mode": mode
            }
        
        chunk_duration = len(audio_data) / (16000 * 2)  # Rough duration estimate
        
        chunk_info = {
            "audio_data": audio_data,
            "speech_ratio": speech_ratio,
            "timestamp": datetime.now(),
            "duration": chunk_duration
        }
        
        self.session_buffers[session_id]["chunks"].append(chunk_info)
        self.session_buffers[session_id]["total_duration"] += chunk_duration
        self.session_buffers[session_id]["last_activity"] = datetime.now()
        
        # Remove old chunks if buffer too long
        while self.session_buffers[session_id]["total_duration"] > self.max_buffer_duration:
            old_chunk = self.session_buffers[session_id]["chunks"].pop(0)
            self.session_buffers[session_id]["total_duration"] -= old_chunk["duration"]
    
    def should_process_buffer(self, session_id: str) -> bool:
        """Determine if buffer should be processed"""
        if session_id not in self.session_buffers:
            return False
            
        buffer = self.session_buffers[session_id]
        
        # Process if we have enough audio with speech
        speech_chunks = [c for c in buffer["chunks"] if c["speech_ratio"] > 0.1]
        
        if buffer["mode"] == "transcribe":
            return len(speech_chunks) >= 2 and buffer["total_duration"] >= 1.0
        else:  # understand mode
            return len(speech_chunks) >= 3 and buffer["total_duration"] >= 2.0
    
    def get_combined_audio(self, session_id: str) -> Optional[bytes]:
        """Get combined audio data from buffer"""
        if session_id not in self.session_buffers:
            return None
            
        buffer = self.session_buffers[session_id]
        combined_audio = b""
        
        for chunk in buffer["chunks"]:
            combined_audio += chunk["audio_data"]
        
        # Clear processed chunks
        self.session_buffers[session_id]["chunks"] = []
        self.session_buffers[session_id]["total_duration"] = 0
        
        return combined_audio if combined_audio else None
    
    def cleanup_old_buffers(self, max_idle_minutes: int = 5):
        """Clean up idle audio buffers"""
        cutoff_time = datetime.now() - timedelta(minutes=max_idle_minutes)
        
        sessions_to_remove = []
        for session_id, buffer in self.session_buffers.items():
            if buffer["last_activity"] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_buffers[session_id]
            
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} idle audio buffers")

# Global audio buffer
long_conversation_buffer = LongConversationBuffer()

# GRACEFUL SHUTDOWN HANDLER
class GracefulShutdownHandler:
    """Handles graceful shutdown of all components"""
    
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.shutdown_tasks = []
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._shutdown())
    
    async def _shutdown(self):
        """Perform graceful shutdown"""
        logger.info("🛑 Starting graceful shutdown...")
        self.shutdown_event.set()
        
        # Close all WebSocket connections
        await ws_manager.disconnect_all()
        
        # Stop audio processor
        if audio_processor:
            await audio_processor.cleanup()
        
        # Cleanup model
        if model_manager:
            await model_manager.cleanup()
        
        # Cancel remaining tasks
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("✅ Graceful shutdown completed")

# Global shutdown handler
shutdown_handler = GracefulShutdownHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with robust error handling"""
    global model_manager
    
    # Startup
    logger.info("🚀 Starting ENHANCED Voxtral Real-Time Server...")
    
    # Setup signal handlers
    shutdown_handler.setup_signal_handlers()
    
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
    
    # Start cleanup tasks
    asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    await shutdown_handler._shutdown()

async def periodic_cleanup():
    """Periodic cleanup of memory and buffers"""
    while not shutdown_handler.shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            conversation_memory.cleanup_old_sessions()
            long_conversation_buffer.cleanup_old_buffers()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Create FastAPI app
app = FastAPI(
    title="Voxtral Mini 3B - ENHANCED Real-Time API",
    description="Enhanced: Conversation Memory + Graceful Shutdown + Longer Audio Support",
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
            "conversation_sessions": len(conversation_memory.conversations),
            "audio_buffers": len(long_conversation_buffer.session_buffers),
            "features": [
                "✅ Conversation Memory & Context",
                "✅ Longer Audio Support (up to 60s)",
                "✅ Graceful Shutdown Handling", 
                "✅ Robust WebSocket Management",
                "✅ User Profile Memory",
                "✅ Automatic Session Cleanup"
            ],
            "system": system_info,
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
        "conversation_memory": "✅ Context-aware responses",
        "longer_audio": "✅ Up to 60 seconds continuous speech",
        "user_profiles": "✅ Remembers user information",
        "supported_languages": [
            "English", "Spanish", "French", "Portuguese", 
            "Hindi", "German", "Dutch", "Italian"
        ],
        "enhanced_capabilities": [
            "✅ Multi-turn conversations with memory",
            "✅ Longer audio processing (30s-1min)",
            "✅ User context and profile memory",
            "✅ Robust error handling & recovery",
            "✅ Graceful shutdown handling",
            "✅ Session-based conversation tracking"
        ]
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """ENHANCED TRANSCRIPTION MODE: With longer audio support"""
    session_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, "transcribe", session_id)
    
    try:
        while not shutdown_handler.shutdown_event.is_set():
            try:
                # Receive WebM audio from browser
                data = await websocket.receive_bytes()
                
                if not data or len(data) < 10:
                    logger.warning("Received invalid/empty audio data")
                    continue
                
                # Process WebM chunk through enhanced audio processor
                result = await audio_processor.process_webm_chunk_transcribe(data)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # Add to long conversation buffer
                    long_conversation_buffer.add_audio_chunk(
                        session_id, result["audio_data"], speech_ratio, "transcribe"
                    )
                    
                    # Check if we should process the accumulated buffer
                    if long_conversation_buffer.should_process_buffer(session_id):
                        combined_audio = long_conversation_buffer.get_combined_audio(session_id)
                        
                        if combined_audio and model_manager and model_manager.is_loaded:
                            logger.info(f"🎤 TRANSCRIBING accumulated audio buffer for session {session_id}")
                            
                            # Get conversation context
                            context = conversation_memory.get_context(session_id)
                            
                            # Transcribe with context
                            transcription_result = await model_manager.transcribe_audio_with_context(
                                combined_audio, context
                            )
                            
                            if (transcription_result.get("text") and 
                                "error" not in transcription_result and
                                len(transcription_result["text"].strip()) > 0):
                                
                                # Extract user information from transcription
                                await extract_user_info(session_id, transcription_result["text"])
                                
                                # Add to conversation memory
                                conversation_memory.add_interaction(
                                    session_id, 
                                    transcription_result["text"], 
                                    "transcription_only",
                                    "transcription"
                                )
                                
                                await websocket.send_json({
                                    **transcription_result,
                                    "session_id": session_id,
                                    "has_context": bool(context)
                                })
                                logger.info(f"✅ TRANSCRIBED: '{transcription_result['text']}'")
                                
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                logger.info("Transcription WebSocket disconnected")
                break
            except Exception as inner_e:
                logger.error(f"Inner WebSocket transcription error: {inner_e}")
                try:
                    await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break
                        
    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}")
    finally:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    """ENHANCED UNDERSTANDING MODE: With conversation memory and longer audio support"""
    session_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, "understand", session_id)
    
    try:
        while not shutdown_handler.shutdown_event.is_set():
            try:
                # Receive JSON message with audio
                message = await websocket.receive_json()
                
                if not isinstance(message, dict) or "audio" not in message:
                    await websocket.send_json({"error": "Invalid message format"})
                    continue
                
                audio_data = message.get("audio")
                user_query = message.get("text", "What can you hear in this audio?")
                
                if not audio_data:
                    await websocket.send_json({"error": "No audio data provided"})
                    continue
                
                # Handle base64 encoded audio
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
                
                # Process through enhanced audio processor
                result = await audio_processor.process_webm_chunk_understand(audio_bytes)
                
                if result and "audio_data" in result:
                    duration_ms = result.get("duration_ms", 0)
                    speech_ratio = result.get("speech_ratio", 0)
                    
                    # Add to long conversation buffer
                    long_conversation_buffer.add_audio_chunk(
                        session_id, result["audio_data"], speech_ratio, "understand"
                    )
                    
                    # Check if we should process the accumulated buffer
                    if long_conversation_buffer.should_process_buffer(session_id):
                        combined_audio = long_conversation_buffer.get_combined_audio(session_id)
                        
                        if combined_audio and model_manager and model_manager.is_loaded:
                            logger.info(f"🎤 UNDERSTANDING accumulated audio buffer for session {session_id}")
                            
                            # Get conversation context and user profile
                            context = conversation_memory.get_context(session_id)
                            user_profile = conversation_memory.get_user_profile(session_id)
                            
                            # Create enhanced query with context
                            enhanced_query = build_contextual_query(user_query, context, user_profile)
                            
                            # Process with context
                            understanding_result = await model_manager.understand_audio_with_context(
                                combined_audio, enhanced_query, context
                            )
                            
                            if ("response" in understanding_result and 
                                "error" not in understanding_result and
                                len(understanding_result["response"].strip()) > 0):
                                
                                # Extract user information from the interaction
                                await extract_user_info_from_interaction(
                                    session_id, user_query, understanding_result["response"]
                                )
                                
                                # Add to conversation memory
                                conversation_memory.add_interaction(
                                    session_id,
                                    user_query,
                                    understanding_result["response"],
                                    "understanding"
                                )
                                
                                await websocket.send_json({
                                    **understanding_result,
                                    "session_id": session_id,
                                    "has_context": bool(context),
                                    "user_profile": user_profile
                                })
                                logger.info(f"✅ UNDERSTOOD: '{understanding_result['response']}'")
                                
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                logger.info("Understanding WebSocket disconnected")
                break
            except json.JSONDecodeError as json_e:
                logger.error(f"JSON decode error: {json_e}")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as inner_e:
                logger.error(f"Inner WebSocket understanding error: {inner_e}")
                try:
                    await websocket.send_json({"error": f"Processing error: {str(inner_e)}"})
                except:
                    break
                
    except Exception as e:
        logger.error(f"WebSocket understanding error: {e}")
    finally:
        ws_manager.disconnect(websocket)

async def extract_user_info(session_id: str, text: str):
    """Extract user information from transcribed text"""
    text_lower = text.lower()
    
    # Extract name
    if "my name is" in text_lower:
        try:
            name_part = text.split("my name is", 1)[1].strip()
            name = name_part.split()[0] if name_part else None
            if name:
                conversation_memory.update_user_profile(session_id, {"name": name})
        except:
            pass
    
    elif "i am" in text_lower and len(text.split()) <= 5:
        try:
            name_part = text.split("i am", 1)[1].strip()
            name = name_part.split()[0] if name_part else None
            if name and name.isalpha():
                conversation_memory.update_user_profile(session_id, {"name": name})
        except:
            pass

async def extract_user_info_from_interaction(session_id: str, user_input: str, ai_response: str):
    """Extract user information from conversation interaction"""
    await extract_user_info(session_id, user_input)

def build_contextual_query(original_query: str, context: str, user_profile: Dict[str, Any]) -> str:
    """Build enhanced query with context and user profile"""
    query_parts = []
    
    if user_profile.get("name"):
        query_parts.append(f"The user's name is {user_profile['name']}.")
    
    if context:
        query_parts.append(context)
    
    query_parts.append(f"User question: {original_query}")
    
    return " ".join(query_parts)

# Additional enhanced endpoints
@app.get("/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "conversation_history": conversation_memory.conversations.get(session_id, []),
        "user_profile": conversation_memory.get_user_profile(session_id)
    }

@app.post("/conversations/{session_id}/clear")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_memory.conversations:
        del conversation_memory.conversations[session_id]
    if session_id in conversation_memory.user_profiles:
        del conversation_memory.user_profiles[session_id]
    
    return {"status": "Conversation cleared", "session_id": session_id}

@app.get("/debug/memory")
async def debug_memory():
    """Debug conversation memory"""
    return {
        "active_conversations": len(conversation_memory.conversations),
        "active_user_profiles": len(conversation_memory.user_profiles),
        "audio_buffers": len(long_conversation_buffer.session_buffers),
        "memory_stats": {
            "max_history_length": conversation_memory.max_history_length,
            "max_context_tokens": conversation_memory.max_context_tokens
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
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server shutdown complete")
