# FINAL PERFECTED SOLUTION - main.py - VAD-FIXED & ROBUST STATE MANAGEMENT
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
import json
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from config.settings import Settings
from config.logging_config import setup_logging
from src.websocket_handler import WebSocketManager
from src.conversation_manager import ConversationManager
from src.model_loader import VoxtralModelManager
from src.audio_processor import PerfectAudioProcessor

# --- Initialization ---
setup_logging()
logger = logging.getLogger(__name__)

settings = Settings()
model_manager = None
ws_manager = WebSocketManager()
conversation_manager = ConversationManager()
audio_processor = PerfectAudioProcessor()

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    logger.info("🚀 Starting FINAL PERFECTED Voxtral Real-Time Server...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        model_manager = VoxtralModelManager()
        await model_manager.load_model()
    except Exception as e:
        logger.critical(f"❌ CRITICAL: Failed to load model on startup: {e}", exc_info=True)
    yield
    logger.info("🛑 Shutting down FINAL PERFECTED server...")
    if model_manager: await model_manager.cleanup()
    logger.info("✅ FINAL PERFECTED graceful shutdown completed")

app = FastAPI(title="Voxtral Mini 3B - FINAL PERFECTED API", version="4.0.0-STABLE", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HTTP Endpoints ---
@app.get("/")
async def root():
    with open("static/index.html", "r") as f: return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_manager and model_manager.is_loaded else "error_not_loaded"
    status_code = 200 if model_status == "loaded" else 503
    return JSONResponse(status_code=status_code, content={"status": "healthy" if model_status == "loaded" else "unhealthy", "model_status": model_status})

@app.get("/model/info")
async def model_info():
    """THE FIX: Re-added the missing /model/info endpoint to prevent 404 errors from the frontend."""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded or in error state.")
    return {
        "model_name": settings.MODEL_NAME, "device": str(settings.DEVICE),
        "dtype": str(settings.TORCH_DTYPE), "context_length": "32K tokens",
        "supported_languages": list(model_manager.supported_languages),
        "architecture": "Voxtral-Mini-3B with Flash Attention 2"
    }

# --- WebSocket Endpoint: Understanding (with Gap Detection) ---
@app.websocket("/ws/understand")
async def websocket_understand(websocket: WebSocket):
    await ws_manager.connect(websocket, "understand")
    conversation_manager.start_conversation(websocket)
    
    conn_context = {
        "pcm_buffer": bytearray(), "last_speech_time": time.time(),
        "processing": False, "user_query": "Please respond naturally.",
        "min_buffer_size": int(16000 * 2 * 0.25) # Min 0.25s of audio
    }

    async def process_audio_chunk(audio_chunk: bytes, reason: str):
        if not audio_chunk: return
        logger.info(f"🧠 Processing {len(audio_chunk)} bytes of audio. Reason: {reason}.")
        
        trans_result = await model_manager.transcribe_audio_pure(audio_chunk)
        if "error" in trans_result or not trans_result.get("text", "").strip():
            logger.warning(f"Skipping empty/failed transcription: {trans_result}")
            return

        text = trans_result["text"]
        logger.info(f"🎤 Transcribed: '{text}'")
        
        context = conversation_manager.get_conversation_context(websocket)
        response = await model_manager.generate_understanding_response(
            transcribed_text=text, user_query=conn_context["user_query"], context=context
        )

        if "error" not in response:
            conversation_manager.add_turn(
                websocket, transcription=text, response=response.get("response", ""), mode="understand"
            )
            try:
                await websocket.send_json(response)
            except WebSocketDisconnect:
                logger.warning("Client disconnected during message send.")

    async def gap_detection_task():
        while not shutdown_event.is_set() and websocket.client_state.name == 'CONNECTED':
            await asyncio.sleep(0.1)
            
            silence_ms = (time.time() - conn_context["last_speech_time"]) * 1000
            
            if not conn_context["processing"] and len(conn_context["pcm_buffer"]) > conn_context["min_buffer_size"] and silence_ms > 300:
                conn_context["processing"] = True
                audio_to_process = conn_context["pcm_buffer"]
                conn_context["pcm_buffer"] = bytearray()
                try:
                    await process_audio_chunk(bytes(audio_to_process), reason=f"Gap ({silence_ms:.0f}ms)")
                finally:
                    conn_context["processing"] = False

    gap_task = asyncio.create_task(gap_detection_task())

    try:
        while not shutdown_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect": break
            
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if "query" in data: conn_context["user_query"] = data["query"]
                except json.JSONDecodeError: pass
            elif "bytes" in message:
                result = await audio_processor.process_webm_chunk_understand(message["bytes"])
                if result and "pcm_data" in result:
                    conn_context["pcm_buffer"].extend(result["pcm_data"])
                    if result["speech_detected"]:
                        conn_context["last_speech_time"] = time.time()
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    finally:
        gap_task.cancel()
        if not conn_context["processing"] and len(conn_context["pcm_buffer"]) > conn_context["min_buffer_size"]:
             logger.info("Client disconnected, performing final flush of audio buffer.")
             await process_audio_chunk(bytes(conn_context["pcm_buffer"]), reason="Final flush")
        conversation_manager.cleanup_conversation(websocket)
        ws_manager.disconnect(websocket)```

#### **4. `src/model_loader.py` (Final Perfected Version)**
This version uses the official `flash_attention_2` implementation for maximum performance, as documented by Hugging Face, and includes more robust error logging.

```python
# FINAL PERFECTED SOLUTION - model_loader.py - FLASH ATTENTION & ROBUST ERROR HANDLING
import asyncio
import logging
import torch
from typing import Optional, Dict, Any
import tempfile
import os
import wave

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """FINAL PERFECTED: Voxtral model manager with Flash Attention 2, correct API usage, and hardened error handling."""
    
    def __init__(
        self, 
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.model: Optional[VoxtralForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.is_loaded = False
        self.supported_languages = {"en", "es", "fr", "pt", "hi", "de", "nl", "it"}
        logger.info(f"✅ FINAL PERFECTED VoxtralModelManager initialized for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        try:
            logger.info(f"🔄 Loading FINAL PERFECTED Voxtral model: {self.model_name}")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # THE FIX: Explicitly use flash_attention_2 for max performance on compatible GPUs
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype, device_map="auto",
                trust_remote_code=True, low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2"
            )
            self.model.eval()
            self.is_loaded = True
            mem_usage = self._get_memory_usage()
            logger.info(f"✅ FINAL PERFECTED Model loaded! Memory usage: {mem_usage.get('gpu_memory_gb', 0):.2f} GB")
        except Exception as e:
            logger.critical(f"❌ FAILED to load model: {e}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def transcribe_audio_pure(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        if not self.is_loaded: return {"error": "Model not loaded"}
        if not audio_data or len(audio_data) < 1000: return {"text": ""}
        
        temp_path = self._create_wav_file(audio_data)
        if not temp_path: return {"error": "Failed to create temporary audio file"}
        
        try:
            inputs = self.processor.apply_transcription_request(
                language=language if language in self.supported_languages else "en",
                audio=temp_path, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.0, do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id, use_cache=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            transcription = self.processor.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
            return {"type": "transcription", "text": transcription, "perfect": True}
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return {"error": f"Transcription failed: {str(e)}"}
        finally:
            if os.path.exists(temp_path): os.unlink(temp_path)
    
    async def generate_understanding_response(self, transcribed_text: str, user_query: str, context: str = "") -> Dict[str, Any]:
        if not self.is_loaded: return {"error": "Model not loaded"}
        if not transcribed_text.strip(): return {"response": "I didn't hear anything clearly. Could you please repeat that?"}

        try:
            system_message = f"You are a helpful AI assistant. The user's speech was: '{transcribed_text}'. The user's instruction is: '{user_query}'. "
            if context: system_message += f"Previous context: {context}"
            
            conversation = [{"role": "system", "content": system_message}]
            inputs = self.processor.apply_chat_template(conversation, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=256, temperature=0.2, top_p=0.95, do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id, use_cache=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            response = self.processor.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()
            return {"type": "understanding", "response": response, "perfect": True}
        except Exception as e:
            logger.error(f"Understanding generation error: {e}", exc_info=True)
            return {"error": f"Understanding failed: {str(e)}"}
    
    def _create_wav_file(self, audio_data: bytes) -> Optional[str]:
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(audio_data)
            return temp_path
        except Exception as e:
            logger.error(f"WAV file creation failed: {e}", exc_info=True)
            return None

    def _get_memory_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available(): return {"gpu_memory_gb": 0.0}
        return {"gpu_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)}

    async def cleanup(self) -> None:
        logger.info("🧹 Cleaning up FINAL PERFECTED model resources...")
        del self.model
        del self.processor
        self.model, self.processor = None, None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("✅ FINAL PERFECTED model cleanup completed.")
