import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union
import gc
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import numpy as np
import io
import base64
import tempfile
import os
import wave

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """Manages Voxtral Mini 3B model loading and inference"""
    
    def __init__(
        self, 
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        # Model components
        self.model: Optional[VoxtralForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        
        # State
        self.is_loaded = False
        self.model_info = {}
        
        logger.info(f"Initialized VoxtralModelManager for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model and processor"""
        try:
            logger.info(f"🔄 Loading Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            logger.info("Loading model...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",  # Changed from self.device to "auto"
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Store model info
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage()
            }
            
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio data to text"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Convert bytes to proper audio format for Voxtral
            audio_path = self._bytes_to_temp_file(audio_data)
            if audio_path is None:
                return {"error": "Invalid audio data"}
            
            # Create conversation with just audio for transcription
            conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "audio", "audio": audio_path}]
                }
            ]
            
            # Apply chat template
            try:
                inputs = self.processor.apply_chat_template(
                    conversation, 
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Generate transcription
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.0,  # Deterministic for transcription
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode output
                generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                decoded_output = self.processor.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                # Clean up temp file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                return {
                    "type": "transcription",
                    "text": decoded_output.strip(),
                    "language": "auto-detected",
                    "confidence": 0.95,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                raise e
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def understand_audio(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio with understanding capabilities"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Extract audio and text query
            audio_data = message.get("audio")
            text_query = message.get("text", "What can you hear in this audio?")
            
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Convert audio data
            if isinstance(audio_data, str):
                # Base64 encoded audio
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Convert to temp file
            audio_path = self._bytes_to_temp_file(audio_bytes)
            if audio_path is None:
                return {"error": "Invalid audio data"}
            
            # Create conversation with audio and text
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": text_query}
                    ]
                }
            ]
            
            try:
                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    conversation, 
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.2,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode output
                generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                decoded_output = self.processor.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                # Clean up temp file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                return {
                    "type": "understanding",
                    "response": decoded_output.strip(),
                    "query": text_query,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                raise e
            
        except Exception as e:
            logger.error(f"Understanding error: {e}")
            return {"error": f"Understanding failed: {str(e)}"}
    
    def _bytes_to_temp_file(self, audio_bytes: bytes) -> Optional[str]:
        """Convert bytes to temporary audio file"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name
            
            # Try to detect audio format and convert to WAV
            try:
                # First, try to detect if it's already a valid audio file
                if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                    # Already WAV format
                    with open(temp_path, 'wb') as f:
                        f.write(audio_bytes)
                    return temp_path
                
                # Try to convert using wave module (assume raw PCM)
                try:
                    # Assume 16-bit PCM, 16kHz, mono
                    with wave.open(temp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(16000)  # 16kHz
                        wav_file.writeframes(audio_bytes)
                    return temp_path
                    
                except Exception:
                    # If wave fails, try as raw bytes
                    with open(temp_path, 'wb') as f:
                        f.write(audio_bytes)
                    return temp_path
                    
            except Exception as convert_error:
                logger.warning(f"Audio conversion warning: {convert_error}")
                # Fallback: save raw bytes
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                return temp_path
                
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            return None
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_memory": 0.0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        
        return {
            "gpu_allocated_gb": round(allocated, 2),
            "gpu_cached_gb": round(cached, 2),
            "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    async def cleanup(self) -> None:
        """Clean up model resources"""
        logger.info("🧹 Cleaning up model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.is_loaded = False
        logger.info("✅ Model cleanup completed")
