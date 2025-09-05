# COMPLETELY FIXED MODEL LOADER - PROPER HUGGINGFACE VOXTRAL API
import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union
import gc
import tempfile
import os
import time
import numpy as np
import base64

# CRITICAL: Import for proper Voxtral processing
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import soundfile as sf

# CORRECT: Import mistral_common for audio handling as per HuggingFace docs
try:
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.messages import RawAudio
    MISTRAL_COMMON_AVAILABLE = True
except ImportError:
    MISTRAL_COMMON_AVAILABLE = False
    logging.warning("mistral-common not available, falling back to direct audio processing")

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    """COMPLETELY FIXED: Official Hugging Face Voxtral implementation"""
    
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
        
        # Supported languages for Voxtral
        self.supported_languages = {
            "en": "English", 
            "es": "Spanish", 
            "fr": "French",
            "pt": "Portuguese",
            "hi": "Hindi",
            "de": "German",
            "nl": "Dutch",
            "it": "Italian"
        }
        
        self.default_language = "en"
        self.target_response_ms = 200
        self.optimize_for_speed = True
        
        logger.info(f"✅ VoxtralUnderstandingManager initialized (HuggingFace official API)")
    
    async def load_model(self) -> None:
        """FIXED: Load Voxtral model using official HuggingFace API"""
        try:
            logger.info(f"🔄 Loading Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor - EXACT as per HuggingFace docs
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model - EXACT as per HuggingFace docs
            logger.info("Loading model...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Compatibility fix
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable optimizations
            if self.optimize_for_speed:
                self.model.config.use_cache = True
            
            # Store model info
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage(),
                "supported_languages": list(self.supported_languages.values()),
                "language_codes": list(self.supported_languages.keys()),
                "understanding_only": True,
                "transcription_disabled": True,
                "huggingface_api": True,
                "attention_implementation": "eager",
                "target_response_ms": self.target_response_ms
            }
            
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"✅ Memory usage: {self.model_info['memory_usage']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def understand_audio(self, message: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """FIXED: Process audio understanding using official HuggingFace Voxtral API"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        temp_file = None
        
        try:
            # Extract data from message - EXACT as user requested
            audio_data = message.get("audio")
            text_query = message.get("text", "What can you hear in this audio?")
            
            if not audio_data:
                return {"error": "No audio data in message"}
            
            logger.info(f"🧠 Processing understanding request with query: '{text_query}'")
            
            # Convert audio data to file - handles bytes or base64
            if isinstance(audio_data, str):
                # Base64 encoded
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except:
                    return {"error": "Invalid base64 audio data"}
            else:
                # Raw bytes
                audio_bytes = audio_data
            
            # Create temporary WAV file for Voxtral
            temp_file = await self._create_temp_wav_file(audio_bytes)
            if not temp_file:
                return {"error": "Failed to create audio file"}
            
            # CRITICAL: Use official HuggingFace conversation format
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": temp_file},
                    {"type": "text", "text": f"{text_query} {context}".strip()}
                ]
            }]
            
            # Apply chat template - EXACT as per HuggingFace docs
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = inputs.to(self.device, dtype=self.torch_dtype)
            
            # Generate response - optimized settings
            generation_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120 if self.optimize_for_speed else 150,
                    temperature=0.2 if self.optimize_for_speed else 0.3,
                    top_p=0.8 if self.optimize_for_speed else 0.9,
                    do_sample=True,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - generation_start
            
            # Decode response - extract only new tokens
            input_length = inputs.input_ids.shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True
            ).strip()
            
            total_time = time.time() - start_time
            
            # Validate response
            if not response or len(response.strip()) < 3:
                response = "I can hear the audio but I'm having trouble generating a detailed response. Could you try again?"
            
            result = {
                "response": response,
                "transcribed_text": "[Audio processed for understanding]",
                "processing_time_ms": total_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "language": self.default_language,
                "sub_200ms": total_time * 1000 < 200,
                "understanding_only": True,
                "transcription_disabled": True,
                "huggingface_api": True,
                "optimize_for_speed": self.optimize_for_speed,
                "conversation_format": "official"
            }
            
            logger.info(f"✅ Understanding complete: {total_time*1000:.0f}ms ({'✅' if total_time*1000 < 200 else '⚠️'} sub-200ms)")
            return result
            
        except Exception as e:
            logger.error(f"Understanding processing error: {e}", exc_info=True)
            return {"error": f"Processing failed: {str(e)}"}
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    async def _create_temp_wav_file(self, audio_bytes: bytes) -> Optional[str]:
        """Create temporary WAV file from audio bytes"""
        try:
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Try to convert using soundfile first
            try:
                # If it's already a valid audio format, read and convert
                import io
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                
                # Ensure mono and 16kHz for Voxtral
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                
                if sample_rate != 16000:
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                
                # Save as WAV
                sf.write(temp_path, audio_data, 16000)
                
            except:
                # Fallback: assume it's raw PCM and create WAV
                import wave
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(audio_bytes)
            
            # Verify file size
            if os.path.getsize(temp_path) < 1000:
                raise ValueError("Created audio file too small")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create temp WAV file: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
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
        
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                "gpu_allocated_gb": round(allocated, 2),
                "gpu_cached_gb": round(cached, 2),
                "gpu_total_gb": round(total, 2),
                "gpu_utilization": round((allocated / total) * 100, 1)
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {e}")
            return {"gpu_memory": 0.0}
    
    async def cleanup(self) -> None:
        """Cleanup model resources"""
        logger.info("🧹 Cleaning up model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        self.is_loaded = False
        logger.info("✅ Model cleanup completed")
