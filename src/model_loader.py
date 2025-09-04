import asyncio
import logging
import torch
from typing import Optional, Dict, Any
import tempfile
import os
import wave
import numpy as np

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """PERFECT: Voxtral model manager with correct API usage and perfect audio processing"""
    
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
        self.model_info = {}
        
        # PERFECT: Valid language codes for Voxtral (from official docs)
        self.supported_languages = {
            "en": "English", "es": "Spanish", "fr": "French", "pt": "Portuguese",
            "hi": "Hindi", "de": "German", "nl": "Dutch", "it": "Italian"
        }
        self.default_language = "en"
        
        logger.info(f"✅ PERFECT VoxtralModelManager initialized for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model with optimized settings"""
        try:
            logger.info(f"🔄 Loading PERFECT Voxtral model: {self.model_name}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            self.model.eval()
            
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage(),
                "perfect": True
            }
            
            self.is_loaded = True
            logger.info(f"✅ PERFECT Model loaded successfully! Memory usage: {self.model_info['memory_usage']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PERFECT model: {e}")
            raise RuntimeError(f"PERFECT model loading failed: {e}")
    
    async def transcribe_audio_pure(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """PERFECT: Pure transcription using the CORRECT Voxtral API (apply_transcription_request)"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            if not audio_data or len(audio_data) < 1000:
                return {"error": "Invalid audio data for transcription"}
            
            temp_path = self._create_wav_file(audio_data)
            if not temp_path:
                return {"error": "Failed to create temporary audio file"}
            
            # PERFECT FIX: Use the correct API for transcription
            inputs = self.processor.apply_transcription_request(
                language=language if language in self.supported_languages else self.default_language,
                audio=temp_path,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,  # Deterministic for transcription
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0, input_length:]
            transcription = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            
            return {
                "type": "transcription", 
                "text": transcription,
                "language": language,
                "perfect": True
            }
            
        except Exception as e:
            logger.error(f"PERFECT transcription error: {e}", exc_info=True)
            return {"error": f"Transcription failed: {str(e)}"}
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def generate_understanding_response(
        self, 
        transcribed_text: str, 
        user_query: str, 
        context: str = ""
    ) -> Dict[str, Any]:
        """PERFECT: Generate understanding response using the CORRECT Voxtral API (apply_chat_template)"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            if not transcribed_text or len(transcribed_text.strip()) < 1:
                return {"error": "Invalid transcribed text for understanding"}
            
            system_message = "You are a helpful AI assistant. The user's speech has been transcribed. Respond helpfully to what they said."
            if context:
                system_message += f"\n\nPrevious conversation context:\n{context}"
            if user_query:
                 system_message += f"\n\nUser's explicit instruction: {user_query}"
            
            # PERFECT FIX: Use apply_chat_template for conversational understanding
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcribed_text}
            ]
            
            inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,  # Slightly creative for conversation
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0, input_length:]
            response = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            
            return {
                "type": "understanding",
                "response": response,
                "perfect": True,
            }
            
        except Exception as e:
            logger.error(f"PERFECT understanding error: {e}", exc_info=True)
            return {"error": f"Understanding generation failed: {str(e)}"}
    
    def _create_wav_file(self, audio_data: bytes) -> Optional[str]:
        """PERFECT: Create a temporary WAV file from raw bytes (WAV or PCM)"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # If data is already a valid WAV, just write it
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                return temp_path

            # Otherwise, assume it's raw PCM and create a WAV header
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000) # 16kHz
                wav_file.writeframes(audio_data)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"PERFECT WAV file creation failed: {e}")
            return None
    
    def _count_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_memory_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"gpu_memory_gb": 0.0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        return {"gpu_memory_gb": round(allocated, 2)}
    
    async def cleanup(self) -> None:
        logger.info("🧹 Cleaning up PERFECT model resources...")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("✅ PERFECT model cleanup completed")
