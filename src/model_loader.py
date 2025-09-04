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
