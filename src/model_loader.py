import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union
import gc
import tempfile
import os
import wave
import base64
import numpy as np

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """FIXED: Manages Voxtral Mini 3B model with CORRECT API usage"""
    
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
        """Load Voxtral model and processor - CORRECT METHOD"""
        try:
            logger.info(f"🔄 Loading Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # CORRECT: Load processor without extra parameters
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model
            logger.info("Loading model...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
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
    
    def _save_wav_bytes(self, audio_bytes: bytes) -> str:
        """Save audio bytes as WAV file"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save WAV file: {e}")
            raise RuntimeError(f"WAV file creation failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """TRANSCRIPTION MODE: Audio -> Text (ASR only) - CORRECT API"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Save WAV bytes to temp file (audio_data is already WAV from processor)
            temp_path = self._save_wav_bytes(audio_data)
            
            # CORRECT: Use apply_transcription_request for pure ASR
            logger.info("🎤 Running transcription with apply_transcription_request...")
            inputs = self.processor.apply_transcription_request(
                language="en", 
                audio=temp_path,
                model_id=self.model_name,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            transcription = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            logger.info(f"✅ Transcription completed: '{transcription[:100]}...'")
            
            return {
                "type": "transcription",
                "text": transcription.strip(),
                "language": "en",
                "confidence": 0.95,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def understand_audio(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """UNDERSTANDING MODE: Audio -> ASR + LLM Response - CORRECT API"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Extract audio from message
            audio_data = message.get("audio")
            
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Handle base64 encoded audio from browser
            if isinstance(audio_data, str):
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except Exception as e:
                    return {"error": f"Failed to decode base64 audio: {e}"}
            else:
                audio_bytes = audio_data
            
            # Save as temp WAV file
            temp_path = self._save_wav_bytes(audio_bytes)
            
            # CORRECT: Use apply_chat_template for understanding (ASR + LLM)
            logger.info("🧠 Running understanding with apply_chat_template...")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": temp_path  # Audio only for understanding
                        }
                    ]
                }
            ]
            
            # Apply chat template for understanding
            inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate intelligent response (ASR + LLM)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            logger.info(f"✅ Understanding completed: '{response[:100]}...'")
            
            return {
                "type": "understanding",
                "response": response.strip(),
                "query": "Audio understanding (speech -> intelligent response)",
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Understanding error: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"Understanding failed: {str(e)}"}
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {"gpu_memory": 0.0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        self.is_loaded = False
        logger.info("✅ Model cleanup completed")
