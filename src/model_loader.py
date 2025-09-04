# PERFECT MODEL LOADER - UNIFIED APPROACH FOR BOTH MODES
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
    """PERFECT: Unified Voxtral model manager with robust audio processing"""
    
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
        
        logger.info(f"Initialized PerfectVoxtralModelManager for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model and processor with error handling"""
        try:
            logger.info(f"🔄 Loading Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor
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
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """TRANSCRIPTION MODE: Audio -> Text (ASR only) with robust error handling"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                return {"error": "Invalid or insufficient audio data"}
            
            # Create temporary WAV file from audio data
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                return {"error": "Failed to create valid audio file"}
            
            # Use transcription request for pure ASR
            inputs = self.processor.apply_transcription_request(
                audio=temp_path,
                language="en", 
                model_id=self.model_name,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            transcription = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            ).strip()
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Validate transcription
            if not transcription or len(transcription) < 2:
                return {"error": "No meaningful transcription generated"}
            
            logger.info(f"✅ Transcription: '{transcription[:100]}...'")
            
            return {
                "type": "transcription",
                "text": transcription,
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
    
    async def understand_audio(self, audio_data: bytes, query: str = None) -> Dict[str, Any]:
        """UNDERSTANDING MODE: Audio -> Intelligent Response (ASR + LLM) with robust handling"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                return {"error": "Invalid or insufficient audio data"}
            
            # Create temporary WAV file from audio data
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                return {"error": "Failed to create valid audio file"}
            
            # UNDERSTANDING MODE: Audio conversation for intelligent response
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": temp_path
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate intelligent response (ASR + LLM)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            ).strip()
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Validate response
            if not response or len(response) < 2:
                return {"error": "No meaningful response generated"}
            
            logger.info(f"✅ Understanding: '{response[:100]}...'")
            
            return {
                "type": "understanding",
                "response": response,
                "query": query or "Audio understanding",
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
    
    def _audio_bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to WAV file with robust error handling"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Check if it's already a WAV file
            if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                # It's already a WAV file
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"✅ Used existing WAV format: {len(audio_bytes)} bytes")
                return temp_path
            
            # Assume raw PCM data
            if len(audio_bytes) % 2 == 1:
                audio_bytes = audio_bytes[:-1]
            
            if len(audio_bytes) < 32:  # Too small
                raise ValueError("Audio data too small")
            
            # Convert to numpy array
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            except Exception as e:
                logger.error(f"Failed to convert audio bytes to array: {e}")
                raise
            
            # Create WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_array.tobytes())
            
            logger.info(f"✅ Created WAV file: {len(audio_bytes)} bytes -> {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create WAV file: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise RuntimeError(f"Audio file creation failed: {e}")
    
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
                "gpu_total_gb": round(total, 2)
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {e}")
            return {"gpu_memory": 0.0}
    
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
