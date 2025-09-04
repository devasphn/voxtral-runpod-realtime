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
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """Manages Voxtral Mini 3B model loading and inference with VAD"""
    
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
    
    def _bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """Convert bytes to temporary WAV file"""
        try:
            # Create temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)  # Close file descriptor
            
            # Handle raw audio bytes - assume they are PCM data from WebSocket
            if len(audio_bytes) % 2 == 1:
                audio_bytes = audio_bytes[:-1]  # Remove last byte if odd length
            
            # Convert to numpy array assuming 16-bit PCM
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Write as WAV file with proper format
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_array.tobytes())
            
            logger.info(f"✅ Created WAV file: {len(audio_bytes)} bytes -> {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create WAV file: {e}")
            raise RuntimeError(f"Audio processing failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio data to text using proper Voxtral transcription API"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Convert bytes to WAV file
            temp_path = self._bytes_to_wav_file(audio_data)
            
            # Create Mistral Audio object
            audio = Audio.from_file(temp_path, strict=False)
            raw_audio = RawAudio.from_audio(audio)
            
            # Create proper transcription request
            transcription_request = TranscriptionRequest(
                model=self.model_name,
                audio=raw_audio,
                language="en",
                temperature=0.0
            )
            
            # Convert to inputs for the model
            inputs = transcription_request.to_openai(exclude=("top_p", "seed"))
            
            # Apply processor for transcription
            model_inputs = self.processor.apply_transcription_request(
                audio=temp_path,
                language="en",
                model_id=self.model_name,
                return_tensors="pt"
            )
            
            # Move to device
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_length = model_inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            decoded_output = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            logger.info(f"✅ Transcription result: {decoded_output[:50]}...")
            
            return {
                "type": "transcription",
                "text": decoded_output.strip(),
                "language": "en",
                "confidence": 0.95,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Clean up temp file on error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def understand_audio(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio with understanding capabilities using correct format"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Extract audio and text query from message
            audio_data = message.get("audio")
            text_query = message.get("text", "What can you hear in this audio?")
            
            if not audio_data:
                return {"error": "No audio data provided"}
            
            # Handle base64 encoded audio
            if isinstance(audio_data, str):
                try:
                    audio_bytes = base64.b64decode(audio_data)
                except Exception as e:
                    return {"error": f"Failed to decode base64 audio: {e}"}
            else:
                audio_bytes = audio_data
            
            # Convert to WAV file
            temp_path = self._bytes_to_wav_file(audio_bytes)
            
            # Create conversation with audio and text using CORRECT Voxtral format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",  # FIXED: Use "input_audio" not "audio"
                            "input_audio": temp_path
                        },
                        {
                            "type": "text", 
                            "text": text_query
                        }
                    ]
                }
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                conversation, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            decoded_output = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            logger.info(f"✅ Understanding result: {decoded_output[:50]}...")
            
            return {
                "type": "understanding",
                "response": decoded_output.strip(),
                "query": text_query,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Understanding error: {e}")
            # Clean up temp file on error
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
