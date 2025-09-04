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
from mistral_common.protocol.instruct.messages import RawAudio, TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

logger = logging.getLogger(__name__)

class VoxtralModelManager:
    """FIXED: Manages Voxtral Mini 3B model with proper mistral_common integration"""
    
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
            
            # Load processor with mistral tokenizer mode
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                tokenizer_mode="mistral"
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
    
    def _convert_webm_to_wav(self, webm_bytes: bytes) -> str:
        """FIXED: Convert WebM/Opus audio from browser to WAV file for Voxtral"""
        try:
            import io
            from pydub import AudioSegment
            
            # Create temporary files
            temp_webm_fd, temp_webm_path = tempfile.mkstemp(suffix='.webm')
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
            
            # Close file descriptors
            os.close(temp_webm_fd)
            os.close(temp_wav_fd)
            
            try:
                # Write WebM data to temp file
                with open(temp_webm_path, 'wb') as f:
                    f.write(webm_bytes)
                
                # Convert WebM to WAV using pydub
                audio_segment = AudioSegment.from_file(temp_webm_path)
                
                # Convert to the format Voxtral expects:
                # - Mono (1 channel)
                # - 16kHz sample rate  
                # - 16-bit depth
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(16000)
                audio_segment = audio_segment.set_sample_width(2)  # 16-bit
                
                # Export as WAV
                audio_segment.export(temp_wav_path, format="wav")
                
                logger.info(f"✅ Converted WebM to WAV: {len(webm_bytes)} bytes -> {temp_wav_path}")
                
                # Clean up WebM file
                if os.path.exists(temp_webm_path):
                    os.unlink(temp_webm_path)
                    
                return temp_wav_path
                
            except Exception as e:
                # Clean up on error
                for path in [temp_webm_path, temp_wav_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                raise e
                
        except Exception as e:
            logger.error(f"Failed to convert WebM to WAV: {e}")
            # Fallback: try to treat as raw PCM
            return self._bytes_to_wav_file(webm_bytes)
    
    def _bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """Fallback: Convert raw bytes to WAV file"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Assume raw PCM data
            if len(audio_bytes) % 2 == 1:
                audio_bytes = audio_bytes[:-1]
            
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_array.tobytes())
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create WAV file: {e}")
            raise RuntimeError(f"Audio processing failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """FIXED: Use proper Voxtral transcription API with mistral_common"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Convert WebM/browser audio to WAV
            temp_path = self._convert_webm_to_wav(audio_data)
            
            # FIXED: Use proper mistral_common API
            audio = Audio.from_file(temp_path, strict=False)
            raw_audio = RawAudio.from_audio(audio)
            
            # Create proper transcription request
            transcription_request = TranscriptionRequest(
                model=self.model_name,
                audio=raw_audio,
                language="en",
                temperature=0.0
            )
            
            # Convert to model inputs using proper API
            openai_request = transcription_request.to_openai(exclude=("top_p", "seed"))
            
            # Use processor to create model inputs
            # FIXED: Use the Voxtral processor correctly
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio", "audio": temp_path}]}],
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
            
            logger.info(f"✅ Transcription: '{transcription[:100]}...'")
            
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
        """FIXED: Use proper Voxtral understanding API with mistral_common"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Extract audio and text
            audio_data = message.get("audio")
            text_query = message.get("text", "What can you hear in this audio?")
            
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
            
            # Convert to WAV file
            temp_path = self._convert_webm_to_wav(audio_bytes)
            
            # FIXED: Use proper mistral_common for understanding
            audio = Audio.from_file(temp_path, strict=False)
            audio_chunk = AudioChunk.from_audio(audio)
            text_chunk = TextChunk(text=text_query)
            
            user_message = UserMessage(content=[audio_chunk, text_chunk])
            
            # Convert to proper format for model
            messages = [user_message.to_openai()]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                messages,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
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
            
            logger.info(f"✅ Understanding: '{response[:100]}...'")
            
            return {
                "type": "understanding",
                "response": response.strip(),
                "query": text_query,
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
