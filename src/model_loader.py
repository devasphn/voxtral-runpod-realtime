import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union, List
import gc
from pathlib import Path
import tempfile

from transformers import VoxtralForConditionalGeneration, AutoProcessor
from mistral_common.audio import Audio
import numpy as np
import requests
from urllib.parse import urlparse

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
    
    async def _process_audio_input(self, audio_input: Union[str, bytes, io.BytesIO]) -> str:
        """Process audio input which can be a file path, URL, bytes, or BytesIO"""
        temp_path = None
        try:
            if isinstance(audio_input, str):
                # Handle URL or file path
                if audio_input.startswith(('http://', 'https://')):
                    # Download from URL
                    response = requests.get(audio_input)
                    response.raise_for_status()
                    audio_path = Path(tempfile.mktemp(suffix='.wav'))
                    with open(audio_path, 'wb') as f:
                        f.write(response.content)
                else:
                    audio_path = Path(audio_input)
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found: {audio_input}")
                return str(audio_path)
            
            # Handle bytes or BytesIO
            elif isinstance(audio_input, (bytes, io.BytesIO)):
                # Convert to bytes if it's BytesIO
                if isinstance(audio_input, io.BytesIO):
                    audio_bytes = audio_input.getvalue()
                else:
                    audio_bytes = audio_input
                
                # Create a temporary WAV file with proper format
                temp_path = Path(tempfile.mktemp(suffix='.wav'))
                
                # Try to load with pydub first to handle format conversion
                try:
                    from pydub import AudioSegment
                    import io as py_io
                    
                    # Try to detect format from bytes
                    audio = AudioSegment.from_file(py_io.BytesIO(audio_bytes))
                    # Convert to required format: 16kHz, 16-bit, mono
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    audio.export(temp_path, format='wav')
                    return str(temp_path)
                except ImportError:
                    logger.warning("pydub not installed, falling back to basic WAV handling")
                    # Fallback to basic WAV handling if pydub is not available
                    with wave.open(str(temp_path), 'wb') as wav_file:
                        wav_file.setnchannels(1)      # Mono
                        wav_file.setsampwidth(2)      # 16-bit
                        wav_file.setframerate(16000)  # 16kHz
                        wav_file.writeframes(audio_bytes)
                    return str(temp_path)
            
            else:
                raise ValueError("audio_input must be a file path, URL, bytes, or BytesIO")
                
        except Exception as e:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise RuntimeError(f"Error processing audio input: {str(e)}")

    async def transcribe_audio(
        self, 
        audio_input: Union[str, bytes, io.BytesIO], 
        language: str = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Can be a file path, URL, or raw bytes of the audio
            language: Optional language code (e.g., 'en', 'es', 'fr')
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Dict containing transcription and metadata
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Process audio input (download if URL, save bytes to temp file if needed)
            audio_path = await self._process_audio_input(audio_input)
            
            # Create conversation with audio for transcription
            conversation = [
                {
                    "role": "user", 
                    "content": [{"type": "audio", "path": audio_path}]
                }
            ]
            
            # Apply chat template and get inputs
            inputs = self.processor.apply_chat_template(
                conversation, 
                return_tensors="pt"
            )
            
            # Move to device and set dtype
            inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                     for k, v in inputs.items()}
            
            # Set default generation parameters if not provided
            if 'max_new_tokens' not in generation_kwargs:
                generation_kwargs['max_new_tokens'] = 1000
            if 'temperature' not in generation_kwargs:
                generation_kwargs['temperature'] = 0.0  # Deterministic for transcription
            if 'do_sample' not in generation_kwargs:
                generation_kwargs['do_sample'] = False
            if 'pad_token_id' not in generation_kwargs:
                generation_kwargs['pad_token_id'] = self.processor.tokenizer.eos_token_id
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode output
            generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            decoded_output = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file if we created one
            if isinstance(audio_input, bytes) or (isinstance(audio_input, str) and audio_input.startswith(('http://', 'https://'))):
                try:
                    Path(audio_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {audio_path}: {e}")
            
            return {
                "type": "transcription",
                "text": decoded_output.strip(),
                "language": language or "auto-detected",
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def understand_audio(
        self, 
        audio_input: Union[str, bytes, io.BytesIO],
        text_query: str = "What can you hear in this audio?",
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Process audio with understanding capabilities
        
        Args:
            audio_input: Can be a file path, URL, or raw bytes of the audio
            text_query: The question or instruction about the audio
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dict containing the model's response and metadata
        """
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        try:
            # Process audio input (download if URL, save bytes to temp file if needed)
            audio_path = await self._process_audio_input(audio_input)
            
            # Create conversation with audio and text
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": audio_path},
                        {"type": "text", "text": text_query}
                    ]
                }
            ]
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                conversation, 
                return_tensors="pt"
            )
            
            # Move to device and set dtype
            inputs = {k: v.to(device=self.device, dtype=self.torch_dtype) 
                     for k, v in inputs.items()}
            
            # Set default generation parameters if not provided
            if 'max_new_tokens' not in generation_kwargs:
                generation_kwargs['max_new_tokens'] = 1000
            if 'temperature' not in generation_kwargs:
                generation_kwargs['temperature'] = 0.2
            if 'top_p' not in generation_kwargs:
                generation_kwargs['top_p'] = 0.95
            if 'do_sample' not in generation_kwargs:
                generation_kwargs['do_sample'] = True
            if 'pad_token_id' not in generation_kwargs:
                generation_kwargs['pad_token_id'] = self.processor.tokenizer.eos_token_id
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode output
            generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            decoded_output = self.processor.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            # Clean up temp file if we created one
            if isinstance(audio_input, bytes) or (isinstance(audio_input, str) and audio_input.startswith(('http://', 'https://'))):
                try:
                    Path(audio_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {audio_path}: {e}")
            
            return {
                "type": "understanding",
                "text": decoded_output.strip(),
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Audio understanding error: {e}", exc_info=True)
            return {"error": f"Audio understanding failed: {str(e)}"}
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return audio_obj
            
        except Exception as e:
            logger.error(f"Audio object creation error: {e}")
            
            # Clean up temp file on error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return None
    
    def _write_wav_file(self, audio_bytes: bytes, output_path: str):
        """Write audio bytes to WAV file with proper format"""
        try:
            # Check if already WAV format
            if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                # Already WAV, just write
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                return
            
            # Assume raw PCM and convert to WAV
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                
                # Ensure audio_bytes is proper length for 16-bit samples
                if len(audio_bytes) % 2 != 0:
                    audio_bytes = audio_bytes[:-1]  # Remove last byte if odd
                
                wav_file.writeframes(audio_bytes)
                
        except Exception as e:
            logger.error(f"WAV file creation error: {e}")
            # Fallback: write raw bytes
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
    
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
