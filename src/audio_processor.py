import asyncio
import logging
import time
import tempfile
import os
import base64
import torch
import numpy as np
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, AudioChunk, TextChunk
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import wave

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    def __init__(self, model_name="mistralai/Voxtral-Mini-3B-2507", device="cuda", torch_dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.is_loaded = False
        self.model = None
        self.processor = None
        self.model_info = {}
        self.target_response_ms = 200

    async def load_model(self):
        """Load the Voxtral model and processor"""
        try:
            logger.info(f"Loading model {self.model_name}")
            
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("✅ Processor loaded successfully")
            
            # Load model with correct configuration
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Force eager attention for compatibility
            )
            
            # Put model in evaluation mode
            self.model.eval()
            logger.info("✅ Model loaded successfully")
            
            # Update model info
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage(),
                "flash_attention_disabled": True,
                "understanding_only": True
            }
            
            self.is_loaded = True
            logger.info(f"✅ Voxtral model loaded on {self.device} with {self.model_info['parameters']} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise

    def _count_parameters(self):
        """Count model parameters"""
        if not self.model:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def _get_memory_usage(self):
        """Get memory usage information"""
        memory_info = {"cpu_memory": 0, "gpu_memory": 0}
        
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_memory"] = process.memory_info().rss / 1024**3  # GB
        except Exception as e:
            logger.warning(f"Could not get CPU memory usage: {e}")
        
        if torch.cuda.is_available() and self.device.type == "cuda":
            try:
                memory_info.update({
                    "gpu_allocated_gb": torch.cuda.memory_allocated(self.device) / 1024**3,
                    "gpu_cached_gb": torch.cuda.memory_reserved(self.device) / 1024**3,
                    "gpu_total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory usage: {e}")
        
        return memory_info

    def _pcm_bytes_to_audio(self, pcm_data: bytes) -> Audio:
        """Convert PCM bytes to mistral Audio object"""
        try:
            # Create temporary WAV file from PCM data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write PCM data as WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(pcm_data)
                
                # Load using mistral Audio
                audio = Audio.from_file(temp_path)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return audio
                
        except Exception as e:
            logger.error(f"Failed to convert PCM to Audio: {e}")
            return None

    def _bytes_to_audio(self, audio_data: bytes) -> Audio:
        """Convert various audio formats to mistral Audio object"""
        try:
            # If it's already PCM data (raw bytes), treat as 16-bit PCM
            if len(audio_data) > 0:
                # First try as PCM data
                try:
                    return self._pcm_bytes_to_audio(audio_data)
                except Exception:
                    pass
                
                # Try as audio file (WebM, WAV, etc.)
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                try:
                    # Try to load directly
                    audio = Audio.from_file(temp_path)
                    os.unlink(temp_path)
                    return audio
                except Exception:
                    # If direct loading fails, try conversion via ffmpeg
                    try:
                        import subprocess
                        wav_path = temp_path + ".wav"
                        subprocess.run([
                            'ffmpeg', '-y', '-loglevel', 'error',
                            '-i', temp_path,
                            '-ar', '16000', '-ac', '1', '-f', 'wav',
                            wav_path
                        ], check=True, capture_output=True)
                        
                        audio = Audio.from_file(wav_path)
                        
                        # Clean up
                        os.unlink(temp_path)
                        os.unlink(wav_path)
                        
                        return audio
                        
                    except Exception as e:
                        logger.error(f"FFmpeg conversion failed: {e}")
                        os.unlink(temp_path)
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert bytes to Audio: {e}")
            return None

    async def understand_audio(self, message):
        """Process audio for understanding using proper Voxtral API"""
        if not self.is_loaded:
            return {"error": "Model not loaded", "type": "understanding"}

        try:
            start_time = time.time()
            
            # Get audio data
            audio_data = message.get("audio")
            text_prompt = message.get("text", "Listen to this audio and provide a helpful response.")
            
            if not audio_data:
                return {"error": "No audio data provided", "type": "understanding"}
            
            # Handle different audio data formats
            if isinstance(audio_data, str):
                try:
                    audio_data = base64.b64decode(audio_data)
                except Exception:
                    return {"error": "Invalid base64 audio data", "type": "understanding"}
            
            if not isinstance(audio_data, bytes):
                return {"error": "Audio data must be bytes or base64 string", "type": "understanding"}
            
            # Convert to mistral Audio object
            audio = self._bytes_to_audio(audio_data)
            if audio is None:
                return {"error": "Failed to process audio data", "type": "understanding"}
            
            # Create proper conversation format for Voxtral
            conversation = [
                UserMessage(content=[
                    AudioChunk(audio=audio),
                    TextChunk(text=text_prompt)
                ])
            ]
            
            # Apply chat template using processor
            inputs = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Move inputs to correct device and dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Limit for faster responses
                    temperature=0.2,     # Slightly creative but focused
                    top_p=0.9,          # Focused sampling
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response (only new tokens)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                response_tokens, 
                skip_special_tokens=True
            ).strip()
            
            # Calculate timing
            processing_time_ms = (time.time() - start_time) * 1000
            sub_200ms = processing_time_ms < 200
            
            logger.info(f"Understanding response generated in {processing_time_ms:.1f}ms")
            
            return {
                "type": "understanding",
                "response": response,
                "transcription": "[Audio processed]",  # We don't do transcription
                "processing_time_ms": processing_time_ms,
                "response_time_ms": processing_time_ms,
                "sub_200ms": sub_200ms,
                "understanding_only": True,
                "flash_attention_disabled": True,
                "audio_duration_ms": len(audio_data) / 32,  # Rough estimate
                "speech_quality": 0.9,  # Placeholder
                "gap_detected": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Understanding processing failed: {e}")
            return {
                "type": "understanding",
                "error": f"Processing failed: {str(e)}",
                "timestamp": time.time()
            }

    async def cleanup(self):
        """Clean up model resources"""
        logger.info("Cleaning up Voxtral model...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.is_loaded = False
        logger.info("✅ Model cleanup completed")
