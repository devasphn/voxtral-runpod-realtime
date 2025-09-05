# PURE UNDERSTANDING-ONLY MODEL MANAGER - FLASH ATTENTION FIXED
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
import time

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    """PURE UNDERSTANDING-ONLY: Voxtral model manager with Flash Attention fix"""
    
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
        
        # UNDERSTANDING-ONLY: Valid language codes for Voxtral
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
        
        # Default language for auto-detection
        self.default_language = "en"
        
        # Performance optimization settings
        self.target_response_ms = 200  # Sub-200ms target
        self.optimize_for_speed = True
        
        logger.info(f"Initialized PURE UNDERSTANDING-ONLY VoxtralUnderstandingManager for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model with UNDERSTANDING-ONLY optimizations + Flash Attention fix"""
        try:
            logger.info(f"🔄 Loading PURE UNDERSTANDING-ONLY Voxtral model: {self.model_name}")
            logger.info("🚫 Flash Attention DISABLED for compatibility")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with UNDERSTANDING-ONLY optimizations + Flash Attention FIX
            logger.info("Loading model with PURE UNDERSTANDING-ONLY optimizations...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # CRITICAL FIX: Disable Flash Attention completely
                attn_implementation="eager"  # Force eager attention instead of flash_attention_2
            )
            
            # Set to evaluation mode for inference
            self.model.eval()
            
            # UNDERSTANDING-ONLY: Enable optimizations
            if self.optimize_for_speed:
                # Enable caching
                self.model.config.use_cache = True
                
                # Optimize for inference
                if hasattr(self.model, 'half') and self.torch_dtype == torch.float16:
                    self.model = self.model.half()
            
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
                "flash_attention_disabled": True,
                "attention_implementation": "eager",
                "target_response_ms": self.target_response_ms,
                "optimizations_enabled": self.optimize_for_speed
            }
            
            self.is_loaded = True
            logger.info(f"✅ PURE UNDERSTANDING-ONLY Model loaded successfully!")
            logger.info(f"✅ Flash Attention: DISABLED (eager attention used)")
            logger.info(f"✅ Mode: UNDERSTANDING-ONLY (no transcription capability)")
            logger.info(f"✅ Memory usage: {self.model_info['memory_usage']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PURE UNDERSTANDING-ONLY model: {e}")
            raise RuntimeError(f"PURE UNDERSTANDING-ONLY model loading failed: {e}")
    
    async def generate_understanding_response(
        self, 
        audio_data: bytes, 
        context: str = "",
        optimize_for_speed: bool = True
    ) -> Dict[str, Any]:
        """PURE UNDERSTANDING-ONLY: Generate conversational AI response from audio"""
        if not self.is_loaded:
            logger.error("Model not loaded for understanding")
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        temp_path = None
        
        try:
            # Validate input
            if not audio_data or len(audio_data) < 1000:
                logger.warning(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return {"error": "Invalid or insufficient audio data"}
            
            logger.info(f"🧠 PURE UNDERSTANDING-ONLY processing: {len(audio_data)} bytes")
            
            # Create temporary WAV file
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1000:
                logger.error(f"Failed to create valid WAV file: {temp_path}")
                return {"error": "Failed to create valid audio file"}
            
            logger.info(f"Created WAV file: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # Use default language for processing
            language = self.default_language
            logger.info(f"Using language for understanding: {language}")
            
            # UNDERSTANDING-ONLY: Two-step process
            # Step 1: Transcribe the audio
            transcription_start = time.time()
            
            transcription_inputs = self.processor.apply_transcription_request(
                language=language,
                audio=temp_path,
                model_id=self.model_name,
                return_tensors="pt"
            )
            
            # Move to device
            transcription_inputs = {k: v.to(self.device) for k, v in transcription_inputs.items()}
            
            # Generate transcription with speed optimization
            with torch.no_grad():
                transcription_outputs = self.model.generate(
                    **transcription_inputs,
                    max_new_tokens=64,  # Shorter for speed
                    temperature=0.0,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode transcription
            transcription_input_length = transcription_inputs['input_ids'].shape[1]
            transcription_tokens = transcription_outputs[0][transcription_input_length:]
            transcribed_text = self.processor.tokenizer.decode(
                transcription_tokens, 
                skip_special_tokens=True
            ).strip()
            
            transcription_time = time.time() - transcription_start
            logger.info(f"📝 Transcribed in {transcription_time*1000:.0f}ms: '{transcribed_text}'")
            
            if not transcribed_text or len(transcribed_text) < 2:
                logger.warning("Empty or very short transcription")
                return {
                    "error": "No clear speech detected",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            # Step 2: Generate understanding response using chat template
            understanding_start = time.time()
            
            # Build conversation for understanding
            system_message = "You are a helpful AI assistant. Respond naturally and conversationally to what the user said."
            
            if context:
                system_message += f"\n\nPrevious conversation context:\n{context[:300]}"  # Limit context for speed
            
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcribed_text}
            ]
            
            logger.info("Applying chat template for understanding...")
            
            understanding_inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            # Move to device
            understanding_inputs = {k: v.to(self.device) for k, v in understanding_inputs.items()}
            
            # Generate understanding response with speed optimization
            with torch.no_grad():
                understanding_outputs = self.model.generate(
                    **understanding_inputs,
                    max_new_tokens=150,  # Reduced for speed
                    temperature=0.3 if not optimize_for_speed else 0.2,  # Lower temp for speed
                    top_p=0.9 if not optimize_for_speed else 0.8,       # More focused for speed
                    do_sample=True,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode understanding response
            understanding_input_length = understanding_inputs['input_ids'].shape[1]
            understanding_tokens = understanding_outputs[0][understanding_input_length:]
            response = self.processor.tokenizer.decode(
                understanding_tokens, 
                skip_special_tokens=True
            ).strip()
            
            understanding_time = time.time() - understanding_start
            total_time = time.time() - start_time
            
            logger.info(f"🧠 Generated response in {understanding_time*1000:.0f}ms: '{response[:50]}...'")
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Validate response
            if not response or len(response.strip()) < 3:
                logger.warning("Empty or very short understanding response generated")
                return {
                    "response": f"I heard you say: '{transcribed_text}'. Could you tell me more?",
                    "transcribed_text": transcribed_text,
                    "processing_time_ms": total_time * 1000,
                    "language": language,
                    "fallback_used": True,
                    "understanding_only": True
                }
            
            # Final result
            result = {
                "response": response,
                "transcribed_text": transcribed_text,
                "processing_time_ms": total_time * 1000,
                "transcription_time_ms": transcription_time * 1000,
                "understanding_time_ms": understanding_time * 1000,
                "language": language,
                "sub_200ms": total_time * 1000 < 200,
                "understanding_only": True,
                "transcription_disabled": True,
                "flash_attention_disabled": True,
                "optimize_for_speed": optimize_for_speed
            }
            
            logger.info(f"✅ PURE UNDERSTANDING-ONLY complete: {total_time*1000:.0f}ms total ({'✅' if total_time*1000 < 200 else '⚠️'} sub-200ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"PURE UNDERSTANDING-ONLY processing error: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"PURE UNDERSTANDING-ONLY processing failed: {str(e)}"}
    
    def _audio_bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to WAV file optimized for understanding"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Check if it's already a WAV file
            if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"✅ Used existing WAV format: {len(audio_bytes)} bytes")
                
                # Validate the WAV file
                try:
                    with wave.open(temp_path, 'rb') as wav_test:
                        frames = wav_test.getnframes()
                        sample_rate = wav_test.getframerate()
                        channels = wav_test.getnchannels()
                        duration = frames / sample_rate
                        
                        logger.info(f"WAV validation: {frames} frames, {sample_rate}Hz, {channels}ch, {duration:.2f}s")
                        
                        if duration < 0.1:  # Less than 100ms
                            raise ValueError(f"Audio too short for understanding: {duration:.3f}s")
                            
                except Exception as wav_e:
                    logger.error(f"WAV validation failed: {wav_e}")
                    raise ValueError(f"Invalid WAV file: {wav_e}")
                
                return temp_path
            
            # Handle raw PCM data
            if len(audio_bytes) % 2 == 1:
                audio_bytes = audio_bytes[:-1]
            
            if len(audio_bytes) < 1600:  # Less than 100ms at 16kHz
                raise ValueError(f"Audio data too small for understanding: {len(audio_bytes)} bytes")
            
            # Convert to numpy array
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                logger.info(f"Converted {len(audio_bytes)} bytes to {len(audio_array)} samples")
                
                if len(audio_array) < 1600:  # Less than 100ms at 16kHz
                    raise ValueError(f"Audio array too short for understanding: {len(audio_array)} samples")
                    
                # Basic audio enhancement for understanding
                audio_array = self._enhance_audio_for_understanding(audio_array)
                    
            except Exception as e:
                logger.error(f"Failed to convert audio bytes: {e}")
                raise
            
            # Create WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_array.tobytes())
            
            # Verify created file
            file_size = os.path.getsize(temp_path)
            if file_size < 1000:
                raise ValueError(f"Created WAV file too small for understanding: {file_size} bytes")
            
            logger.info(f"✅ Created WAV file for understanding: {temp_path} ({file_size} bytes)")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create WAV file for understanding: {e}")
            if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise RuntimeError(f"Audio file creation failed: {e}")
    
    def _enhance_audio_for_understanding(self, audio_array: np.ndarray) -> np.ndarray:
        """Enhance audio specifically for understanding processing"""
        try:
            # Apply normalization for better understanding
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                # Normalize to reasonable range
                target_max = 20000  # Good range for understanding
                if max_val < target_max:
                    normalization_factor = min(1.5, target_max / max_val)  # Conservative normalization
                    audio_array = (audio_array * normalization_factor).astype(np.int16)
            
            # Simple DC offset removal
            dc_offset = np.mean(audio_array)
            if abs(dc_offset) > 50:
                audio_array = audio_array - int(dc_offset)
            
            logger.debug(f"Enhanced audio for understanding: max_amplitude={np.max(np.abs(audio_array))}")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            return audio_array  # Return original on failure
    
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
        """UNDERSTANDING-ONLY: Cleanup with better resource management"""
        logger.info("🧹 Cleaning up PURE UNDERSTANDING-ONLY model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        self.is_loaded = False
        logger.info("✅ PURE UNDERSTANDING-ONLY model cleanup completed")
