# FIXED MODEL LOADER - WITH CORRECT LANGUAGE CODES AND PROPER ERROR HANDLING
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
    """FIXED: Voxtral model manager with correct language handling and context support"""
    
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
        
        # FIXED: Valid language codes for Voxtral (ISO 639-1)
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
        
        logger.info(f"Initialized FIXED VoxtralModelManager for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model with proper error handling"""
        try:
            logger.info(f"🔄 Loading FIXED Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with optimizations
            logger.info("Loading model...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Store model info
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage(),
                "supported_languages": list(self.supported_languages.values()),
                "fixed_version": True
            }
            
            self.is_loaded = True
            logger.info(f"✅ FIXED Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FIXED model: {e}")
            raise RuntimeError(f"FIXED model loading failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes, context: str = "", language: str = None) -> Dict[str, Any]:
        """FIXED: Audio transcription with correct language handling"""
        if not self.is_loaded:
            logger.error("Model not loaded for transcription")
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                logger.warning(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return {"error": "Invalid or insufficient audio data"}
            
            logger.info(f"FIXED transcription processing: {len(audio_data)} bytes")
            
            # Create temporary WAV file
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                logger.error(f"Failed to create valid WAV file: {temp_path}")
                return {"error": "Failed to create valid audio file"}
            
            logger.info(f"Created WAV file: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # FIXED: Use proper language codes or None for auto-detection
            valid_language = None
            if language and language in self.supported_languages:
                valid_language = language
            elif language and language not in ["auto", None]:
                logger.warning(f"Invalid language code: {language}, using auto-detection")
            
            # Enhanced transcription with context
            if context:
                # Use context-aware transcription
                conversation = [
                    {
                        "role": "system",
                        "content": f"Previous conversation context: {context[:500]}..."
                    },
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
                
                inputs = self.processor.apply_chat_template(
                    conversation,
                    return_tensors="pt"
                )
            else:
                # FIXED: Standard transcription with proper language handling
                inputs = self.processor.apply_transcription_request(
                    audio=temp_path,
                    language=valid_language,  # FIXED: Use valid language or None
                    model_id=self.model_name,
                    return_tensors="pt"
                )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("Running FIXED transcription inference...")
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,  # Deterministic
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
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
            
            logger.info(f"FIXED transcription result: '{transcription}'")
            
            # Detect language from text if not specified
            detected_language = self._detect_language_from_text(transcription) if not valid_language else valid_language
            
            # Return result
            if not transcription:
                logger.warning("Empty transcription generated")
                return {
                    "type": "transcription",
                    "text": "",
                    "language": detected_language or "unknown",
                    "confidence": 0.0,
                    "timestamp": asyncio.get_event_loop().time(),
                    "fixed": True,
                    "context_used": bool(context)
                }
            
            logger.info(f"✅ FIXED transcription successful: '{transcription}' (lang: {detected_language})")
            
            return {
                "type": "transcription",
                "text": transcription,
                "language": detected_language or "auto",
                "confidence": 0.95,
                "timestamp": asyncio.get_event_loop().time(),
                "fixed": True,
                "context_used": bool(context)
            }
            
        except Exception as e:
            logger.error(f"FIXED transcription error: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"FIXED transcription failed: {str(e)}"}
    
    async def understand_audio(self, audio_data: bytes, query: str = None, context: str = "") -> Dict[str, Any]:
        """FIXED: Audio understanding with proper error handling"""
        if not self.is_loaded:
            logger.error("Model not loaded for understanding")
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                logger.warning(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return {"error": "Invalid or insufficient audio data"}
            
            logger.info(f"FIXED understanding processing: {len(audio_data)} bytes")
            
            # Create temporary WAV file
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                logger.error(f"Failed to create valid WAV file: {temp_path}")
                return {"error": "Failed to create valid audio file"}
            
            # Enhanced conversation with context
            system_message = "You are a helpful AI assistant that can understand and respond to audio in multiple languages."
            
            if context:
                system_message += f"\n\nConversation context:\n{context[:800]}"
            
            if query:
                system_message += f"\n\nUser's question: {query}"
            
            conversation = [
                {
                    "role": "system",
                    "content": system_message
                },
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
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("Running FIXED understanding inference...")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
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
            
            logger.info(f"FIXED understanding result: '{response}'")
            
            # Extract transcription if possible
            transcription = self._extract_transcription_from_response(response)
            detected_language = self._detect_language_from_text(response)
            
            # Return result
            if not response:
                logger.warning("Empty understanding response generated")
                return {
                    "type": "understanding",
                    "response": "I couldn't understand the audio clearly. Could you please repeat?",
                    "transcription": "",
                    "query": query or "Audio understanding",
                    "timestamp": asyncio.get_event_loop().time(),
                    "fixed": True,
                    "context_used": bool(context),
                    "language": detected_language or "unknown"
                }
            
            logger.info(f"✅ FIXED understanding successful: '{response[:100]}...'")
            
            return {
                "type": "understanding",
                "response": response,
                "transcription": transcription,
                "query": query or "Audio understanding",
                "timestamp": asyncio.get_event_loop().time(),
                "fixed": True,
                "context_used": bool(context),
                "language": detected_language or "auto"
            }
            
        except Exception as e:
            logger.error(f"FIXED understanding error: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"FIXED understanding failed: {str(e)}"}
    
    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text"""
        if not text:
            return "unknown"
        
        # Simple character-based detection
        hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहा़ि्ीुूेैोौंःऽ')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        text_chars = set(text.lower())
        has_hindi = bool(text_chars.intersection(hindi_chars))
        has_english = bool(text_chars.intersection(english_chars))
        
        if has_hindi and has_english:
            return "hi-en"
        elif has_hindi:
            return "hi"
        elif has_english:
            return "en"
        else:
            return "auto"
    
    def _extract_transcription_from_response(self, response: str) -> str:
        """Extract transcription from understanding response"""
        if '"' in response:
            import re
            quoted = re.findall(r'"([^"]*)"', response)
            if quoted:
                return quoted[0]
        return ""
    
    def _audio_bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """FIXED: Convert audio bytes to WAV file"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd)
            
            # Check if it's already a WAV file
            if audio_bytes.startswith(b'RIFF') and b'WAVE' in audio_bytes[:20]:
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"✅ Used existing WAV format: {len(audio_bytes)} bytes")
                return temp_path
            
            # Handle raw PCM data
            if len(audio_bytes) % 2 == 1:
                audio_bytes = audio_bytes[:-1]
            
            if len(audio_bytes) < 32:
                raise ValueError(f"Audio data too small: {len(audio_bytes)} bytes")
            
            # Convert to numpy array
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                logger.info(f"Converted {len(audio_bytes)} bytes to {len(audio_array)} samples")
            except Exception as e:
                logger.error(f"Failed to convert audio bytes: {e}")
                raise
            
            # Create WAV file
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_array.tobytes())
            
            logger.info(f"✅ Created FIXED WAV file: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create FIXED WAV file: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise RuntimeError(f"FIXED audio file creation failed: {e}")
    
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
        """FIXED: Cleanup with better resource management"""
        logger.info("🧹 Cleaning up FIXED model resources...")
        
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
        logger.info("✅ FIXED model cleanup completed")
