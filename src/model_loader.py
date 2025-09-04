# ENHANCED MODEL LOADER - WITH CONVERSATION CONTEXT AND MULTILINGUAL SUPPORT
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
    """ENHANCED: Voxtral model manager with conversation context and multilingual support"""
    
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
        
        # Enhanced language support
        self.supported_languages = {
            "en": "English",
            "hi": "Hindi", 
            "hi-en": "Hindi-English Mixed",
            "es": "Spanish",
            "fr": "French",
            "pt": "Portuguese",
            "de": "German",
            "nl": "Dutch",
            "it": "Italian"
        }
        
        logger.info(f"Initialized Enhanced VoxtralModelManager for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """Load Voxtral model and processor with enhanced error handling"""
        try:
            logger.info(f"🔄 Loading Enhanced Voxtral model: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load processor with enhanced language support
            logger.info("Loading processor with multilingual support...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with optimizations
            logger.info("Loading model with conversation optimizations...")
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
            
            # Store enhanced model info
            self.model_info = {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.torch_dtype),
                "parameters": self._count_parameters(),
                "memory_usage": self._get_memory_usage(),
                "supported_languages": list(self.supported_languages.values()),
                "conversation_enhanced": True,
                "multilingual_support": True
            }
            
            self.is_loaded = True
            logger.info(f"✅ Enhanced Model loaded successfully: {self.model_info}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load enhanced model: {e}")
            raise RuntimeError(f"Enhanced model loading failed: {e}")
    
    async def transcribe_audio(self, audio_data: bytes, context: str = "", language: str = None) -> Dict[str, Any]:
        """ENHANCED TRANSCRIPTION: Audio → Text with conversation context"""
        if not self.is_loaded:
            logger.error("Model not loaded for transcription")
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                logger.warning(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return {"error": "Invalid or insufficient audio data"}
            
            logger.info(f"Enhanced transcription processing: {len(audio_data)} bytes with context: {bool(context)}")
            
            # Create temporary WAV file from audio data
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                logger.error(f"Failed to create valid WAV file: {temp_path}")
                return {"error": "Failed to create valid audio file"}
            
            logger.info(f"Created WAV file: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # Enhanced transcription request with context
            if context:
                # Use context-aware transcription
                conversation = [
                    {
                        "role": "system",
                        "content": f"Previous conversation context: {context}\\n\\nTranscribe the following audio in the same language style as the context. Handle code-switching between Hindi and English naturally."
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
                # Standard transcription
                inputs = self.processor.apply_transcription_request(
                    audio=temp_path,
                    language=language or "auto",  # Auto-detect language
                    model_id=self.model_name,
                    return_tensors="pt"
                )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("Running enhanced transcription inference...")
            
            # Generate transcription with better parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,  # Deterministic for transcription
                    do_sample=False,
                    repetition_penalty=1.1,  # Prevent repetition
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
            
            logger.info(f"Enhanced transcription result: '{transcription}'")
            
            # Detect language
            detected_language = self._detect_language_from_text(transcription)
            
            # Return enhanced result
            if not transcription:
                logger.warning("Empty transcription generated")
                return {
                    "type": "transcription",
                    "text": "",
                    "language": detected_language,
                    "confidence": 0.0,
                    "timestamp": asyncio.get_event_loop().time(),
                    "enhanced": True,
                    "context_used": bool(context),
                    "debug": "empty_transcription"
                }
            
            logger.info(f"✅ Enhanced transcription successful: '{transcription}' (lang: {detected_language})")
            
            return {
                "type": "transcription",
                "text": transcription,
                "language": detected_language,
                "confidence": 0.95,
                "timestamp": asyncio.get_event_loop().time(),
                "enhanced": True,
                "context_used": bool(context),
                "multilingual": detected_language in ["hi-en", "mixed"]
            }
            
        except Exception as e:
            logger.error(f"Enhanced transcription error: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"Enhanced transcription failed: {str(e)}"}
    
    async def understand_audio(self, audio_data: bytes, query: str = None, context: str = "") -> Dict[str, Any]:
        """ENHANCED UNDERSTANDING: Audio → Intelligent Response with conversation context"""
        if not self.is_loaded:
            logger.error("Model not loaded for understanding")
            return {"error": "Model not loaded"}
        
        temp_path = None
        try:
            # Validate input
            if not audio_data or len(audio_data) < 100:
                logger.warning(f"Invalid audio data: {len(audio_data) if audio_data else 0} bytes")
                return {"error": "Invalid or insufficient audio data"}
            
            logger.info(f"Enhanced understanding processing: {len(audio_data)} bytes with context: {bool(context)}")
            
            # Create temporary WAV file from audio data
            temp_path = self._audio_bytes_to_wav_file(audio_data)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 100:
                logger.error(f"Failed to create valid WAV file: {temp_path}")
                return {"error": "Failed to create valid audio file"}
            
            logger.info(f"Created WAV file: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # Enhanced conversation with context
            system_message = "You are a helpful AI assistant that can understand and respond to audio in multiple languages, including Hindi and English. You can handle code-switching naturally."
            
            if context:
                system_message += f"\\n\\nPrevious conversation context:\\n{context}\\n\\nPlease provide a response that considers this conversation history and maintains context continuity."
            
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
            
            # Apply chat template for enhanced understanding
            inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("Running enhanced understanding inference...")
            
            # Generate intelligent response with better parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,  # Slightly creative but consistent
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
            
            logger.info(f"Enhanced understanding result: '{response}'")
            
            # Try to extract transcription from the response (model might include it)
            transcription = self._extract_transcription_from_response(response)
            detected_language = self._detect_language_from_text(response)
            
            # Return enhanced result
            if not response:
                logger.warning("Empty understanding response generated")
                return {
                    "type": "understanding",
                    "response": "I couldn't understand the audio clearly. Could you please repeat?",
                    "transcription": "",
                    "query": query or "Audio understanding",
                    "timestamp": asyncio.get_event_loop().time(),
                    "enhanced": True,
                    "context_used": bool(context),
                    "language": detected_language,
                    "debug": "empty_response"
                }
            
            logger.info(f"✅ Enhanced understanding successful: '{response[:100]}...' (lang: {detected_language})")
            
            return {
                "type": "understanding",
                "response": response,
                "transcription": transcription,
                "query": query or "Audio understanding with context",
                "timestamp": asyncio.get_event_loop().time(),
                "enhanced": True,
                "context_used": bool(context),
                "language": detected_language,
                "multilingual": detected_language in ["hi-en", "mixed"]
            }
            
        except Exception as e:
            logger.error(f"Enhanced understanding error: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            return {"error": f"Enhanced understanding failed: {str(e)}"}
    
    def _detect_language_from_text(self, text: str) -> str:
        """Detect language from text for better response categorization"""
        if not text:
            return "unknown"
        
        # Simple character-based language detection
        hindi_chars = set('अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहा़ि्ीुूेैोौंःऽ')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        text_chars = set(text.lower())
        has_hindi = bool(text_chars.intersection(hindi_chars))
        has_english = bool(text_chars.intersection(english_chars))
        
        if has_hindi and has_english:
            return "hi-en"  # Code-switched
        elif has_hindi:
            return "hi"
        elif has_english:
            return "en"
        else:
            return "auto"
    
    def _extract_transcription_from_response(self, response: str) -> str:
        """Try to extract transcription if model includes it in understanding response"""
        # Simple heuristic - if response starts with quotes or "User said:", extract it
        if '"' in response:
            # Try to find quoted text
            import re
            quoted = re.findall(r'"([^"]*)"', response)
            if quoted:
                return quoted[0]
        
        # Look for patterns like "User said: ..." or "You said: ..."
        patterns = [
            r"(?:User said|You said|I heard):\s*(.+?)(?:\\n|\\.|$)",
            r"(?:Audio contains|Audio says):\s*(.+?)(?:\\n|\\.|$)"
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""  # No transcription found
    
    def _audio_bytes_to_wav_file(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to WAV file with enhanced error handling"""
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
                raise ValueError(f"Audio data too small: {len(audio_bytes)} bytes")
            
            # Convert to numpy array
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                logger.info(f"Converted {len(audio_bytes)} bytes to {len(audio_array)} samples")
            except Exception as e:
                logger.error(f"Failed to convert audio bytes to array: {e}")
                raise
            
            # Create WAV file with proper parameters
            with wave.open(temp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_array.tobytes())
            
            logger.info(f"✅ Created enhanced WAV file: {len(audio_bytes)} bytes → {temp_path} ({os.path.getsize(temp_path)} bytes)")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create enhanced WAV file: {e}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise RuntimeError(f"Enhanced audio file creation failed: {e}")
    
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
        """Enhanced cleanup with better resource management"""
        logger.info("🧹 Cleaning up enhanced model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Enhanced GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        
        gc.collect()
        self.is_loaded = False
        logger.info("✅ Enhanced model cleanup completed")
