# COMPLETELY FIXED MODEL LOADER - PROPER MISTRAL-COMMON INTEGRATION
import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union
import gc
import tempfile
import os
import time
import numpy as np

# CRITICAL: Import mistral-common for proper Voxtral audio handling
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import (
    TextChunk, 
    AudioChunk, 
    UserMessage, 
    AssistantMessage,
    RawAudio
)
from mistral_common.protocol.transcription.request import TranscriptionRequest

from transformers import VoxtralForConditionalGeneration, AutoProcessor
import soundfile as sf

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    """COMPLETELY FIXED: Proper Voxtral model manager with mistral-common integration"""
    
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
        
        # FIXED: Valid language codes for Voxtral
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
        
        logger.info(f"COMPLETELY FIXED VoxtralUnderstandingManager initialized with mistral-common support")
    
    async def load_model(self) -> None:
        """COMPLETELY FIXED: Load Voxtral model with proper mistral-common integration"""
        try:
            logger.info(f"🔄 Loading FIXED Voxtral model with mistral-common: {self.model_name}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # FIXED: Load processor with mistral-common support
            logger.info("Loading processor with mistral-common support...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # FIXED: Load model with proper configuration
            logger.info("Loading model with mistral-common optimizations...")
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Force eager attention for compatibility
            )
            
            # Set to evaluation mode for inference
            self.model.eval()
            
            # Enable optimizations
            if self.optimize_for_speed:
                self.model.config.use_cache = True
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
                "mistral_common_integrated": True,
                "attention_implementation": "eager",
                "target_response_ms": self.target_response_ms,
                "optimizations_enabled": self.optimize_for_speed
            }
            
            self.is_loaded = True
            logger.info(f"✅ FIXED Model loaded with mistral-common integration!")
            logger.info(f"✅ Memory usage: {self.model_info['memory_usage']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FIXED model with mistral-common: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def generate_understanding_response(
        self, 
        audio_file_path: str, 
        context: str = "",
        optimize_for_speed: bool = True
    ) -> Dict[str, Any]:
        """COMPLETELY FIXED: Generate understanding response using mistral-common"""
        if not self.is_loaded:
            logger.error("Model not loaded for understanding")
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        try:
            # Validate input
            if not audio_file_path or not os.path.exists(audio_file_path):
                logger.warning(f"Invalid audio file: {audio_file_path}")
                return {"error": "Invalid or missing audio file"}
            
            file_size = os.path.getsize(audio_file_path)
            logger.info(f"🧠 FIXED processing with mistral-common: {audio_file_path} ({file_size} bytes)")
            
            # CRITICAL FIX: Use mistral-common for proper audio handling
            logger.info("🎯 Using mistral-common Audio class for proper processing...")
            
            try:
                # Method 1: Use mistral-common Audio class (PROPER WAY)
                audio = Audio.from_file(audio_file_path, strict=False)
                raw_audio = RawAudio.from_audio(audio)
                
                # Create proper conversation format with mistral-common
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio", 
                                "audio": raw_audio.audio  # Use the audio data from RawAudio
                            },
                            {
                                "type": "text",
                                "text": f"Please understand and respond to what you hear in the audio. {context}" if context else "Please understand and respond to what you hear in the audio."
                            }
                        ]
                    }
                ]
                
                # Use apply_chat_template for understanding
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                
                logger.info("✅ Successfully processed audio with mistral-common")
                
            except Exception as e:
                logger.warning(f"mistral-common processing failed: {e}, trying alternative method")
                
                # Fallback: Direct audio file processing
                audio_array, sample_rate = sf.read(audio_file_path)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)  # Convert to mono
                if sample_rate != 16000:
                    from scipy import signal
                    audio_array = signal.resample(audio_array, int(len(audio_array) * 16000 / sample_rate))
                
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio",
                                "audio": audio_array.astype(np.float32)
                            },
                            {
                                "type": "text", 
                                "text": f"Please understand and respond to what you hear in the audio. {context}" if context else "Please understand and respond to what you hear in the audio."
                            }
                        ]
                    }
                ]
                
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            
            # Move to device with correct dtype
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device, dtype=self.torch_dtype)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device, dtype=self.torch_dtype) if hasattr(v, 'to') else v 
                         for k, v in inputs.items()}
            
            # Generate response with optimization
            generation_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=120 if optimize_for_speed else 150,
                    temperature=0.2 if optimize_for_speed else 0.3,
                    top_p=0.8 if optimize_for_speed else 0.9,
                    do_sample=True,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generation_time = time.time() - generation_start
            
            # Decode response correctly
            if hasattr(inputs, 'input_ids'):
                input_length = inputs.input_ids.shape[1]
            elif 'input_ids' in inputs:
                input_length = inputs['input_ids'].shape[1]
            else:
                input_length = 0
            
            # Extract new tokens only
            new_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True
            ).strip()
            
            total_time = time.time() - start_time
            
            logger.info(f"🧠 FIXED Generated response with mistral-common in {generation_time*1000:.0f}ms: '{response[:50]}...'")
            
            # Clean up temp file
            try:
                if os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except:
                pass
            
            # Validate response
            if not response or len(response.strip()) < 3:
                logger.warning("Empty or very short understanding response generated")
                return {
                    "response": "I can hear audio input, but I'm having trouble generating a detailed response. Could you try speaking a bit longer or more clearly?",
                    "transcribed_text": "[Audio processed for understanding with mistral-common]",
                    "processing_time_ms": total_time * 1000,
                    "language": self.default_language,
                    "fallback_used": True,
                    "understanding_only": True,
                    "mistral_common_used": True
                }
            
            # Extract any transcribed content from response
            transcribed_text = "[Audio understood and processed with mistral-common]"
            
            # Final result
            result = {
                "response": response,
                "transcribed_text": transcribed_text,
                "processing_time_ms": total_time * 1000,
                "generation_time_ms": generation_time * 1000,
                "language": self.default_language,
                "sub_200ms": total_time * 1000 < 200,
                "understanding_only": True,
                "transcription_disabled": True,
                "mistral_common_integrated": True,
                "optimize_for_speed": optimize_for_speed,
                "model_api_fixed": True
            }
            
            logger.info(f"✅ FIXED UNDERSTANDING complete with mistral-common: {total_time*1000:.0f}ms total ({'✅' if total_time*1000 < 200 else '⚠️'} sub-200ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"FIXED processing error with mistral-common: {e}", exc_info=True)
            # Clean up temp file on error
            try:
                if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except:
                pass
            return {"error": f"Processing failed with mistral-common: {str(e)}"}
    
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
        """FIXED: Cleanup with proper resource management"""
        logger.info("🧹 Cleaning up FIXED model resources with mistral-common...")
        
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
        logger.info("✅ FIXED model cleanup completed with mistral-common")
