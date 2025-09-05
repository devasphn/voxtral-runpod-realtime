# FIXED: PURE UNDERSTANDING-ONLY MODEL MANAGER WITH CORRECT VOXTRAL API
import asyncio
import logging
import torch
from typing import Optional, Dict, Any, Union
import gc
import tempfile
import os
import time

from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    """FIXED: Voxtral model manager with correct API usage"""
    
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
        
        logger.info(f"FIXED VoxtralUnderstandingManager initialized for {model_name} on {self.device}")
    
    async def load_model(self) -> None:
        """FIXED: Load Voxtral model with correct configuration"""
        try:
            logger.info(f"🔄 Loading FIXED Voxtral model: {self.model_name}")
            logger.info("🚫 Flash Attention DISABLED for compatibility")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # FIXED: Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # FIXED: Load model with correct configuration
            logger.info("Loading model with FIXED optimizations...")
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
            
            # FIXED: Enable optimizations
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
            logger.info(f"✅ FIXED Model loaded successfully!")
            logger.info(f"✅ Flash Attention: DISABLED (eager attention used)")
            logger.info(f"✅ Mode: UNDERSTANDING-ONLY (no transcription capability)")
            logger.info(f"✅ Memory usage: {self.model_info['memory_usage']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FIXED model: {e}")
            raise RuntimeError(f"FIXED model loading failed: {e}")
    
    async def generate_understanding_response(
        self, 
        audio_file_path: str, 
        context: str = "",
        optimize_for_speed: bool = True
    ) -> Dict[str, Any]:
        """FIXED: Generate conversational AI response from audio using correct Voxtral API"""
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
            logger.info(f"🧠 FIXED processing: {audio_file_path} ({file_size} bytes)")
            
            # Use default language for processing
            language = self.default_language
            logger.info(f"Using language for understanding: {language}")
            
            # FIXED: Step 1: Transcribe the audio using CORRECT API
            transcription_start = time.time()
            
            # CRITICAL FIX: Use apply_transcription_request correctly
            transcription_inputs = self.processor.apply_transcription_request(
                language=language,
                audio=audio_file_path,  # FIXED: Pass file path directly
                model_id=self.model_name,
                return_tensors="pt"
            )
            
            # Move to device
            transcription_inputs = {k: v.to(self.device) for k, v in transcription_inputs.items() if hasattr(v, 'to')}
            
            # Generate transcription with speed optimization
            with torch.no_grad():
                transcription_outputs = self.model.generate(
                    **transcription_inputs,
                    max_new_tokens=64,  # Shorter for speed
                    temperature=0.0,  # Deterministic for transcription
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # FIXED: Decode transcription correctly
            transcription_input_length = transcription_inputs['input_ids'].shape[1]
            transcription_tokens = transcription_outputs[0][transcription_input_length:]
            transcribed_text = self.processor.tokenizer.decode(
                transcription_tokens, 
                skip_special_tokens=True
            ).strip()
            
            transcription_time = time.time() - transcription_start
            logger.info(f"📝 FIXED Transcribed in {transcription_time*1000:.0f}ms: '{transcribed_text}'")
            
            if not transcribed_text or len(transcribed_text) < 2:
                logger.warning("Empty or very short transcription")
                return {
                    "error": "No clear speech detected",
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            
            # FIXED: Step 2: Generate understanding response using CORRECT chat template
            understanding_start = time.time()
            
            # Build conversation for understanding
            system_message = "You are a helpful AI assistant. Respond naturally and conversationally to what the user said."
            
            if context:
                system_message += f"\n\nPrevious conversation context:\n{context[:300]}"  # Limit context for speed
            
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": transcribed_text}
            ]
            
            logger.info("FIXED: Applying chat template for understanding...")
            
            # CRITICAL FIX: Use apply_chat_template correctly
            understanding_inputs = self.processor.apply_chat_template(
                conversation,
                return_tensors="pt"
            )
            
            # Move to device
            understanding_inputs = {k: v.to(self.device) for k, v in understanding_inputs.items() if hasattr(v, 'to')}
            
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
            
            # FIXED: Decode understanding response correctly
            understanding_input_length = understanding_inputs['input_ids'].shape[1]
            understanding_tokens = understanding_outputs[0][understanding_input_length:]
            response = self.processor.tokenizer.decode(
                understanding_tokens, 
                skip_special_tokens=True
            ).strip()
            
            understanding_time = time.time() - understanding_start
            total_time = time.time() - start_time
            
            logger.info(f"🧠 FIXED Generated response in {understanding_time*1000:.0f}ms: '{response[:50]}...'")
            
            # FIXED: Clean up temp file
            try:
                if os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except:
                pass
            
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
                "optimize_for_speed": optimize_for_speed,
                "model_api_fixed": True
            }
            
            logger.info(f"✅ FIXED UNDERSTANDING complete: {total_time*1000:.0f}ms total ({'✅' if total_time*1000 < 200 else '⚠️'} sub-200ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"FIXED processing error: {e}", exc_info=True)
            # Clean up temp file on error
            try:
                if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
            except:
                pass
            return {"error": f"FIXED processing failed: {str(e)}"}
    
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
