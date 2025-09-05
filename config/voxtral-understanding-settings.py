import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """UNDERSTANDING-ONLY Application settings - RunPod Compatible"""
    
    # Get current working directory dynamically
    WORK_DIR: str = os.getcwd()
    
    # Model Configuration - UNDERSTANDING-ONLY
    MODEL_NAME: str = "mistralai/Voxtral-Mini-3B-2507"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16
    
    # Server Configuration - UNDERSTANDING-ONLY
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    HEALTH_PORT: int = 8005
    
    # WebSocket Configuration - UNDERSTANDING-ONLY
    WS_UNDERSTAND_PORT: int = 8766  # Only understanding port needed
    MAX_CONCURRENT_CONNECTIONS: int = 10  # Increased for understanding-only
    MAX_MESSAGE_SIZE: int = 20 * 1024 * 1024  # 20MB for longer audio segments
    WEBSOCKET_TIMEOUT: int = 120  # Longer timeout for understanding processing
    
    # Audio Processing - UNDERSTANDING-ONLY with Gap Detection
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    GAP_THRESHOLD_MS: int = 300  # 0.3 second gap detection
    MIN_SPEECH_DURATION_MS: int = 500  # Minimum 0.5 seconds
    MAX_SPEECH_DURATION_MS: int = 30000  # Maximum 30 seconds
    VAD_MODE: int = 1  # Moderate aggressiveness for gap detection
    
    # Model Generation - UNDERSTANDING-ONLY
    UNDERSTANDING_TEMPERATURE: float = 0.3  # Creative but controlled
    UNDERSTANDING_TOP_P: float = 0.9  # Focused generation
    MAX_NEW_TOKENS: int = 200  # Shorter for faster response
    USE_CACHE: bool = True  # Enable caching for speed
    
    # Performance Optimization - Sub-200ms Target
    TARGET_RESPONSE_MS: int = 200  # Sub-200ms response target
    ENABLE_FLASH_ATTENTION: bool = True
    ENABLE_MEMORY_EFFICIENT_ATTENTION: bool = True
    OPTIMIZE_FOR_SPEED: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Storage - Dynamic paths based on working directory
    @property
    def LOG_FILE(self) -> str:
        return os.path.join(self.WORK_DIR, "logs", "voxtral-understanding.log")
    
    @property 
    def MODEL_CACHE_DIR(self) -> str:
        return os.path.join(self.WORK_DIR, "models")
    
    @property
    def TEMP_DIR(self) -> str:
        return os.path.join(self.WORK_DIR, "temp")
    
    # RunPod Integration
    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
    RUNPOD_PUBLIC_IP: Optional[str] = os.getenv("RUNPOD_PUBLIC_IP")
    RUNPOD_TCP_PORT_8766: Optional[str] = os.getenv("RUNPOD_TCP_PORT_8766")
    
    # Security
    API_KEY: Optional[str] = os.getenv("API_KEY")
    CORS_ORIGINS: List[str] = ["*"]
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_TIMEOUT: int = 5  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories dynamically
        os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
        
        print(f"✅ UNDERSTANDING-ONLY Settings initialized with WORK_DIR: {self.WORK_DIR}")
        print(f"✅ Model cache: {self.MODEL_CACHE_DIR}")
        print(f"✅ Log file: {self.LOG_FILE}")
        print(f"✅ Gap detection: {self.GAP_THRESHOLD_MS}ms")
        print(f"✅ Target response: {self.TARGET_RESPONSE_MS}ms")
    
    @property
    def websocket_urls(self) -> Dict[str, str]:
        """Get WebSocket URLs for RunPod - UNDERSTANDING-ONLY"""
        if self.RUNPOD_POD_ID:
            return {
                "understand": f"wss://{self.RUNPOD_POD_ID}-{self.WS_UNDERSTAND_PORT}.proxy.runpod.net/ws"
            }
        return {
            "understand": f"ws://localhost:{self.WS_UNDERSTAND_PORT}/ws"
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dict - UNDERSTANDING-ONLY"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.DEVICE,
            "dtype": str(self.TORCH_DTYPE),
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "understanding_temperature": self.UNDERSTANDING_TEMPERATURE,
            "understanding_top_p": self.UNDERSTANDING_TOP_P,
            "use_cache": self.USE_CACHE,
            "optimize_for_speed": self.OPTIMIZE_FOR_SPEED,
            "target_response_ms": self.TARGET_RESPONSE_MS,
            "understanding_only": True
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration - UNDERSTANDING-ONLY"""
        return {
            "sample_rate": self.AUDIO_SAMPLE_RATE,
            "channels": self.AUDIO_CHANNELS,
            "gap_threshold_ms": self.GAP_THRESHOLD_MS,
            "min_speech_duration_ms": self.MIN_SPEECH_DURATION_MS,
            "max_speech_duration_ms": self.MAX_SPEECH_DURATION_MS,
            "vad_mode": self.VAD_MODE,
            "understanding_only": True,
            "gap_detection": True
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "target_response_ms": self.TARGET_RESPONSE_MS,
            "enable_flash_attention": self.ENABLE_FLASH_ATTENTION,
            "enable_memory_efficient_attention": self.ENABLE_MEMORY_EFFICIENT_ATTENTION,
            "optimize_for_speed": self.OPTIMIZE_FOR_SPEED,
            "max_concurrent_connections": self.MAX_CONCURRENT_CONNECTIONS,
            "websocket_timeout": self.WEBSOCKET_TIMEOUT
        }
