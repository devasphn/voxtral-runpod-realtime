import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings
import torch

class Settings(BaseSettings):
    """Application settings - RunPod Compatible"""
    
    # Get current working directory dynamically
    WORK_DIR: str = os.getcwd()
    
    # Model Configuration
    MODEL_NAME: str = "mistralai/Voxtral-Mini-3B-2507"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE: torch.dtype = torch.bfloat16
    
    # Server Configuration  
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    HEALTH_PORT: int = 8080
    
    # WebSocket Configuration
    WS_TRANSCRIBE_PORT: int = 8765
    WS_UNDERSTAND_PORT: int = 8766
    MAX_CONCURRENT_CONNECTIONS: int = 6
    MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    WEBSOCKET_TIMEOUT: int = 60  # seconds
    
    # Audio Processing
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    CHUNK_DURATION_MS: int = 30
    VAD_MODE: int = 3  # Aggressive voice activity detection
    MAX_AUDIO_LENGTH: int = 30 * 60  # 30 minutes in seconds
    
    # Model Generation
    MAX_NEW_TOKENS: int = 500
    TRANSCRIPTION_TEMPERATURE: float = 0.0  # Deterministic
    UNDERSTANDING_TEMPERATURE: float = 0.2
    UNDERSTANDING_TOP_P: float = 0.95
    
    # Performance
    BATCH_SIZE: int = 1
    USE_FLASH_ATTENTION: bool = True
    ENABLE_MEMORY_EFFICIENT_ATTENTION: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Storage - Dynamic paths based on working directory
    @property
    def LOG_FILE(self) -> str:
        return os.path.join(self.WORK_DIR, "logs", "voxtral.log")
    
    @property 
    def MODEL_CACHE_DIR(self) -> str:
        return os.path.join(self.WORK_DIR, "models")
    
    @property
    def TEMP_DIR(self) -> str:
        return os.path.join(self.WORK_DIR, "temp")
    
    # RunPod Integration
    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
    RUNPOD_PUBLIC_IP: Optional[str] = os.getenv("RUNPOD_PUBLIC_IP")
    RUNPOD_TCP_PORT_8765: Optional[str] = os.getenv("RUNPOD_TCP_PORT_8765")
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
        
        print(f"✅ Settings initialized with WORK_DIR: {self.WORK_DIR}")
        print(f"✅ Model cache: {self.MODEL_CACHE_DIR}")
        print(f"✅ Log file: {self.LOG_FILE}")
    
    @property
    def websocket_urls(self) -> Dict[str, str]:
        """Get WebSocket URLs for RunPod"""
        if self.RUNPOD_POD_ID:
            return {
                "transcribe": f"wss://{self.RUNPOD_POD_ID}-{self.WS_TRANSCRIBE_PORT}.proxy.runpod.net/ws",
                "understand": f"wss://{self.RUNPOD_POD_ID}-{self.WS_UNDERSTAND_PORT}.proxy.runpod.net/ws"
            }
        return {
            "transcribe": f"ws://localhost:{self.WS_TRANSCRIBE_PORT}/ws",
            "understand": f"ws://localhost:{self.WS_UNDERSTAND_PORT}/ws"
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dict"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.DEVICE,
            "dtype": str(self.TORCH_DTYPE),
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "transcription_temperature": self.TRANSCRIPTION_TEMPERATURE,
            "understanding_temperature": self.UNDERSTANDING_TEMPERATURE,
            "understanding_top_p": self.UNDERSTANDING_TOP_P
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return {
            "sample_rate": self.AUDIO_SAMPLE_RATE,
            "channels": self.AUDIO_CHANNELS,
            "chunk_duration_ms": self.CHUNK_DURATION_MS,
            "vad_mode": self.VAD_MODE,
            "max_audio_length": self.MAX_AUDIO_LENGTH
        }
