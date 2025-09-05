# UNDERSTANDING-ONLY SETTINGS - FIXED CONFIGURATION
import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """UNDERSTANDING-ONLY Application settings - Optimized for Conversational AI"""
    
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
    WS_UNDERSTAND_PORT: int = 8766  # Single understanding endpoint
    MAX_CONCURRENT_CONNECTIONS: int = 10  # Increased for understanding-only
    MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    WEBSOCKET_TIMEOUT: int = 60  # seconds
    
    # UNDERSTANDING-ONLY: Audio Processing with Gap Detection
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    GAP_THRESHOLD_MS: int = 300  # 0.3 second gap detection
    MIN_SPEECH_DURATION_MS: int = 500  # Minimum 0.5 seconds
    MAX_SPEECH_DURATION_MS: int = 30000  # Maximum 30 seconds
    
    # UNDERSTANDING-ONLY: Model Generation Settings
    TARGET_RESPONSE_MS: int = 200  # Sub-200ms response target
    UNDERSTANDING_MAX_TOKENS: int = 200  # Reduced for speed
    UNDERSTANDING_TEMPERATURE: float = 0.3  # Balanced creativity
    UNDERSTANDING_TOP_P: float = 0.9  # Focused responses
    
    # Performance Optimization - UNDERSTANDING-ONLY
    BATCH_SIZE: int = 1
    USE_FLASH_ATTENTION: bool = True
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
    
    # RunPod Integration - UNDERSTANDING-ONLY
    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
    RUNPOD_PUBLIC_IP: Optional[str] = os.getenv("RUNPOD_PUBLIC_IP")
    RUNPOD_TCP_PORT_8766: Optional[str] = os.getenv("RUNPOD_TCP_PORT_8766")
    
    # Security
    API_KEY: Optional[str] = os.getenv("API_KEY")
    CORS_ORIGINS: List[str] = ["*"]
    
    # UNDERSTANDING-ONLY: Conversation Management
    MAX_CONVERSATION_TURNS: int = 30
    CONTEXT_WINDOW_MINUTES: int = 15
    ENABLE_CONTEXT_MEMORY: bool = True
    
    # Monitoring - UNDERSTANDING-ONLY
    ENABLE_METRICS: bool = True
    METRICS_INTERVAL: int = 30  # seconds
    HEALTH_CHECK_TIMEOUT: int = 5  # seconds
    TRACK_RESPONSE_TIMES: bool = True
    
    # UNDERSTANDING-ONLY: Feature Flags
    UNDERSTANDING_ONLY: bool = True  # Always true for this mode
    TRANSCRIPTION_DISABLED: bool = True  # Transcription completely disabled
    GAP_DETECTION_ENABLED: bool = True  # 0.3s gap detection enabled
    CONTEXT_AWARE_RESPONSES: bool = True  # Context-aware conversation
    
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
        """Get WebSocket URLs for UNDERSTANDING-ONLY RunPod deployment"""
        if self.RUNPOD_POD_ID:
            return {
                "understand": f"wss://{self.RUNPOD_POD_ID}-{self.WS_UNDERSTAND_PORT}.proxy.runpod.net/ws"
            }
        return {
            "understand": f"ws://localhost:{self.WS_UNDERSTAND_PORT}/ws"
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get UNDERSTANDING-ONLY model configuration"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.DEVICE,
            "dtype": str(self.TORCH_DTYPE),
            "understanding_max_tokens": self.UNDERSTANDING_MAX_TOKENS,
            "understanding_temperature": self.UNDERSTANDING_TEMPERATURE,
            "understanding_top_p": self.UNDERSTANDING_TOP_P,
            "target_response_ms": self.TARGET_RESPONSE_MS,
            "optimize_for_speed": self.OPTIMIZE_FOR_SPEED,
            "understanding_only": self.UNDERSTANDING_ONLY
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get UNDERSTANDING-ONLY audio processing configuration"""
        return {
            "sample_rate": self.AUDIO_SAMPLE_RATE,
            "channels": self.AUDIO_CHANNELS,
            "gap_threshold_ms": self.GAP_THRESHOLD_MS,
            "min_speech_duration_ms": self.MIN_SPEECH_DURATION_MS,
            "max_speech_duration_ms": self.MAX_SPEECH_DURATION_MS,
            "gap_detection_enabled": self.GAP_DETECTION_ENABLED
        }
    
    def get_conversation_config(self) -> Dict[str, Any]:
        """Get conversation management configuration"""
        return {
            "max_turns": self.MAX_CONVERSATION_TURNS,
            "context_window_minutes": self.CONTEXT_WINDOW_MINUTES,
            "enable_context_memory": self.ENABLE_CONTEXT_MEMORY,
            "context_aware_responses": self.CONTEXT_AWARE_RESPONSES
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        return {
            "target_response_ms": self.TARGET_RESPONSE_MS,
            "batch_size": self.BATCH_SIZE,
            "use_flash_attention": self.USE_FLASH_ATTENTION,
            "enable_memory_efficient_attention": self.ENABLE_MEMORY_EFFICIENT_ATTENTION,
            "optimize_for_speed": self.OPTIMIZE_FOR_SPEED,
            "track_response_times": self.TRACK_RESPONSE_TIMES
        }
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get UNDERSTANDING-ONLY feature flags"""
        return {
            "understanding_only": self.UNDERSTANDING_ONLY,
            "transcription_disabled": self.TRANSCRIPTION_DISABLED,
            "gap_detection_enabled": self.GAP_DETECTION_ENABLED,
            "context_aware_responses": self.CONTEXT_AWARE_RESPONSES,
            "enable_metrics": self.ENABLE_METRICS
        }
