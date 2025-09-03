import logging
import logging.handlers
import os
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    enable_json_logging: bool = False
):
    """Setup logging configuration"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    if enable_json_logging:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Set specific loggers
    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Application loggers
    app_loggers = [
        "main",
        "websocket_handler", 
        "model_loader",
        "audio_processor",
        "utils"
    ]
    
    for logger_name in app_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
    
    logging.info(f"Logging configured: level={log_level}, file={log_file}")

class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self, context_data: dict = None):
        super().__init__()
        self.context_data = context_data or {}
    
    def filter(self, record):
        # Add context data to record
        for key, value in self.context_data.items():
            setattr(record, key, value)
        return True

def get_logger(name: str, context: dict = None) -> logging.Logger:
    """Get logger with optional context"""
    logger = logging.getLogger(name)
    
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger
