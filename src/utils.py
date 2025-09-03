import asyncio
import logging
import psutil
import torch
import platform
import time
from typing import Dict, Any, Optional, List
import subprocess
import json
import os

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        "timestamp": time.time(),
        "platform": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation()
        },
        "cpu": {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
            "used_gb": round(psutil.virtual_memory().used / 1024**3, 2),
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / 1024**3, 2),
            "free_gb": round(psutil.disk_usage('/').free / 1024**3, 2),
            "used_gb": round(psutil.disk_usage('/').used / 1024**3, 2),
            "percent": psutil.disk_usage('/').percent
        },
        "gpu": get_gpu_info(),
        "process": {
            "pid": os.getpid(),
            "cpu_percent": psutil.Process().cpu_percent(),
            "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 2)
        }
    }
    
    return info

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = device_props.total_memory / 1024**3
            
            device_info = {
                "index": i,
                "name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "memory": {
                    "total_gb": round(memory_total, 2),
                    "allocated_gb": round(memory_allocated, 2),
                    "cached_gb": round(memory_cached, 2),
                    "free_gb": round(memory_total - memory_allocated, 2)
                },
                "multiprocessor_count": device_props.multiprocessor_count
            }
            gpu_info["devices"].append(device_info)
    
    return gpu_info

def get_runpod_info() -> Dict[str, Any]:
    """Get RunPod specific information"""
    runpod_info = {}
    
    # RunPod environment variables
    runpod_vars = [
        "RUNPOD_POD_ID",
        "RUNPOD_PUBLIC_IP", 
        "RUNPOD_POD_HOSTNAME",
        "RUNPOD_CPU_COUNT",
        "RUNPOD_MEM_GB",
        "RUNPOD_GPU_COUNT",
        "RUNPOD_VOLUME_PATH",
        "RUNPOD_REALTIME",
        "RUNPOD_AI_API_KEY"
    ]
    
    for var in runpod_vars:
        value = os.getenv(var)
        if value:
            runpod_info[var.lower()] = value
    
    # TCP port mappings
    tcp_ports = {}
    for key, value in os.environ.items():
        if key.startswith("RUNPOD_TCP_PORT_"):
            port_internal = key.split("_")[-1]
            tcp_ports[port_internal] = value
    
    if tcp_ports:
        runpod_info["tcp_ports"] = tcp_ports
    
    return runpod_info

async def check_model_health(model_manager) -> Dict[str, Any]:
    """Check model health and performance"""
    if not model_manager or not model_manager.is_loaded:
        return {
            "status": "unhealthy",
            "reason": "Model not loaded"
        }
    
    try:
        # Simple inference test
        test_start = time.time()
        
        # Create small test audio (silence)
        import numpy as np
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        test_audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        # Test transcription
        result = await model_manager.transcribe_audio(test_audio_bytes)
        
        inference_time = time.time() - test_start
        
        if "error" in result:
            return {
                "status": "unhealthy", 
                "reason": result["error"],
                "inference_time": inference_time
            }
        
        return {
            "status": "healthy",
            "inference_time": inference_time,
            "memory_usage": model_manager._get_memory_usage()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": str(e)
        }

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency statistics"""
    if not latencies:
        return {}
    
    latencies.sort()
    n = len(latencies)
    
    return {
        "min": min(latencies),
        "max": max(latencies),
        "mean": sum(latencies) / n,
        "median": latencies[n // 2],
        "p95": latencies[int(0.95 * n)] if n > 20 else latencies[-1],
        "p99": latencies[int(0.99 * n)] if n > 100 else latencies[-1]
    }

async def cleanup_temp_files(temp_dir: str, max_age_seconds: int = 3600):
    """Clean up temporary files older than max_age"""
    try:
        current_time = time.time()
        cleaned = 0
        
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} temporary files")
            
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")

def validate_audio_format(audio_data: bytes) -> bool:
    """Validate if audio data is in supported format"""
    try:
        # Check for common audio file headers
        if audio_data.startswith(b'RIFF'):  # WAV
            return True
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\\xff\\xfb'):  # MP3
            return True
        elif audio_data.startswith(b'OggS'):  # OGG
            return True
        elif audio_data.startswith(b'fLaC'):  # FLAC
            return True
        
        # Check if it's raw PCM by length
        if len(audio_data) % 2 == 0 and len(audio_data) > 1000:  # Even length, reasonable size
            return True
        
        return False
        
    except Exception:
        return False

class PerformanceMonitor:
    """Monitor performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = []
        self.throughput_times = []
        self.error_count = 0
        self.request_count = 0
    
    def record_latency(self, latency: float):
        """Record request latency"""
        self.latencies.append(latency)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
    
    def record_request(self, processing_time: float):
        """Record request processing time"""
        self.throughput_times.append(processing_time)
        if len(self.throughput_times) > self.window_size:
            self.throughput_times.pop(0)
        self.request_count += 1
    
    def record_error(self):
        """Record error occurrence"""
        self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
        
        if self.latencies:
            stats["latency"] = calculate_latency_stats(self.latencies)
        
        if self.throughput_times:
            avg_processing_time = sum(self.throughput_times) / len(self.throughput_times)
            stats["avg_processing_time"] = avg_processing_time
            stats["requests_per_second"] = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        return stats
    
    def reset(self):
        """Reset all metrics"""
        self.latencies.clear()
        self.throughput_times.clear()
        self.error_count = 0
        self.request_count = 0
