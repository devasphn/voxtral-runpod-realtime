# ULTIMATE REAL-TIME AUDIO PROCESSOR - FIXED WITH PROPER VAD
import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import io
import wave
import json
import collections
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import webrtcvad

logger = logging.getLogger(__name__)

class RealTimeVAD:
    """Real-time Voice Activity Detection with 300ms silence threshold"""
    
    def __init__(self, sample_rate: int = 16000, vad_mode: int = 2, silence_threshold_ms: int = 300):
        self.sample_rate = sample_rate
        self.silence_threshold_ms = silence_threshold_ms
        self.silence_threshold_frames = int((silence_threshold_ms / 1000) * sample_rate / 160)  # 10ms frames
        
        # Initialize WebRTC VAD
        try:
            self.vad = webrtcvad.Vad(vad_mode)
            self.vad_enabled = True
            logger.info(f"✅ Real-time VAD initialized: {silence_threshold_ms}ms threshold")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # VAD state tracking
        self.is_speaking = False
        self.silent_frames = 0
        self.speech_frames = 0
        self.accumulated_audio = bytearray()
        self.last_speech_time = 0
        
        # Statistics
        self.total_frames_processed = 0
        self.speech_segments_detected = 0
    
    def process_frame(self, audio_frame: bytes) -> Dict[str, Any]:
        """Process single audio frame and return VAD decision"""
        try:
            self.total_frames_processed += 1
            current_time = time.time()
            
            # Ensure frame is proper size for WebRTC VAD (160 samples for 16kHz, 10ms)
            if len(audio_frame) != 320:  # 160 samples * 2 bytes
                return {"speech": False, "accumulated": False, "speech_ended": False}
            
            is_speech = False
            if self.vad_enabled and self.vad:
                try:
                    is_speech = self.vad.is_speech(audio_frame, self.sample_rate)
                except:
                    is_speech = False
            
            # Update speech state
            if is_speech:
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.accumulated_audio.clear()
                    logger.debug("🗣️ Speech started")
                
                self.speech_frames += 1
                self.silent_frames = 0
                self.last_speech_time = current_time
                
            else:
                if self.is_speaking:
                    self.silent_frames += 1
                    
                    # Check if silence threshold reached
                    if self.silent_frames >= self.silence_threshold_frames:
                        # Speech ended
                        self.is_speaking = False
                        speech_duration = len(self.accumulated_audio) / (self.sample_rate * 2) * 1000  # ms
                        
                        result = {
                            "speech": False,
                            "speech_ended": True,
                            "accumulated_audio": bytes(self.accumulated_audio),
                            "duration_ms": speech_duration,
                            "speech_frames": self.speech_frames,
                            "speech_ratio": min(1.0, self.speech_frames / max(1, self.total_frames_processed)),
                            "silence_duration_ms": self.silence_threshold_ms
                        }
                        
                        # Reset for next segment
                        self.accumulated_audio.clear()
                        self.speech_frames = 0
                        self.silent_frames = 0
                        self.speech_segments_detected += 1
                        
                        logger.debug(f"🔇 Speech ended: {speech_duration:.0f}ms")
                        return result
            
            # Accumulate audio while speaking
            if self.is_speaking:
                self.accumulated_audio.extend(audio_frame)
            
            return {
                "speech": is_speech,
                "accumulated": self.is_speaking,
                "speech_ended": False,
                "frames_speaking": self.speech_frames if self.is_speaking else 0
            }
            
        except Exception as e:
            logger.error(f"VAD frame processing error: {e}")
            return {"speech": False, "accumulated": False, "speech_ended": False}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get VAD statistics"""
        return {
            "total_frames": self.total_frames_processed,
            "speech_segments": self.speech_segments_detected,
            "is_speaking": self.is_speaking,
            "silence_threshold_ms": self.silence_threshold_ms,
            "vad_enabled": self.vad_enabled,
            "accumulated_duration_ms": len(self.accumulated_audio) / (self.sample_rate * 2) * 1000
        }

class FixedAudioProcessor:
    """ULTIMATE: Real-time audio processor with proper VAD and streaming"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        conversation_manager=None,
        vad_threshold_ms: int = 300,
        real_time_mode: bool = True
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.conversation_manager = conversation_manager
        self.real_time_mode = real_time_mode
        
        # Initialize Real-time VAD
        self.vad = RealTimeVAD(
            sample_rate=sample_rate,
            vad_mode=2,  # Balanced mode
            silence_threshold_ms=vad_threshold_ms
        )
        
        # Per-connection VAD instances
        self.connection_vads: Dict[str, RealTimeVAD] = {}
        
        # Statistics and monitoring
        self.total_audio_processed = 0
        self.successful_transcriptions = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # ThreadPool for async processing
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RealTimeAudio")
        
        logger.info(f"✅ ULTIMATE Real-time Audio Processor initialized")
        logger.info(f"   VAD threshold: {vad_threshold_ms}ms")
        logger.info(f"   Sample rate: {sample_rate}Hz")
        logger.info(f"   Real-time mode: {real_time_mode}")
    
    def get_connection_vad(self, websocket) -> RealTimeVAD:
        """Get or create VAD instance for connection"""
        conn_id = str(id(websocket))
        if conn_id not in self.connection_vads:
            self.connection_vads[conn_id] = RealTimeVAD(
                sample_rate=self.sample_rate,
                vad_mode=2,
                silence_threshold_ms=300
            )
        return self.connection_vads[conn_id]
    
    async def process_realtime_audio(self, audio_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """ULTIMATE: Process real-time audio with VAD"""
        try:
            if not audio_data or len(audio_data) < 320:  # Minimum frame size
                return None
            
            vad_instance = self.get_connection_vad(websocket) if websocket else self.vad
            
            # Process in 10ms frames (160 samples = 320 bytes for 16kHz)
            frame_size = 320
            results = []
            
            for i in range(0, len(audio_data), frame_size):
                frame = audio_data[i:i + frame_size]
                if len(frame) == frame_size:
                    vad_result = vad_instance.process_frame(frame)
                    
                    if vad_result.get("speech_ended"):
                        # Speech segment complete - return for processing
                        self.total_audio_processed += 1
                        return vad_result
            
            return None
            
        except Exception as e:
            logger.error(f"Real-time audio processing error: {e}")
            return None
    
    def is_vad_active(self) -> bool:
        """Check if any VAD instance is currently detecting speech"""
        return any(vad.is_speaking for vad in self.connection_vads.values()) or self.vad.is_speaking
    
    def get_vad_stats(self) -> Dict[str, Any]:
        """Get comprehensive VAD statistics"""
        total_segments = sum(vad.speech_segments_detected for vad in self.connection_vads.values())
        total_frames = sum(vad.total_frames_processed for vad in self.connection_vads.values())
        
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        return {
            "total_audio_processed": self.total_audio_processed,
            "successful_transcriptions": self.successful_transcriptions,
            "active_connections": len(self.connection_vads),
            "total_speech_segments": total_segments,
            "total_frames_processed": total_frames,
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "vad_instances_speaking": sum(1 for vad in self.connection_vads.values() if vad.is_speaking),
            "real_time_mode": self.real_time_mode
        }
    
    def cleanup_old_vad_data(self):
        """Clean up old VAD instances for disconnected connections"""
        # This would be called by background cleanup
        # For now, we'll rely on WebSocket disconnect cleanup
        pass
    
    def cleanup_connection(self, websocket):
        """Clean up VAD data for specific connection"""
        conn_id = str(id(websocket))
        if conn_id in self.connection_vads:
            del self.connection_vads[conn_id]
            logger.debug(f"Cleaned up VAD for connection {conn_id}")
    
    async def cleanup(self):
        """ULTIMATE: Cleanup resources"""
        logger.info("🧹 Cleaning up ULTIMATE audio processor...")
        
        # Clean up all VAD instances
        self.connection_vads.clear()
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Executor shutdown error: {e}")
        
        logger.info("✅ ULTIMATE audio processor cleaned up")
