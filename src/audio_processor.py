# UNDERSTANDING-ONLY AUDIO PROCESSOR - FIXED IMPORTS
import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import io
import wave
import json
import collections
import threading
from concurrent.futures import ThreadPoolExecutor
import time

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    logging.warning("WebRTC VAD not available, using basic silence detection")

logger = logging.getLogger(__name__)

class UnderstandingAudioProcessor:
    """UNDERSTANDING-ONLY: Audio processor with 0.3-second gap detection for conversational AI"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        gap_threshold_ms: int = 300,
        conversation_manager=None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.gap_threshold_ms = gap_threshold_ms
        self.conversation_manager = conversation_manager
        
        # UNDERSTANDING-ONLY: Audio buffering for gap detection
        self.audio_segments = {}  # Per-connection audio segments
        self.speech_buffers = {}  # Per-connection speech buffers
        self.silence_counters = {}  # Per-connection silence tracking
        self.last_audio_time = {}  # Per-connection timing
        
        # UNDERSTANDING-ONLY: Gap detection thresholds
        self.min_speech_duration_ms = 500  # Minimum 0.5 seconds
        self.max_speech_duration_ms = 30000  # Maximum 30 seconds
        self.gap_threshold_samples = int(sample_rate * (gap_threshold_ms / 1000.0))
        
        # UNDERSTANDING-ONLY: WebRTC VAD for accurate gap detection
        self.vad = None
        self.vad_enabled = False
        if VAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(1)  # Moderate aggressiveness for conversation
                self.vad_enabled = True
                logger.info("✅ UNDERSTANDING-ONLY WebRTC VAD initialized (mode 1)")
            except Exception as e:
                logger.warning(f"WebRTC VAD initialization failed: {e}")
        
        # Statistics
        self.segments_processed = 0
        self.speech_segments_detected = 0
        self.gaps_detected = 0
        self.processing_times = collections.deque(maxlen=50)
        
        # UNDERSTANDING-ONLY: ThreadPoolExecutor for processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="UnderstandingAudio")
        
        logger.info(f"✅ UNDERSTANDING-ONLY AudioProcessor: {sample_rate}Hz, gap: {gap_threshold_ms}ms, VAD: {self.vad_enabled}")
    
    async def process_audio_understanding(self, audio_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """UNDERSTANDING-ONLY: Process audio with gap detection for conversational AI"""
        start_time = time.time()
        
        try:
            # Get connection ID
            conn_id = id(websocket) if websocket else "default"
            
            # Initialize connection tracking
            if conn_id not in self.audio_segments:
                self.audio_segments[conn_id] = bytearray()
                self.speech_buffers[conn_id] = []
                self.silence_counters[conn_id] = 0
                self.last_audio_time[conn_id] = time.time()
            
            # Validate input
            if not audio_data or len(audio_data) < 100:
                logger.debug("Insufficient audio data for understanding")
                return None
            
            # Convert WebM to PCM if needed
            pcm_data = await self._convert_to_pcm(audio_data)
            if not pcm_data:
                return None
            
            # Add to connection's audio buffer
            self.audio_segments[conn_id].extend(pcm_data)
            self.last_audio_time[conn_id] = time.time()
            
            # Analyze current segment for speech
            segment_duration_ms = len(pcm_data) / 2 / self.sample_rate * 1000
            speech_detected = self._detect_speech_in_segment(pcm_data)
            
            # Update speech buffer and silence counter
            if speech_detected:
                self.speech_buffers[conn_id].append(time.time())
                self.silence_counters[conn_id] = 0
                logger.debug(f"UNDERSTANDING-ONLY speech detected: {segment_duration_ms:.0f}ms")
            else:
                self.silence_counters[conn_id] += 1
                logger.debug(f"UNDERSTANDING-ONLY silence: counter={self.silence_counters[conn_id]}")
            
            # Calculate durations
            total_audio_ms = len(self.audio_segments[conn_id]) / 2 / self.sample_rate * 1000
            silence_duration_ms = self.silence_counters[conn_id] * segment_duration_ms
            
            # Check if we have enough speech data
            if total_audio_ms < self.min_speech_duration_ms:
                return {
                    "audio_received": True,
                    "speech_complete": False,
                    "segment_duration_ms": segment_duration_ms,
                    "total_duration_ms": total_audio_ms,
                    "silence_duration_ms": silence_duration_ms,
                    "gap_will_trigger_at_ms": self.gap_threshold_ms,
                    "speech_detected": speech_detected,
                    "understanding_only": True
                }
            
            # Check for gap detection (0.3 second silence)
            gap_detected = (
                silence_duration_ms >= self.gap_threshold_ms or 
                total_audio_ms >= self.max_speech_duration_ms
            )
            
            if gap_detected:
                logger.info(f"🎯 UNDERSTANDING-ONLY gap detected: {silence_duration_ms:.0f}ms silence, {total_audio_ms:.0f}ms total")
                
                # Process complete speech segment
                result = await self._process_complete_speech_segment(conn_id)
                
                # Reset buffers for next segment
                self._reset_connection_buffers(conn_id)
                
                # Record stats
                self.gaps_detected += 1
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                return result
            else:
                # Return intermediate feedback
                return {
                    "audio_received": True,
                    "speech_complete": False,
                    "segment_duration_ms": segment_duration_ms,
                    "total_duration_ms": total_audio_ms,
                    "silence_duration_ms": silence_duration_ms,
                    "gap_will_trigger_at_ms": self.gap_threshold_ms,
                    "remaining_to_gap_ms": max(0, self.gap_threshold_ms - silence_duration_ms),
                    "speech_detected": speech_detected,
                    "understanding_only": True
                }
                
        except Exception as e:
            logger.error(f"UNDERSTANDING-ONLY audio processing error: {e}")
            return {"error": f"UNDERSTANDING-ONLY processing failed: {str(e)}"}
    
    async def _convert_to_pcm(self, audio_data: bytes) -> Optional[bytes]:
        """Convert audio data to PCM format"""
        try:
            # Check if already PCM/WAV
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
                # Extract PCM data from WAV
                try:
                    wav_io = io.BytesIO(audio_data)
                    with wave.open(wav_io, 'rb') as wav_file:
                        return wav_file.readframes(wav_file.getnframes())
                except Exception as e:
                    logger.debug(f"WAV parsing failed: {e}")
            
            # For WebM or other formats, assume raw PCM for now
            # In a production system, you'd use ffmpeg for conversion
            if len(audio_data) % 2 == 1:
                audio_data = audio_data[:-1]  # Ensure even length
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None
    
    def _detect_speech_in_segment(self, pcm_data: bytes) -> bool:
        """Detect speech in audio segment using VAD or energy analysis"""
        try:
            if len(pcm_data) < 320:  # Less than 20ms at 16kHz
                return False
            
            # Method 1: WebRTC VAD (most accurate)
            if self.vad_enabled and self.vad:
                try:
                    # Process in 10ms frames (160 samples at 16kHz)
                    frame_size = 160 * 2  # 160 samples * 2 bytes
                    speech_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(pcm_data) - frame_size, frame_size):
                        frame = pcm_data[i:i + frame_size]
                        if len(frame) == frame_size:
                            try:
                                if self.vad.is_speech(frame, self.sample_rate):
                                    speech_frames += 1
                            except:
                                pass
                            total_frames += 1
                    
                    if total_frames > 0:
                        speech_ratio = speech_frames / total_frames
                        return speech_ratio > 0.3  # 30% speech threshold
                        
                except Exception as e:
                    logger.debug(f"VAD processing error: {e}")
            
            # Method 2: Energy-based detection (fallback)
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            if len(audio_array) == 0:
                return False
            
            # RMS energy
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            energy_threshold = 200.0  # Adjust based on your audio setup
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            
            # Simple heuristic: energy above threshold and reasonable ZCR
            has_energy = rms_energy > energy_threshold
            has_speech_zcr = 0.01 <= zcr_normalized <= 0.3
            
            return has_energy and has_speech_zcr
            
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return False
    
    async def _process_complete_speech_segment(self, conn_id: str) -> Dict[str, Any]:
        """Process complete speech segment for understanding"""
        try:
            if conn_id not in self.audio_segments or len(self.audio_segments[conn_id]) == 0:
                return {"error": "No audio data to process"}
            
            # Get the complete audio segment
            complete_audio = bytes(self.audio_segments[conn_id])
            
            # Create WAV format for understanding processing
            wav_data = self._create_wav_from_pcm(complete_audio)
            
            # Calculate metrics
            duration_ms = len(complete_audio) / 2 / self.sample_rate * 1000
            speech_quality = self._analyze_speech_quality(complete_audio)
            
            # Update statistics
            self.segments_processed += 1
            if speech_quality > 0.3:
                self.speech_segments_detected += 1
            
            logger.info(f"✅ UNDERSTANDING-ONLY segment complete: {duration_ms:.0f}ms, quality: {speech_quality:.3f}")
            
            return {
                "speech_complete": True,
                "audio_data": wav_data,
                "duration_ms": duration_ms,
                "speech_quality": speech_quality,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "gap_detected": True,
                "understanding_only": True,
                "processed_at": time.time()
            }
            
        except Exception as e:
            logger.error(f"Complete speech segment processing error: {e}")
            return {"error": f"Speech segment processing failed: {str(e)}"}
    
    def _create_wav_from_pcm(self, pcm_data: bytes) -> bytes:
        """Create WAV file from PCM data"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            return wav_io.getvalue()
            
        except Exception as e:
            logger.error(f"WAV creation error: {e}")
            raise RuntimeError(f"WAV creation failed: {e}")
    
    def _analyze_speech_quality(self, pcm_data: bytes) -> float:
        """Analyze speech quality for understanding processing"""
        try:
            if len(pcm_data) == 0:
                return 0.0
            
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Energy analysis
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            energy_score = min(1.0, rms_energy / 1000.0)
            
            # Dynamic range analysis
            max_amplitude = np.max(np.abs(audio_array))
            dynamic_score = min(1.0, max_amplitude / 10000.0) if max_amplitude > 0 else 0.0
            
            # Zero crossing rate analysis
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            zcr_score = 1.0 if 0.01 <= zcr_normalized <= 0.2 else 0.5
            
            # Combined quality score
            quality = (energy_score * 0.5 + dynamic_score * 0.3 + zcr_score * 0.2)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Speech quality analysis error: {e}")
            return 0.5  # Default moderate quality
    
    def _reset_connection_buffers(self, conn_id: str):
        """Reset buffers for a connection after processing"""
        if conn_id in self.audio_segments:
            self.audio_segments[conn_id].clear()
        if conn_id in self.speech_buffers:
            self.speech_buffers[conn_id].clear()
        if conn_id in self.silence_counters:
            self.silence_counters[conn_id] = 0
    
    def cleanup_connection(self, websocket):
        """Cleanup connection data when WebSocket disconnects"""
        conn_id = id(websocket)
        
        # Clean up all connection data
        self.audio_segments.pop(conn_id, None)
        self.speech_buffers.pop(conn_id, None)
        self.silence_counters.pop(conn_id, None)
        self.last_audio_time.pop(conn_id, None)
        
        logger.info(f"🧹 UNDERSTANDING-ONLY audio cleanup for connection: {conn_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        speech_detection_rate = (
            self.speech_segments_detected / max(self.segments_processed, 1)
        )
        
        return {
            "understanding_only": True,
            "gap_threshold_ms": self.gap_threshold_ms,
            "segments_processed": self.segments_processed,
            "speech_segments_detected": self.speech_segments_detected,
            "gaps_detected": self.gaps_detected,
            "speech_detection_rate": round(speech_detection_rate, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.audio_segments),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_ms": self.max_speech_duration_ms
        }
    
    def reset(self):
        """Reset processor state"""
        self.audio_segments.clear()
        self.speech_buffers.clear()
        self.silence_counters.clear()
        self.last_audio_time.clear()
        
        self.segments_processed = 0
        self.speech_segments_detected = 0
        self.gaps_detected = 0
        self.processing_times.clear()
        
        logger.info("✅ UNDERSTANDING-ONLY audio processor reset")
    
    async def cleanup(self):
        """UNDERSTANDING-ONLY: Enhanced cleanup"""
        logger.info("🧹 Starting UNDERSTANDING-ONLY audio processor cleanup...")
        
        # Reset all buffers
        self.reset()
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"UNDERSTANDING-ONLY executor shutdown error: {e}")
        
        logger.info("✅ UNDERSTANDING-ONLY audio processor fully cleaned up")

# Backward compatibility alias
AudioProcessor = UnderstandingAudioProcessor
