# UNDERSTANDING-ONLY AUDIO PROCESSOR - 0.3 SECOND GAP DETECTION
import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import io
import wave
import json
import collections
import ffmpeg
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import webrtcvad
import time

logger = logging.getLogger(__name__)

class UnderstandingAudioProcessor:
    """UNDERSTANDING-ONLY: Audio processor with 0.3-second gap detection for natural speech boundaries"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        gap_threshold_ms: int = 300,  # 0.3 second gap detection
        conversation_manager=None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.gap_threshold_ms = gap_threshold_ms
        self.conversation_manager = conversation_manager
        
        # UNDERSTANDING-ONLY: Audio processing pipeline
        self.audio_buffer = bytearray()
        self.speech_segments = []
        self.last_speech_time = 0
        self.current_segment_start = 0
        
        # Gap detection settings
        self.gap_threshold_samples = int((gap_threshold_ms / 1000.0) * sample_rate * 2)  # 2 bytes per sample
        self.min_speech_duration_ms = 500  # Minimum 0.5 seconds for processing
        self.max_speech_duration_ms = 30000  # Maximum 30 seconds per segment
        
        # Enhanced Voice Activity Detection for gap detection
        try:
            self.vad = webrtcvad.Vad(1)  # Moderate aggressiveness for gap detection
            self.vad_enabled = True
            logger.info("✅ Enhanced WebRTC VAD initialized for gap detection (mode 1)")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # Statistics
        self.segments_processed = 0
        self.gaps_detected = 0
        self.total_audio_length = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # ThreadPoolExecutor for audio processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="UnderstandProc")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        # Speech quality thresholds for understanding
        self.speech_threshold = 0.4  # Higher threshold for better understanding quality
        self.silence_threshold = 0.1  # Threshold for detecting silence/gaps
        
        logger.info(f"✅ UNDERSTANDING-ONLY AudioProcessor: {sample_rate}Hz, {channels}ch, gap={gap_threshold_ms}ms, VAD: {self.vad_enabled}")
        logger.info(f"   Gap threshold: {self.gap_threshold_samples} bytes ({gap_threshold_ms}ms)")
        logger.info(f"   Min speech: {self.min_speech_duration_ms}ms, Max speech: {self.max_speech_duration_ms}ms")
    
    async def process_audio_understanding(self, audio_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """UNDERSTANDING-ONLY: Process audio with 0.3-second gap detection"""
        start_time = time.time()
        
        try:
            # Input validation
            if not audio_data or len(audio_data) < 100:
                logger.debug("Insufficient audio data for understanding")
                return None
            
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            current_time = time.time()
            
            # Convert to PCM for analysis
            pcm_data = self._extract_pcm_from_audio(audio_data)
            if not pcm_data:
                return {"audio_received": True, "duration_ms": 0}
            
            # Analyze speech activity
            speech_activity = self._analyze_speech_activity(pcm_data)
            
            if speech_activity["has_speech"]:
                self.last_speech_time = current_time
                if self.current_segment_start == 0:
                    self.current_segment_start = current_time
            
            # Calculate current segment duration
            if self.current_segment_start > 0:
                segment_duration_ms = (current_time - self.current_segment_start) * 1000
            else:
                segment_duration_ms = 0
            
            # Check for gap detection (0.3 seconds of silence after speech)
            gap_detected = False
            silence_duration_ms = 0
            
            if self.last_speech_time > 0:
                silence_duration_ms = (current_time - self.last_speech_time) * 1000
                gap_detected = silence_duration_ms >= self.gap_threshold_ms
            
            # Process complete speech segment
            if gap_detected and segment_duration_ms >= self.min_speech_duration_ms:
                logger.info(f"🎯 GAP DETECTED: {silence_duration_ms:.0f}ms silence, segment: {segment_duration_ms:.0f}ms")
                
                # Extract the complete speech segment
                segment_data = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Reset tracking
                self.current_segment_start = 0
                self.last_speech_time = 0
                self.gaps_detected += 1
                self.segments_processed += 1
                
                # Process the complete segment
                result = self._process_complete_segment(segment_data, segment_duration_ms)
                
                # Record processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                result["speech_complete"] = True
                result["gap_detected"] = True
                result["silence_duration_ms"] = silence_duration_ms
                
                return result
            
            # Handle maximum segment length
            elif segment_duration_ms >= self.max_speech_duration_ms:
                logger.info(f"📏 MAX SEGMENT: Processing {segment_duration_ms:.0f}ms segment")
                
                # Process the long segment
                segment_data = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Reset tracking
                self.current_segment_start = current_time  # Continue with new segment
                self.segments_processed += 1
                
                result = self._process_complete_segment(segment_data, segment_duration_ms)
                result["speech_complete"] = True
                result["max_length_reached"] = True
                
                return result
            
            # Return intermediate feedback
            return {
                "audio_received": True,
                "duration_ms": len(self.audio_buffer) / (self.sample_rate * 2) * 1000,
                "speech_ratio": speech_activity.get("speech_ratio", 0),
                "segment_duration_ms": segment_duration_ms,
                "silence_duration_ms": silence_duration_ms,
                "gap_will_trigger_at_ms": self.gap_threshold_ms,
                "understanding_only": True
            }
            
        except Exception as e:
            logger.error(f"UNDERSTANDING-ONLY audio processing error: {e}", exc_info=True)
            return {"error": f"UNDERSTANDING-ONLY processing failed: {str(e)}"}
    
    def _extract_pcm_from_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Extract PCM data from audio for analysis"""
        try:
            # Check if it's already a WAV file
            if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]:
                # Extract PCM data from WAV
                try:
                    with io.BytesIO(audio_data) as wav_io:
                        with wave.open(wav_io, 'rb') as wav_file:
                            return wav_file.readframes(wav_file.getnframes())
                except Exception as e:
                    logger.debug(f"WAV extraction failed: {e}")
            
            # Assume raw PCM data
            if len(audio_data) % 2 == 1:
                audio_data = audio_data[:-1]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"PCM extraction failed: {e}")
            return None
    
    def _analyze_speech_activity(self, pcm_data: bytes) -> Dict[str, Any]:
        """Analyze speech activity for gap detection"""
        try:
            if not pcm_data or len(pcm_data) < 320:  # Minimum for 20ms at 16kHz
                return {"has_speech": False, "speech_ratio": 0.0}
            
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Method 1: Enhanced WebRTC VAD for gap detection
            speech_ratio = 0.0
            if self.vad_enabled and self.vad:
                try:
                    frame_size = 320  # 20ms at 16kHz
                    speech_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(audio_array) - frame_size, frame_size):
                        frame = audio_array[i:i + frame_size]
                        if len(frame) == frame_size:
                            frame_bytes = frame.astype(np.int16).tobytes()
                            try:
                                if self.vad.is_speech(frame_bytes, self.sample_rate):
                                    speech_frames += 1
                            except:
                                pass
                            total_frames += 1
                    
                    if total_frames > 0:
                        speech_ratio = speech_frames / total_frames
                        
                except Exception as e:
                    logger.debug(f"VAD analysis error: {e}")
            
            # Method 2: Energy-based analysis for gap detection
            if speech_ratio == 0.0:  # Fallback if VAD fails
                audio_float = audio_array.astype(np.float64)
                rms_energy = np.sqrt(np.mean(audio_float ** 2))
                energy_threshold = 200.0  # Adjusted for gap detection
                speech_ratio = min(1.0, max(0.0, (rms_energy - energy_threshold) / energy_threshold))
            
            has_speech = speech_ratio > self.silence_threshold
            
            logger.debug(f"Speech activity: ratio={speech_ratio:.3f}, has_speech={has_speech}")
            
            return {
                "has_speech": has_speech,
                "speech_ratio": speech_ratio,
                "energy_analysis": True
            }
            
        except Exception as e:
            logger.error(f"Speech activity analysis failed: {e}")
            return {"has_speech": False, "speech_ratio": 0.0}
    
    def _process_complete_segment(self, segment_data: bytes, duration_ms: float) -> Dict[str, Any]:
        """Process a complete speech segment for understanding"""
        try:
            # Create WAV file for the segment
            wav_bytes = self._pcm_to_wav_optimized(segment_data)
            
            # Calculate final speech quality
            speech_ratio = self._estimate_final_speech_quality(segment_data)
            
            # Update statistics
            self.total_audio_length += duration_ms
            
            logger.info(f"🎤 UNDERSTANDING segment: {duration_ms:.0f}ms, quality: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "processed_at": time.time(),
                "understanding_only": True,
                "gap_detection": True,
                "segment_complete": True
            }
            
        except Exception as e:
            logger.error(f"Complete segment processing error: {e}")
            return {"error": f"Segment processing failed: {str(e)}"}
    
    def _estimate_final_speech_quality(self, pcm_data: bytes) -> float:
        """Estimate final speech quality for the complete segment"""
        try:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Combined quality metrics
            audio_float = audio_array.astype(np.float64)
            
            # RMS energy
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            energy_score = min(1.0, max(0.0, (rms_energy - 100) / 400))
            
            # Dynamic range
            dynamic_range = np.max(audio_array) - np.min(audio_array)
            range_score = min(1.0, dynamic_range / 32768.0)
            
            # Zero crossing rate (for speech characteristics)
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            zcr_score = 1.0 if 0.01 <= zcr_normalized <= 0.15 else 0.5
            
            # Combined score
            final_score = (energy_score * 0.5 + range_score * 0.3 + zcr_score * 0.2)
            
            logger.debug(f"Speech quality: energy={energy_score:.3f}, range={range_score:.3f}, "
                        f"zcr={zcr_score:.3f}, final={final_score:.3f}")
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Speech quality estimation failed: {e}")
            return 0.5  # Default moderate quality
    
    def _pcm_to_wav_optimized(self, pcm_data: bytes) -> bytes:
        """Convert PCM to WAV format optimized for understanding"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            logger.debug(f"WAV conversion: {len(pcm_data)} PCM → {len(wav_bytes)} WAV bytes")
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"WAV conversion error: {e}")
            raise RuntimeError(f"WAV conversion failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        gap_detection_rate = (
            self.gaps_detected / max(self.segments_processed, 1)
        )
        
        return {
            "segments_processed": self.segments_processed,
            "gaps_detected": self.gaps_detected,
            "gap_detection_rate": round(gap_detection_rate, 3),
            "total_audio_length_ms": self.total_audio_length,
            "buffer_size": len(self.audio_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "gap_threshold_ms": self.gap_threshold_ms,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_ms": self.max_speech_duration_ms,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.active_connections),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "understanding_only": True,
            "speech_threshold": self.speech_threshold,
            "silence_threshold": self.silence_threshold,
            "current_segment_duration_ms": (
                (time.time() - self.current_segment_start) * 1000 
                if self.current_segment_start > 0 else 0
            )
        }
    
    def reset(self):
        """Reset processor state"""
        self.audio_buffer.clear()
        self.speech_segments.clear()
        self.segments_processed = 0
        self.gaps_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        self.last_speech_time = 0
        self.current_segment_start = 0
        
        logger.info("✅ UNDERSTANDING-ONLY audio processor reset")
    
    async def cleanup(self):
        """UNDERSTANDING-ONLY: Enhanced cleanup"""
        logger.info("🧹 Starting UNDERSTANDING-ONLY audio processor cleanup...")
        
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"UNDERSTANDING-ONLY executor shutdown error: {e}")
        
        self.reset()
        logger.info("✅ UNDERSTANDING-ONLY audio processor fully cleaned up")


# Alternative implementation if the above doesn't work
class AudioProcessor(UnderstandingAudioProcessor):
    """Fallback implementation for backward compatibility"""
    pass
