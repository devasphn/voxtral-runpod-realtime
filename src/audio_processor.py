import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import io
import wave
import json
from pydub import AudioSegment
import webrtcvad
import collections

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Real-time audio processing for Voxtral model"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,  # 30ms chunks
        vad_mode: int = 3  # Aggressive VAD
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.frame_duration_ms = 30  # 30ms frames for VAD
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(vad_mode)
        
        # Buffer for audio chunks
        self.audio_buffer = collections.deque(maxlen=100)  # ~3 seconds at 30ms chunks
        self.silence_threshold = 10  # consecutive silent chunks before processing
        self.silence_count = 0
        
        logger.info(f"AudioProcessor initialized: {sample_rate}Hz, {channels}ch")
    
    def process_chunk(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process incoming audio chunk"""
        try:
            # Convert to AudioSegment for processing
            audio_segment = self._bytes_to_audio_segment(audio_data)
            if audio_segment is None:
                return None
            
            # Normalize audio format
            audio_segment = self._normalize_audio(audio_segment)
            
            # Convert to bytes for VAD
            audio_bytes = audio_segment.raw_data
            
            # Voice activity detection
            is_speech = self._detect_voice_activity(audio_bytes)
            
            # Add to buffer
            self.audio_buffer.append({
                "audio": audio_bytes,
                "is_speech": is_speech,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Check if we should process accumulated audio
            if self._should_process_buffer():
                return self._process_buffer()
            
            return None
            
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            return None
    
    def _bytes_to_audio_segment(self, audio_data: bytes) -> Optional[AudioSegment]:
        """Convert bytes to AudioSegment"""
        try:
            # Try different formats
            audio_io = io.BytesIO(audio_data)
            
            # Try as WAV first
            try:
                return AudioSegment.from_wav(audio_io)
            except:
                pass
            
            # Try as raw PCM
            try:
                audio_io.seek(0)
                return AudioSegment(
                    data=audio_data,
                    sample_width=2,  # 16-bit
                    frame_rate=self.sample_rate,
                    channels=self.channels
                )
            except:
                pass
            
            # Try auto-detection
            try:
                audio_io.seek(0)
                return AudioSegment.from_file(audio_io)
            except:
                pass
            
            logger.warning("Could not decode audio data")
            return None
            
        except Exception as e:
            logger.error(f"Audio decoding error: {e}")
            return None
    
    def _normalize_audio(self, audio_segment: AudioSegment) -> AudioSegment:
        """Normalize audio to required format"""
        # Convert to mono if needed
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Convert to target sample rate
        if audio_segment.frame_rate != self.sample_rate:
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
        
        # Ensure 16-bit depth
        audio_segment = audio_segment.set_sample_width(2)
        
        return audio_segment
    
    def _detect_voice_activity(self, audio_bytes: bytes) -> bool:
        """Detect voice activity in audio chunk"""
        try:
            # VAD expects 16kHz, 16-bit PCM
            frame_length = int(self.sample_rate * self.frame_duration_ms / 1000) * 2  # 2 bytes per sample
            
            if len(audio_bytes) < frame_length:
                return False
            
            # Use only first frame if chunk is longer
            frame = audio_bytes[:frame_length]
            
            # VAD detection
            return self.vad.is_speech(frame, self.sample_rate)
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Assume speech on error
    
    def _should_process_buffer(self) -> bool:
        """Determine if buffer should be processed"""
        if len(self.audio_buffer) < 5:  # Need minimum chunks
            return False
        
        # Check for speech activity
        recent_chunks = list(self.audio_buffer)[-5:]  # Last 5 chunks
        speech_chunks = sum(1 for chunk in recent_chunks if chunk["is_speech"])
        
        # Process if we have speech or enough silence
        if speech_chunks > 0:
            self.silence_count = 0
            return True
        else:
            self.silence_count += 1
            return self.silence_count >= self.silence_threshold
    
    def _process_buffer(self) -> Dict[str, Any]:
        """Process accumulated audio buffer"""
        try:
            if not self.audio_buffer:
                return {"error": "Empty buffer"}
            
            # Combine audio chunks
            combined_audio = b"".join(chunk["audio"] for chunk in self.audio_buffer)
            
            # Create AudioSegment
            audio_segment = AudioSegment(
                data=combined_audio,
                sample_width=2,
                frame_rate=self.sample_rate,
                channels=self.channels
            )
            
            # Apply noise reduction (simple)
            audio_segment = self._reduce_noise(audio_segment)
            
            # Export as WAV bytes
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()
            
            # Calculate statistics
            duration_ms = len(audio_segment)
            speech_chunks = sum(1 for chunk in self.audio_buffer if chunk["is_speech"])
            
            # Clear buffer
            self.audio_buffer.clear()
            self.silence_count = 0
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_chunks / len(self.audio_buffer) if self.audio_buffer else 0,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "processed_at": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _reduce_noise(self, audio_segment: AudioSegment) -> AudioSegment:
        """Simple noise reduction"""
        try:
            # Apply high-pass filter to remove low-frequency noise
            audio_segment = audio_segment.high_pass_filter(80)  # Remove below 80Hz
            
            # Normalize volume
            audio_segment = audio_segment.normalize()
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_segment
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        return {
            "buffer_size": len(self.audio_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_duration_ms": self.chunk_duration_ms,
            "silence_count": self.silence_count,
            "silence_threshold": self.silence_threshold
        }
    
    def reset(self):
        """Reset audio processor state"""
        self.audio_buffer.clear()
        self.silence_count = 0
        logger.info("Audio processor reset")
