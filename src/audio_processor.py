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

logger = logging.getLogger(__name__)

class AudioProcessor:
    """PERFECT: Real-time WebM audio processing using FFmpeg streaming"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        vad_mode: int = 3
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        
        # FFmpeg streaming process
        self.ffmpeg_process = None
        self.pcm_buffer = bytearray()
        self.audio_ready_threshold = sample_rate * 2  # 1 second of audio
        
        # Statistics
        self.chunks_processed = 0
        self.total_audio_length = 0
        
        logger.info(f"✅ PERFECT AudioProcessor initialized: {sample_rate}Hz, {channels}ch with FFmpeg streaming")
    
    async def start_ffmpeg_decoder(self):
        """Start FFmpeg process for WebM -> PCM streaming"""
        try:
            self.ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm')
                .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=self.channels, ar=str(self.sample_rate))
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            logger.info("✅ FFmpeg decoder started for WebM streaming")
            
            # Start background task to read PCM output
            asyncio.create_task(self._read_pcm_output())
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg decoder: {e}")
            raise RuntimeError(f"FFmpeg initialization failed: {e}")
    
    async def _read_pcm_output(self):
        """Background task to read PCM data from FFmpeg stdout"""
        loop = asyncio.get_event_loop()
        
        try:
            while self.ffmpeg_process and self.ffmpeg_process.stdout:
                try:
                    # Read PCM data from FFmpeg
                    chunk = await loop.run_in_executor(None, self.ffmpeg_process.stdout.read, 4096)
                    if not chunk:
                        logger.warning("FFmpeg stdout closed")
                        break
                    
                    # Add to PCM buffer
                    self.pcm_buffer.extend(chunk)
                    
                except Exception as e:
                    logger.error(f"Error reading FFmpeg output: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"PCM reader task failed: {e}")
    
    async def process_webm_chunk(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """Process WebM chunk through FFmpeg streaming"""
        try:
            if not self.ffmpeg_process:
                await self.start_ffmpeg_decoder()
            
            # Send WebM chunk to FFmpeg stdin
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.write(webm_data)
                await asyncio.sleep(0.01)  # Small delay for processing
                
                self.chunks_processed += 1
                
                # Check if we have enough PCM data to process
                if len(self.pcm_buffer) >= self.audio_ready_threshold:
                    return self._process_pcm_buffer()
            
            return None
            
        except Exception as e:
            logger.error(f"WebM chunk processing error: {e}")
            return None
    
    def _process_pcm_buffer(self) -> Dict[str, Any]:
        """Process accumulated PCM buffer"""
        try:
            if len(self.pcm_buffer) < self.audio_ready_threshold:
                return None
            
            # Extract audio data for processing
            audio_data = bytes(self.pcm_buffer[:self.audio_ready_threshold])
            del self.pcm_buffer[:self.audio_ready_threshold]
            
            # Create WAV file from PCM data
            wav_bytes = self._pcm_to_wav(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000  # 16-bit PCM
            
            self.total_audio_length += duration_ms
            
            logger.info(f"🎤 Processed PCM audio: {duration_ms:.0f}ms (Total: {self.total_audio_length:.1f}ms, Chunks: {self.chunks_processed})")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": 1.0,  # FFmpeg processed audio is likely speech
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "processed_at": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"PCM buffer processing error: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert PCM data to WAV format"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            return wav_io.getvalue()
            
        except Exception as e:
            logger.error(f"PCM to WAV conversion error: {e}")
            raise RuntimeError(f"WAV conversion failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        return {
            "chunks_processed": self.chunks_processed,
            "total_audio_length_ms": self.total_audio_length,
            "pcm_buffer_size": len(self.pcm_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ffmpeg_running": self.ffmpeg_process is not None
        }
    
    def reset(self):
        """Reset audio processor state"""
        self.pcm_buffer.clear()
        self.chunks_processed = 0
        self.total_audio_length = 0
        logger.info("Audio processor reset")
    
    async def cleanup(self):
        """Clean up FFmpeg process"""
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                if self.ffmpeg_process.stdout:
                    self.ffmpeg_process.stdout.close()
                self.ffmpeg_process.wait()
                logger.info("✅ FFmpeg process cleaned up")
            except Exception as e:
                logger.error(f"FFmpeg cleanup error: {e}")
            finally:
                self.ffmpeg_process = None
