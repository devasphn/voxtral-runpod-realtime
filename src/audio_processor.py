# FIXED AUDIO PROCESSOR - UNIFIED APPROACH WITH IMPROVED SPEECH DETECTION
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

logger = logging.getLogger(__name__)

class AudioProcessor:
    """UNIFIED: WebM audio processing using FFmpeg streaming for BOTH modes"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        
        # Unified FFmpeg streaming approach for both modes
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # PCM buffers for both modes
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # Audio ready thresholds (reduced for better responsiveness)
        self.transcribe_threshold = sample_rate * 1  # 1 second for transcription
        self.understand_threshold = sample_rate * 2  # 2 seconds for understanding
        
        # Statistics
        self.chunks_processed = 0
        self.total_audio_length = 0
        
        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"✅ UNIFIED AudioProcessor initialized: {sample_rate}Hz, {channels}ch")
    
    async def start_ffmpeg_decoder(self, mode: str):
        """Start FFmpeg process for specific mode"""
        try:
            ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm')
                .output(
                    'pipe:1', 
                    format='s16le', 
                    acodec='pcm_s16le', 
                    ac=self.channels, 
                    ar=str(self.sample_rate)
                )
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
            )
            
            if mode == "transcribe":
                self.transcribe_ffmpeg_process = ffmpeg_process
                # Start background PCM reader
                asyncio.create_task(self._read_pcm_output(mode))
                logger.info("✅ FFmpeg transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                # Start background PCM reader
                asyncio.create_task(self._read_pcm_output(mode))
                logger.info("✅ FFmpeg understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start FFmpeg decoder for {mode}: {e}")
            raise RuntimeError(f"FFmpeg {mode} initialization failed: {e}")
    
    async def _read_pcm_output(self, mode: str):
        """Background task to read PCM data from FFmpeg stdout"""
        loop = asyncio.get_event_loop()
        
        ffmpeg_process = (
            self.transcribe_ffmpeg_process if mode == "transcribe" 
            else self.understand_ffmpeg_process
        )
        
        pcm_buffer = (
            self.transcribe_pcm_buffer if mode == "transcribe" 
            else self.understand_pcm_buffer
        )
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout:
                try:
                    # Read PCM data from FFmpeg
                    chunk = await loop.run_in_executor(
                        self.executor, 
                        ffmpeg_process.stdout.read, 
                        4096
                    )
                    
                    if not chunk:
                        logger.warning(f"FFmpeg {mode} stdout closed")
                        break
                    
                    # Add to PCM buffer
                    pcm_buffer.extend(chunk)
                    
                except Exception as e:
                    logger.error(f"Error reading FFmpeg {mode} output: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"PCM reader task failed for {mode}: {e}")
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """Process WebM chunk for TRANSCRIPTION mode using FFmpeg streaming"""
        try:
            # Validate input
            if not webm_data or len(webm_data) < 10:
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe")
            
            # Send WebM chunk to FFmpeg stdin
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    await asyncio.sleep(0.005)  # Small delay for processing
                    
                    self.chunks_processed += 1
                    
                    # Check if we have enough PCM data to process
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        return self._process_pcm_buffer("transcribe")
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg transcription process ended, restarting...")
                    await self.start_ffmpeg_decoder("transcribe")
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg transcription: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"WebM transcription chunk processing error: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """Process WebM chunk for UNDERSTANDING mode using FFmpeg streaming"""
        try:
            # Validate input
            if not webm_data or len(webm_data) < 10:
                return None
            
            if not self.understand_ffmpeg_process:
                await self.start_ffmpeg_decoder("understand")
            
            # Send WebM chunk to FFmpeg stdin
            if self.understand_ffmpeg_process and self.understand_ffmpeg_process.stdin:
                try:
                    self.understand_ffmpeg_process.stdin.write(webm_data)
                    self.understand_ffmpeg_process.stdin.flush()
                    await asyncio.sleep(0.005)  # Small delay for processing
                    
                    # Check if we have enough PCM data to process
                    if len(self.understand_pcm_buffer) >= self.understand_threshold:
                        return self._process_pcm_buffer("understand")
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg understanding process ended, restarting...")
                    await self.start_ffmpeg_decoder("understand")
                except Exception as e:
                    logger.error(f"Error writing to FFmpeg understanding: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"WebM understanding chunk processing error: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _process_pcm_buffer(self, mode: str) -> Dict[str, Any]:
        """Process accumulated PCM buffer for specified mode"""
        try:
            pcm_buffer = (
                self.transcribe_pcm_buffer if mode == "transcribe" 
                else self.understand_pcm_buffer
            )
            threshold = (
                self.transcribe_threshold if mode == "transcribe" 
                else self.understand_threshold
            )
            
            if len(pcm_buffer) < threshold:
                return None
            
            # Extract audio data for processing
            audio_data = bytes(pcm_buffer[:threshold])
            del pcm_buffer[:threshold]
            
            # Create WAV file from PCM data
            wav_bytes = self._pcm_to_wav(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000  # 16-bit PCM
            
            self.total_audio_length += duration_ms
            
            # IMPROVED speech detection with better logging
            speech_ratio = self._estimate_speech_ratio(audio_data)
            
            logger.info(f"🎤 Processed {mode} PCM audio: {duration_ms:.0f}ms (Total: {self.total_audio_length:.1f}ms) - Speech Ratio: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"PCM buffer processing error for {mode}: {e}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _estimate_speech_ratio(self, pcm_data: bytes) -> float:
        """IMPROVED: Estimate speech activity ratio with better thresholds"""
        try:
            # Convert PCM to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Calculate RMS energy for the entire audio
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            
            # MUCH lower threshold for better detection
            # Typical silence RMS is < 100, speech is > 500
            silence_threshold = 200  # Much lower than before
            
            # Calculate frame-based VAD with improved logic
            frame_size = int(self.sample_rate * 0.025)  # 25ms frames
            hop_size = int(self.sample_rate * 0.010)   # 10ms hop
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_array) - frame_size, hop_size):
                frame = audio_array[i:i + frame_size]
                frame_rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
                
                # Much more lenient threshold
                if frame_rms > silence_threshold:
                    speech_frames += 1
                total_frames += 1
            
            frame_ratio = speech_frames / max(total_frames, 1) if total_frames > 0 else 0.0
            
            # Return higher of RMS-based or frame-based detection
            rms_ratio = min(1.0, rms_energy / 1000.0)  # Normalize to 0-1
            final_ratio = max(frame_ratio, rms_ratio)
            
            logger.debug(f"Speech detection: RMS={rms_energy:.1f}, Frame ratio={frame_ratio:.3f}, Final={final_ratio:.3f}")
            
            return final_ratio
            
        except Exception as e:
            logger.error(f"Speech ratio estimation error: {e}")
            return 0.5  # Return moderate value if estimation fails
    
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
        """Get comprehensive audio processing statistics"""
        return {
            "chunks_processed": self.chunks_processed,
            "total_audio_length_ms": self.total_audio_length,
            "transcribe_pcm_buffer_size": len(self.transcribe_pcm_buffer),
            "understand_pcm_buffer_size": len(self.understand_pcm_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcribe_ffmpeg_running": self.transcribe_ffmpeg_process is not None,
            "understand_ffmpeg_running": self.understand_ffmpeg_process is not None
        }
    
    def reset(self):
        """Reset audio processor state"""
        self.transcribe_pcm_buffer.clear()
        self.understand_pcm_buffer.clear()
        self.chunks_processed = 0
        self.total_audio_length = 0
        logger.info("Audio processor reset")
    
    async def cleanup(self):
        """Clean up FFmpeg processes"""
        processes = [
            ("transcribe", self.transcribe_ffmpeg_process),
            ("understand", self.understand_ffmpeg_process)
        ]
        
        for mode, process in processes:
            if process:
                try:
                    if process.stdin:
                        process.stdin.close()
                    if process.stdout:
                        process.stdout.close()
                    process.wait()
                    logger.info(f"✅ FFmpeg {mode} process cleaned up")
                except Exception as e:
                    logger.error(f"FFmpeg {mode} cleanup error: {e}")
        
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("✅ Audio processor fully cleaned up")
