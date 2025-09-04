# FIXED AUDIO PROCESSOR - WITH CORRECT THREADPOOL HANDLING AND STREAMING
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

class FixedAudioProcessor:
    """FIXED: Audio processor with corrected ThreadPool handling and streaming"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        conversation_manager=None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.conversation_manager = conversation_manager
        
        # FIXED: FFmpeg processes for both modes
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # FIXED: PCM buffers with proper management
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # FIXED: Processing thresholds (ensure integers)
        self.transcribe_threshold = int(sample_rate * 0.5)  # 500ms
        self.understand_threshold = int(sample_rate * 1.0)  # 1 second
        
        # Voice Activity Detection
        try:
            self.vad = webrtcvad.Vad(2)
            self.vad_enabled = True
            logger.info("✅ WebRTC VAD initialized")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # Statistics
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # FIXED: ThreadPoolExecutor without timeout parameter
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AudioProc")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        logger.info(f"✅ FIXED AudioProcessor initialized: {sample_rate}Hz, {channels}ch, VAD: {self.vad_enabled}")
    
    async def start_ffmpeg_decoder(self, mode: str, websocket=None):
        """FIXED: Start FFmpeg decoder with better error handling"""
        try:
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # FIXED: Improved FFmpeg configuration
            ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm', thread_queue_size=512)
                .output(
                    'pipe:1', 
                    format='s16le', 
                    acodec='pcm_s16le', 
                    ac=self.channels, 
                    ar=str(self.sample_rate),
                    audio_bitrate='128k',
                    bufsize='512k'
                )
                .run_async(
                    pipe_stdin=True, 
                    pipe_stdout=True, 
                    pipe_stderr=True, 
                    quiet=True,
                    overwrite_output=True
                )
            )
            
            if mode == "transcribe":
                self.transcribe_ffmpeg_process = ffmpeg_process
                asyncio.create_task(self._read_pcm_output_fixed(mode, websocket))
                logger.info("✅ FIXED FFmpeg transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                asyncio.create_task(self._read_pcm_output_fixed(mode, websocket))
                logger.info("✅ FIXED FFmpeg understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start FIXED FFmpeg decoder for {mode}: {e}")
            raise RuntimeError(f"FIXED FFmpeg {mode} initialization failed: {e}")
    
    async def _read_pcm_output_fixed(self, mode: str, websocket=None):
        """FIXED: Background PCM reader with proper error handling"""
        loop = asyncio.get_event_loop()
        conn_id = id(websocket) if websocket else None
        
        ffmpeg_process = (
            self.transcribe_ffmpeg_process if mode == "transcribe" 
            else self.understand_ffmpeg_process
        )
        
        pcm_buffer = (
            self.transcribe_pcm_buffer if mode == "transcribe" 
            else self.understand_pcm_buffer
        )
        
        consecutive_errors = 0
        max_errors = 3  # FIXED: Reduced error threshold
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout:
                try:
                    # FIXED: Read PCM data with timeout
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            ffmpeg_process.stdout.read, 
                            4096
                        ),
                        timeout=3.0  # FIXED: Reduced timeout
                    )
                    
                    if not chunk:
                        logger.warning(f"FIXED FFmpeg {mode} stdout closed")
                        break
                    
                    # Add to PCM buffer
                    pcm_buffer.extend(chunk)
                    
                    # FIXED: Buffer size management (30 seconds max)
                    max_buffer_size = int(self.sample_rate * 30 * 2)  # 30 seconds
                    if len(pcm_buffer) > max_buffer_size:
                        excess = len(pcm_buffer) - max_buffer_size
                        del pcm_buffer[:excess]
                        logger.debug(f"Trimmed {mode} buffer by {excess} bytes")
                    
                    # Update activity
                    if conn_id:
                        self.last_activity[conn_id] = time.time()
                    
                    consecutive_errors = 0  # Reset on success
                    
                except asyncio.TimeoutError:
                    logger.debug(f"FIXED FFmpeg {mode} read timeout (normal)")
                    consecutive_errors += 1
                except Exception as e:
                    logger.error(f"Error reading FIXED FFmpeg {mode} output: {e}")
                    consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    logger.error(f"Too many consecutive errors in {mode}, stopping")
                    break
                    
        except Exception as e:
            logger.error(f"FIXED PCM reader failed for {mode}: {e}")
        finally:
            # Cleanup
            if conn_id and conn_id in self.active_connections:
                self.active_connections.discard(conn_id)
                self.last_activity.pop(conn_id, None)
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """FIXED: Transcription processing with proper error handling"""
        start_time = time.time()
        
        try:
            # FIXED: Better input validation
            if not webm_data or len(webm_data) < 50:  # Minimum valid WebM size
                logger.debug("Insufficient WebM data")
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe", websocket)
            
            # FIXED: Send to FFmpeg with error handling
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    
                    # FIXED: Small processing delay
                    await asyncio.sleep(0.01)
                    
                    self.chunks_processed += 1
                    
                    # Check if ready to process
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        result = self._process_pcm_buffer_fixed("transcribe", websocket)
                        
                        # Record processing time
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg transcription pipe broken, restarting...")
                    await self._restart_ffmpeg_process("transcribe", websocket)
                except Exception as e:
                    logger.error(f"Error in FIXED transcription processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"FIXED transcription error: {e}")
            return {"error": f"FIXED processing failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """FIXED: Understanding processing with proper error handling"""
        start_time = time.time()
        
        try:
            # FIXED: Input validation
            if not webm_data or len(webm_data) < 50:
                logger.debug("Insufficient WebM data for understanding")
                return None
            
            if not self.understand_ffmpeg_process:
                await self.start_ffmpeg_decoder("understand", websocket)
            
            # FIXED: Send to FFmpeg
            if self.understand_ffmpeg_process and self.understand_ffmpeg_process.stdin:
                try:
                    self.understand_ffmpeg_process.stdin.write(webm_data)
                    self.understand_ffmpeg_process.stdin.flush()
                    
                    await asyncio.sleep(0.01)
                    
                    # Check if ready to process
                    if len(self.understand_pcm_buffer) >= self.understand_threshold:
                        result = self._process_pcm_buffer_fixed("understand", websocket)
                        
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg understanding pipe broken, restarting...")
                    await self._restart_ffmpeg_process("understand", websocket)
                except Exception as e:
                    logger.error(f"Error in FIXED understanding processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"FIXED understanding error: {e}")
            return {"error": f"FIXED processing failed: {str(e)}"}
    
    def _process_pcm_buffer_fixed(self, mode: str, websocket=None) -> Dict[str, Any]:
        """FIXED: PCM buffer processing with proper integer handling"""
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
            
            # FIXED: Extract audio with overlap (proper integer slicing)
            overlap_samples = int(self.sample_rate * 0.1)  # 100ms overlap
            end_index = min(threshold + overlap_samples, len(pcm_buffer))
            
            audio_data = bytes(pcm_buffer[:end_index])
            del pcm_buffer[:threshold]  # Keep overlap
            
            # Create WAV file
            wav_bytes = self._pcm_to_wav_fixed(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000
            self.total_audio_length += duration_ms
            
            # FIXED: Speech detection
            speech_ratio = self._estimate_speech_ratio_fixed(audio_data)
            
            if speech_ratio > 0.1:
                self.speech_chunks_detected += 1
            
            # Get conversation context
            conversation_context = ""
            if self.conversation_manager and websocket:
                conversation_context = self.conversation_manager.get_conversation_context(websocket)
            
            logger.info(f"🎤 FIXED processed {mode}: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": time.time(),
                "fixed": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"FIXED PCM buffer processing error for {mode}: {e}")
            return {"error": f"FIXED processing failed: {str(e)}"}
    
    def _estimate_speech_ratio_fixed(self, pcm_data: bytes) -> float:
        """FIXED: Speech detection with WebRTC VAD and fallbacks"""
        try:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Method 1: WebRTC VAD
            if self.vad_enabled and self.vad:
                try:
                    frame_size = int(self.sample_rate * 0.030)  # 30ms frames
                    speech_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(audio_array) - frame_size, frame_size):
                        frame = audio_array[i:i + frame_size]
                        if len(frame) == frame_size:
                            frame_bytes = frame.astype(np.int16).tobytes()
                            if self.vad.is_speech(frame_bytes, self.sample_rate):
                                speech_frames += 1
                            total_frames += 1
                    
                    if total_frames > 0:
                        return speech_frames / total_frames
                        
                except Exception as e:
                    logger.debug(f"VAD error: {e}")
            
            # Method 2: Energy-based detection
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            energy_ratio = min(1.0, rms_energy / 1000.0)
            
            # Method 3: Zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / len(audio_array)
            zcr_ratio = 1.0 if 0.01 <= zcr_normalized <= 0.5 else 0.5
            
            # Combine methods
            final_ratio = (energy_ratio * 0.7 + zcr_ratio * 0.3)
            
            return min(1.0, final_ratio)
            
        except Exception as e:
            logger.error(f"FIXED speech ratio error: {e}")
            return 0.3  # Conservative fallback
    
    def _pcm_to_wav_fixed(self, pcm_data: bytes) -> bytes:
        """FIXED: PCM to WAV conversion"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            logger.debug(f"FIXED WAV conversion: {len(pcm_data)} PCM → {len(wav_bytes)} WAV bytes")
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"FIXED WAV conversion error: {e}")
            raise RuntimeError(f"FIXED WAV conversion failed: {e}")
    
    async def _restart_ffmpeg_process(self, mode: str, websocket=None):
        """FIXED: FFmpeg restart with proper cleanup"""
        try:
            logger.info(f"🔄 Restarting FIXED {mode} FFmpeg process...")
            
            # Clean up old process
            if mode == "transcribe" and self.transcribe_ffmpeg_process:
                try:
                    self.transcribe_ffmpeg_process.terminate()
                    await asyncio.sleep(0.5)
                except:
                    pass
                self.transcribe_ffmpeg_process = None
            elif mode == "understand" and self.understand_ffmpeg_process:
                try:
                    self.understand_ffmpeg_process.terminate()
                    await asyncio.sleep(0.5)
                except:
                    pass
                self.understand_ffmpeg_process = None
            
            # Restart
            await self.start_ffmpeg_decoder(mode, websocket)
            
        except Exception as e:
            logger.error(f"Failed to restart FIXED {mode} process: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        speech_detection_rate = (
            self.speech_chunks_detected / max(self.chunks_processed, 1)
        )
        
        return {
            "chunks_processed": self.chunks_processed,
            "speech_chunks_detected": self.speech_chunks_detected,
            "speech_detection_rate": round(speech_detection_rate, 3),
            "total_audio_length_ms": self.total_audio_length,
            "transcribe_buffer_size": len(self.transcribe_pcm_buffer),
            "understand_buffer_size": len(self.understand_pcm_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcribe_ffmpeg_running": self.transcribe_ffmpeg_process is not None,
            "understand_ffmpeg_running": self.understand_ffmpeg_process is not None,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.active_connections),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "fixed_version": True
        }
    
    def reset(self):
        """Reset processor state"""
        self.transcribe_pcm_buffer.clear()
        self.understand_pcm_buffer.clear()
        self.chunks_processed = 0
        self.speech_chunks_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        
        logger.info("✅ FIXED audio processor reset")
    
    async def cleanup(self):
        """FIXED: Cleanup without timeout parameter"""
        logger.info("🧹 Starting FIXED audio processor cleanup...")
        
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
                    if process.stderr:
                        process.stderr.close()
                    
                    process.terminate()
                    try:
                        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await asyncio.to_thread(process.wait)
                    
                    logger.info(f"✅ FIXED FFmpeg {mode} process cleaned up")
                except Exception as e:
                    logger.error(f"FIXED FFmpeg {mode} cleanup error: {e}")
        
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # FIXED: Shutdown without timeout parameter
        try:
            self.executor.shutdown(wait=True)  # Removed timeout parameter
        except Exception as e:
            logger.error(f"FIXED executor shutdown error: {e}")
        
        self.reset()
        logger.info("✅ FIXED audio processor fully cleaned up")

# Create alias for backwards compatibility
AudioProcessor = FixedAudioProcessor
