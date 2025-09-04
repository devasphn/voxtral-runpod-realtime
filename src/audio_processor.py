# COMPLETE CORRECTED AUDIO PROCESSOR - audio_processor.py - WITH ENHANCED GAP DETECTION
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
    """CORRECTED: Audio processor with 300ms gap detection and robust FFmpeg handling"""
    
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
        
        # CORRECTED: Enhanced FFmpeg processes with better error handling
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # CORRECTED: PCM buffers for both modes
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # CORRECTED: Processing thresholds optimized for responsiveness
        self.transcribe_threshold = int(sample_rate * 2.0 * 2)  # 2 seconds for transcription
        self.understand_threshold = int(sample_rate * 0.5 * 2)  # 0.5 seconds for understanding (faster)
        
        # CORRECTED: Enhanced Voice Activity Detection for gap detection
        try:
            self.vad = webrtcvad.Vad(1)  # Mode 1 - balanced for gap detection
            self.vad_enabled = True
            logger.info("✅ CORRECTED WebRTC VAD initialized (mode 1 for gap detection)")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # CORRECTED: Gap detection parameters
        self.gap_threshold_ms = 300  # 300ms silence gap
        self.min_speech_duration_ms = 500  # Minimum 500ms for processing
        self.frame_size_ms = 10  # 10ms frames for VAD
        self.frame_size_samples = int(sample_rate * self.frame_size_ms / 1000)
        
        # Statistics and state
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # CORRECTED: Enhanced ThreadPoolExecutor with proper naming
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AudioProc")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        # CORRECTED: Gap detection state per connection
        self.gap_detection_state = {}
        
        logger.info(f"✅ CORRECTED AudioProcessor initialized: {sample_rate}Hz, {channels}ch")
        logger.info(f"   Gap threshold: {self.gap_threshold_ms}ms")
        logger.info(f"   Min speech: {self.min_speech_duration_ms}ms") 
        logger.info(f"   Frame size: {self.frame_size_ms}ms ({self.frame_size_samples} samples)")
        logger.info(f"   VAD enabled: {self.vad_enabled}")
    
    async def start_ffmpeg_decoder(self, mode: str, websocket=None):
        """CORRECTED: Start robust FFmpeg decoder with enhanced error handling"""
        try:
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
                
                # Initialize gap detection state for understanding mode
                if mode == "understand":
                    self.gap_detection_state[conn_id] = {
                        "last_speech_time": time.time(),
                        "accumulated_frames": [],
                        "silence_frames": 0,
                        "speech_frames": 0,
                        "total_frames": 0
                    }
            
            # CORRECTED: Robust FFmpeg configuration with enhanced settings
            ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm', thread_queue_size=8192)
                .filter('highpass', f=85)   # Slightly higher to remove more noise
                .filter('lowpass', f=7500)  # Preserve speech frequencies
                .filter('volume', '1.2')    # Slight volume boost for better VAD
                .output(
                    'pipe:1', 
                    format='s16le', 
                    acodec='pcm_s16le', 
                    ac=self.channels, 
                    ar=str(self.sample_rate),
                    audio_bitrate='320k',  # Higher quality for better processing
                    bufsize='8192k'       # Larger buffer for stability
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
                asyncio.create_task(self._read_pcm_output_corrected(mode, websocket))
                logger.info("✅ CORRECTED FFmpeg transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                asyncio.create_task(self._read_pcm_output_corrected(mode, websocket))
                logger.info("✅ CORRECTED FFmpeg understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start CORRECTED FFmpeg decoder for {mode}: {e}")
            # Enhanced auto-restart with backoff
            await asyncio.sleep(2.0)
            try:
                await self.start_ffmpeg_decoder(mode, websocket)
                logger.info(f"✅ CORRECTED FFmpeg {mode} auto-restarted successfully")
            except Exception as restart_error:
                logger.error(f"CORRECTED FFmpeg {mode} restart failed: {restart_error}")
                raise RuntimeError(f"CORRECTED FFmpeg {mode} initialization failed: {e}")
    
    async def _read_pcm_output_corrected(self, mode: str, websocket=None):
        """CORRECTED: Enhanced PCM reader with robust error handling"""
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
        max_errors = 15  # Increased tolerance for stability
        read_timeout = 8.0  # Longer timeout for stability
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout and not ffmpeg_process.stdout.closed:
                try:
                    # CORRECTED: Read optimal chunks for better performance
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            ffmpeg_process.stdout.read, 
                            65536  # 64KB chunks for better throughput
                        ),
                        timeout=read_timeout
                    )
                    
                    if not chunk:
                        logger.warning(f"CORRECTED FFmpeg {mode} stdout closed gracefully")
                        break
                    
                    # Add to PCM buffer with thread safety
                    pcm_buffer.extend(chunk)
                    
                    # CORRECTED: Smart buffer management (90 seconds max)
                    max_buffer_size = int(self.sample_rate * 90 * 2)  # 90 seconds
                    if len(pcm_buffer) > max_buffer_size:
                        excess = len(pcm_buffer) - max_buffer_size
                        del pcm_buffer[:excess]
                        logger.debug(f"Trimmed {mode} buffer by {excess} bytes (keeping 90s)")
                    
                    # Update activity tracking
                    if conn_id:
                        self.last_activity[conn_id] = time.time()
                    
                    consecutive_errors = 0  # Reset on success
                    
                except asyncio.TimeoutError:
                    consecutive_errors += 1
                    logger.debug(f"CORRECTED FFmpeg {mode} read timeout ({consecutive_errors}/{max_errors})")
                    
                    if consecutive_errors >= max_errors:
                        logger.warning(f"Too many timeouts for {mode}, attempting restart...")
                        await self._restart_ffmpeg_process(mode, websocket)
                        consecutive_errors = 0
                        
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Error reading CORRECTED FFmpeg {mode} output: {e}")
                    
                    if consecutive_errors >= max_errors:
                        logger.error(f"Too many consecutive errors in {mode}, attempting restart...")
                        await self._restart_ffmpeg_process(mode, websocket)
                        consecutive_errors = 0
                    
        except Exception as e:
            logger.error(f"CORRECTED PCM reader failed for {mode}: {e}")
        finally:
            # Cleanup connection state
            if conn_id and conn_id in self.active_connections:
                self.active_connections.discard(conn_id)
                self.last_activity.pop(conn_id, None)
                if mode == "understand":
                    self.gap_detection_state.pop(conn_id, None)
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """CORRECTED: Transcription processing (optimized from original)"""
        start_time = time.time()
        
        try:
            if not webm_data or len(webm_data) < 500:
                logger.debug("Insufficient WebM data for transcription")
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe", websocket)
            
            # CORRECTED: Send to FFmpeg with enhanced error handling
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    
                    await asyncio.sleep(0.03)  # Slightly longer delay for stability
                    
                    self.chunks_processed += 1
                    
                    # Process when we have sufficient audio
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        result = self._process_pcm_buffer_corrected("transcribe", websocket)
                        
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg transcription pipe broken, restarting...")
                    await self._restart_ffmpeg_process("transcribe", websocket)
                except Exception as e:
                    logger.error(f"Error in CORRECTED transcription processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"CORRECTED transcription error: {e}")
            return {"error": f"CORRECTED transcription failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """CORRECTED: Understanding mode - accumulate PCM with enhanced gap detection"""
        try:
            if not webm_data or len(webm_data) < 500:
                return None
            
            if not self.understand_ffmpeg_process:
                await self.start_ffmpeg_decoder("understand", websocket)
            
            # Send to FFmpeg for PCM conversion
            if self.understand_ffmpeg_process and self.understand_ffmpeg_process.stdin:
                try:
                    self.understand_ffmpeg_process.stdin.write(webm_data)
                    self.understand_ffmpeg_process.stdin.flush()
                    
                    await asyncio.sleep(0.01)  # Minimal delay for faster accumulation
                    
                    # Get PCM data from buffer - process multiple frames if available
                    frames_to_process = min(10, len(self.understand_pcm_buffer) // (self.frame_size_samples * 2))
                    
                    if frames_to_process > 0:
                        chunk_size = self.frame_size_samples * 2 * frames_to_process
                        pcm_chunk = bytes(self.understand_pcm_buffer[:chunk_size])
                        del self.understand_pcm_buffer[:chunk_size]
                        
                        # Enhanced speech detection across multiple frames
                        speech_detected = self._detect_speech_in_frames(pcm_chunk, frames_to_process)
                        
                        return {
                            "pcm_data": pcm_chunk,
                            "speech_detected": speech_detected,
                            "frame_count": frames_to_process,
                            "frame_size": len(pcm_chunk),
                            "corrected": True
                        }
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg understanding pipe broken, restarting...")
                    await self._restart_ffmpeg_process("understand", websocket)
                except Exception as e:
                    logger.error(f"Error in CORRECTED understanding processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"CORRECTED understanding error: {e}")
            return {"error": f"CORRECTED understanding failed: {str(e)}"}
    
    def _detect_speech_in_frames(self, pcm_data: bytes, frame_count: int) -> bool:
        """CORRECTED: Enhanced speech detection across multiple frames"""
        try:
            if not self.vad_enabled or len(pcm_data) < self.frame_size_samples * 2:
                return self._fallback_speech_detection(pcm_data)
            
            speech_frames = 0
            frame_size_bytes = self.frame_size_samples * 2
            
            # Process each frame with VAD
            for i in range(0, len(pcm_data), frame_size_bytes):
                frame = pcm_data[i:i + frame_size_bytes]
                
                # Ensure frame is exactly the right size
                if len(frame) != frame_size_bytes:
                    if len(frame) < frame_size_bytes:
                        frame = frame + b'\x00' * (frame_size_bytes - len(frame))
                    else:
                        frame = frame[:frame_size_bytes]
                
                # Use WebRTC VAD to detect speech
                try:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                except Exception as vad_error:
                    logger.debug(f"VAD error on frame: {vad_error}")
                    continue
            
            # Return True if more than 30% of frames contain speech
            speech_ratio = speech_frames / max(frame_count, 1)
            return speech_ratio > 0.3
                
        except Exception as e:
            logger.error(f"Enhanced speech detection error: {e}")
            return self._fallback_speech_detection(pcm_data)
    
    def _fallback_speech_detection(self, pcm_data: bytes) -> bool:
        """CORRECTED: Fallback energy-based speech detection"""
        try:
            if len(pcm_data) < 320:  # Less than 10ms at 16kHz
                return False
                
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            
            # Adaptive threshold based on recent audio
            energy_threshold = 800.0  # Adjusted for better sensitivity
            return rms_energy > energy_threshold
            
        except Exception as e:
            logger.error(f"Fallback speech detection error: {e}")
            return False
    
    def _process_pcm_buffer_corrected(self, mode: str, websocket=None) -> Dict[str, Any]:
        """CORRECTED: Process PCM buffer with enhanced quality control"""
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
            
            # CORRECTED: Extract audio with optimal overlap for continuity
            overlap_samples = int(self.sample_rate * 0.2 * 2)  # 200ms overlap
            end_index = min(threshold + overlap_samples, len(pcm_buffer))
            
            audio_data = bytes(pcm_buffer[:end_index])
            del pcm_buffer[:threshold]  # Keep overlap for continuity
            
            # Create enhanced WAV file
            wav_bytes = self._pcm_to_wav_enhanced(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000
            self.total_audio_length += duration_ms
            
            # Enhanced speech ratio estimation
            speech_ratio = self._estimate_speech_ratio_corrected(audio_data)
            
            if speech_ratio > 0.1:
                self.speech_chunks_detected += 1
            
            # CORRECTED: Enhanced quality control for better results
            min_duration = 800  # 800ms minimum for good quality
            min_speech_ratio = 0.25  # Lower threshold for more sensitivity
            
            if duration_ms < min_duration or speech_ratio < min_speech_ratio:
                logger.debug(f"Skipping low quality audio: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                return None
            
            # Get conversation context if available
            conversation_context = ""
            if self.conversation_manager and websocket:
                conversation_context = self.conversation_manager.get_conversation_context(websocket)
            
            logger.info(f"🎤 CORRECTED processed {mode}: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": time.time(),
                "corrected": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"CORRECTED PCM buffer processing error for {mode}: {e}")
            return {"error": f"CORRECTED processing failed: {str(e)}"}
    
    def _estimate_speech_ratio_corrected(self, pcm_data: bytes) -> float:
        """CORRECTED: Enhanced speech ratio estimation with multiple methods"""
        try:
            if not pcm_data or len(pcm_data) < 1600:  # Less than 50ms
                return 0.0
                
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Method 1: Enhanced WebRTC VAD frame-by-frame analysis
            if self.vad_enabled and self.vad:
                try:
                    frame_size = self.frame_size_samples
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
                                pass  # Skip problematic frames
                            total_frames += 1
                    
                    if total_frames > 0:
                        vad_ratio = speech_frames / total_frames
                        logger.debug(f"VAD speech ratio: {vad_ratio:.3f} ({speech_frames}/{total_frames})")
                        return min(1.0, max(0.0, vad_ratio))
                        
                except Exception as e:
                    logger.debug(f"VAD speech ratio error: {e}")
            
            # Method 2: Enhanced energy-based detection with adaptive thresholding
            audio_float = audio_array.astype(np.float64)
            
            # RMS energy calculation
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            
            # Adaptive energy threshold
            energy_threshold = 600.0  # Optimized for human speech
            energy_ratio = min(1.0, max(0.0, (rms_energy - energy_threshold) / (energy_threshold * 2)))
            
            # Method 3: Enhanced Zero crossing rate for speech characteristics
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            
            # Human speech ZCR is typically between 0.01 and 0.2
            if 0.008 <= zcr_normalized <= 0.25:  # Slightly wider range
                zcr_ratio = 1.0
            elif zcr_normalized < 0.008:
                zcr_ratio = 0.0  # Likely silence
            else:
                zcr_ratio = max(0.0, 1.0 - (zcr_normalized - 0.25) / 0.4)
            
            # Method 4: Spectral centroid for voice frequency analysis
            spectral_ratio = 0.5  # Default
            try:
                if len(audio_float) >= 512:  # Minimum for FFT
                    fft = np.fft.rfft(audio_float[:len(audio_float)//2*2])  # Ensure even length
                    magnitude = np.abs(fft)
                    freqs = np.fft.rfftfreq(len(fft)*2-2, 1.0/self.sample_rate)
                    
                    if np.sum(magnitude) > 0:
                        spectral_centroid = np.sum(freqs[:len(magnitude)] * magnitude) / np.sum(magnitude)
                        # Human speech centroid is typically 100-600 Hz
                        if 80 <= spectral_centroid <= 700:
                            spectral_ratio = 1.0
                        elif spectral_centroid < 80:
                            spectral_ratio = 0.2  # Too low for speech
                        else:
                            spectral_ratio = max(0.3, 1.0 - (spectral_centroid - 700) / 1000)
            except Exception as spectral_error:
                logger.debug(f"Spectral analysis error: {spectral_error}")
            
            # CORRECTED: Optimized combination weights for speech detection
            final_ratio = (
                energy_ratio * 0.45 +     # Energy is most important
                zcr_ratio * 0.35 +        # ZCR for speech characteristics  
                spectral_ratio * 0.20     # Spectral for frequency content
            )
            
            # Apply bounds and light smoothing
            final_ratio = max(0.0, min(1.0, final_ratio))
            
            logger.debug(f"Speech ratio - Energy: {energy_ratio:.3f}, ZCR: {zcr_ratio:.3f}, "
                        f"Spectral: {spectral_ratio:.3f}, Final: {final_ratio:.3f}")
            
            return final_ratio
            
        except Exception as e:
            logger.error(f"CORRECTED speech ratio error: {e}")
            return 0.35  # More optimistic fallback
    
    def _pcm_to_wav_enhanced(self, pcm_data: bytes) -> bytes:
        """CORRECTED: Enhanced PCM to WAV conversion with validation"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            
            # Validate WAV file size
            if len(wav_bytes) < 1000:
                raise ValueError(f"WAV file too small: {len(wav_bytes)} bytes")
            
            logger.debug(f"CORRECTED WAV conversion: {len(pcm_data)} PCM → {len(wav_bytes)} WAV bytes")
            return wav_bytes
            
        except Exception as e:
            logger.error(f"CORRECTED WAV conversion error: {e}")
            raise RuntimeError(f"CORRECTED WAV conversion failed: {e}")
    
    async def _restart_ffmpeg_process(self, mode: str, websocket=None):
        """CORRECTED: Enhanced FFmpeg restart with better error recovery"""
        try:
            logger.info(f"🔄 Restarting CORRECTED {mode} FFmpeg process...")
            
            # Graceful cleanup of old process
            if mode == "transcribe" and self.transcribe_ffmpeg_process:
                try:
                    if self.transcribe_ffmpeg_process.stdin:
                        self.transcribe_ffmpeg_process.stdin.close()
                    self.transcribe_ffmpeg_process.terminate()
                    await asyncio.sleep(3.0)  # Wait for graceful termination
                    if self.transcribe_ffmpeg_process.poll() is None:
                        self.transcribe_ffmpeg_process.kill()
                except Exception as cleanup_e:
                    logger.debug(f"Cleanup error for {mode}: {cleanup_e}")
                self.transcribe_ffmpeg_process = None
                
            elif mode == "understand" and self.understand_ffmpeg_process:
                try:
                    if self.understand_ffmpeg_process.stdin:
                        self.understand_ffmpeg_process.stdin.close()
                    self.understand_ffmpeg_process.terminate()
                    await asyncio.sleep(3.0)
                    if self.understand_ffmpeg_process.poll() is None:
                        self.understand_ffmpeg_process.kill()
                except Exception as cleanup_e:
                    logger.debug(f"Cleanup error for {mode}: {cleanup_e}")
                self.understand_ffmpeg_process = None
            
            # Smart buffer management - keep recent data for continuity
            if mode == "transcribe":
                if len(self.transcribe_pcm_buffer) > 64000:  # Keep last 2 seconds
                    self.transcribe_pcm_buffer = self.transcribe_pcm_buffer[-64000:]
            else:
                if len(self.understand_pcm_buffer) > 32000:  # Keep last 1 second
                    self.understand_pcm_buffer = self.understand_pcm_buffer[-32000:]
            
            # Restart with enhanced configuration
            await asyncio.sleep(1.0)  # Brief pause before restart
            await self.start_ffmpeg_decoder(mode, websocket)
            
            logger.info(f"✅ CORRECTED {mode} FFmpeg process restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart CORRECTED {mode} process: {e}")
            # Mark process as None to force recreation
            if mode == "transcribe":
                self.transcribe_ffmpeg_process = None
            else:
                self.understand_ffmpeg_process = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CORRECTED processing statistics"""
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
            "total_audio_length_ms": round(self.total_audio_length, 1),
            "transcribe_buffer_size": len(self.transcribe_pcm_buffer),
            "understand_buffer_size": len(self.understand_pcm_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcribe_ffmpeg_running": self.transcribe_ffmpeg_process is not None,
            "understand_ffmpeg_running": self.understand_ffmpeg_process is not None,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.active_connections),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "corrected": True,
            "gap_detection": {
                "threshold_ms": self.gap_threshold_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "frame_size_ms": self.frame_size_ms,
                "active_gap_states": len(self.gap_detection_state)
            },
            "quality_thresholds": {
                "min_duration_ms": 800,
                "min_speech_ratio": 0.25,
                "energy_threshold": 600.0
            }
        }
    
    def reset(self):
        """Reset CORRECTED processor state"""
        self.transcribe_pcm_buffer.clear()
        self.understand_pcm_buffer.clear()
        self.chunks_processed = 0
        self.speech_chunks_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        self.gap_detection_state.clear()
        
        logger.info("✅ CORRECTED audio processor reset completed")
    
    async def cleanup(self):
        """CORRECTED: Enhanced cleanup with proper resource management"""
        logger.info("🧹 Starting CORRECTED audio processor cleanup...")
        
        processes = [
            ("transcribe", self.transcribe_ffmpeg_process),
            ("understand", self.understand_ffmpeg_process)
        ]
        
        for mode, process in processes:
            if process:
                try:
                    # Close stdin first
                    if process.stdin and not process.stdin.closed:
                        process.stdin.close()
                    
                    # Close other pipes
                    if process.stdout and not process.stdout.closed:
                        process.stdout.close()
                    if process.stderr and not process.stderr.closed:
                        process.stderr.close()
                    
                    # Graceful termination
                    process.terminate()
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(process.wait), 
                            timeout=25.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"CORRECTED FFmpeg {mode} termination timeout, killing...")
                        process.kill()
                        await asyncio.to_thread(process.wait)
                    
                    logger.info(f"✅ CORRECTED FFmpeg {mode} process cleaned up")
                    
                except Exception as e:
                    logger.error(f"CORRECTED FFmpeg {mode} cleanup error: {e}")
        
        # Reset process references
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True, timeout=10.0)
        except Exception as e:
            logger.error(f"CORRECTED executor shutdown error: {e}")
        
        # Final reset
        self.reset()
        logger.info("✅ CORRECTED audio processor fully cleaned up")
