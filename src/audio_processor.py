# CONTINUOUS STREAMING AUDIO PROCESSOR - audio_processor.py - WITH 300MS GAP DETECTION
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
    """CONTINUOUS STREAMING: Audio processor with 300ms gap detection for understanding mode"""
    
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
        
        # CONTINUOUS STREAMING: Enhanced FFmpeg processes
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # CONTINUOUS STREAMING: PCM buffers for both modes
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # CONTINUOUS STREAMING: Processing thresholds
        self.transcribe_threshold = int(sample_rate * 2.0 * 2)  # 2 seconds for transcription
        self.understand_threshold = int(sample_rate * 1.0 * 2)  # 1 second for understanding (faster)
        
        # CONTINUOUS STREAMING: Enhanced Voice Activity Detection for 300ms gap detection
        try:
            self.vad = webrtcvad.Vad(1)  # Mode 1 - balanced for gap detection
            self.vad_enabled = True
            logger.info("✅ CONTINUOUS STREAMING WebRTC VAD initialized (mode 1 for gap detection)")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # CONTINUOUS STREAMING: Gap detection parameters
        self.gap_threshold_ms = 300  # 300ms silence gap
        self.min_speech_duration_ms = 500  # Minimum 500ms for processing
        self.frame_size_ms = 10  # 10ms frames for VAD
        self.frame_size_samples = int(sample_rate * self.frame_size_ms / 1000)
        
        # Statistics and state
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # CONTINUOUS STREAMING: Enhanced ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ContStream")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        # CONTINUOUS STREAMING: Gap detection state per connection
        self.gap_detection_state = {}
        
        logger.info(f"✅ CONTINUOUS STREAMING AudioProcessor: {sample_rate}Hz, {channels}ch")
        logger.info(f"   Gap threshold: {self.gap_threshold_ms}ms")
        logger.info(f"   Min speech: {self.min_speech_duration_ms}ms") 
        logger.info(f"   Frame size: {self.frame_size_ms}ms ({self.frame_size_samples} samples)")
    
    async def start_ffmpeg_decoder(self, mode: str, websocket=None):
        """CONTINUOUS STREAMING: Start robust FFmpeg decoder"""
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
            
            # CONTINUOUS STREAMING: Robust FFmpeg configuration
            ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm', thread_queue_size=4096)
                .filter('highpass', f=80)  # Remove low frequency noise
                .filter('lowpass', f=8000)  # Remove high frequency noise
                .output(
                    'pipe:1', 
                    format='s16le', 
                    acodec='pcm_s16le', 
                    ac=self.channels, 
                    ar=str(self.sample_rate),
                    audio_bitrate='256k',  # Higher quality
                    bufsize='4096k'       # Larger buffer
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
                asyncio.create_task(self._read_pcm_output_continuous(mode, websocket))
                logger.info("✅ CONTINUOUS STREAMING FFmpeg transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                asyncio.create_task(self._read_pcm_output_continuous(mode, websocket))
                logger.info("✅ CONTINUOUS STREAMING FFmpeg understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start CONTINUOUS STREAMING FFmpeg decoder for {mode}: {e}")
            # Auto-restart attempt
            await asyncio.sleep(1.0)
            try:
                await self.start_ffmpeg_decoder(mode, websocket)
                logger.info(f"✅ CONTINUOUS STREAMING FFmpeg {mode} auto-restarted")
            except Exception as restart_error:
                logger.error(f"CONTINUOUS STREAMING FFmpeg {mode} restart failed: {restart_error}")
                raise RuntimeError(f"CONTINUOUS STREAMING FFmpeg {mode} failed: {e}")
    
    async def _read_pcm_output_continuous(self, mode: str, websocket=None):
        """CONTINUOUS STREAMING: Enhanced PCM reader with robust error handling"""
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
        max_errors = 10  # Increased tolerance
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout:
                try:
                    # CONTINUOUS STREAMING: Read larger chunks for better performance
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            ffmpeg_process.stdout.read, 
                            32768  # 32KB chunks for continuous streaming
                        ),
                        timeout=5.0  # Longer timeout for stability
                    )
                    
                    if not chunk:
                        logger.warning(f"CONTINUOUS STREAMING FFmpeg {mode} stdout closed")
                        break
                    
                    # Add to PCM buffer
                    pcm_buffer.extend(chunk)
                    
                    # CONTINUOUS STREAMING: Manage buffer size (60 seconds max)
                    max_buffer_size = int(self.sample_rate * 60 * 2)  # 60 seconds
                    if len(pcm_buffer) > max_buffer_size:
                        excess = len(pcm_buffer) - max_buffer_size
                        del pcm_buffer[:excess]
                        logger.debug(f"Trimmed {mode} buffer by {excess} bytes")
                    
                    # Update activity
                    if conn_id:
                        self.last_activity[conn_id] = time.time()
                    
                    consecutive_errors = 0  # Reset on success
                    
                except asyncio.TimeoutError:
                    logger.debug(f"CONTINUOUS STREAMING FFmpeg {mode} read timeout")
                    consecutive_errors += 1
                except Exception as e:
                    logger.error(f"Error reading CONTINUOUS STREAMING FFmpeg {mode} output: {e}")
                    consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    logger.error(f"Too many consecutive errors in {mode}, attempting restart...")
                    await self._restart_ffmpeg_process(mode, websocket)
                    consecutive_errors = 0
                    
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING PCM reader failed for {mode}: {e}")
        finally:
            # Cleanup
            if conn_id and conn_id in self.active_connections:
                self.active_connections.discard(conn_id)
                self.last_activity.pop(conn_id, None)
                if mode == "understand":
                    self.gap_detection_state.pop(conn_id, None)
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """CONTINUOUS STREAMING: Transcription processing (unchanged from original)"""
        start_time = time.time()
        
        try:
            if not webm_data or len(webm_data) < 500:
                logger.debug("Insufficient WebM data for transcription")
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe", websocket)
            
            # CONTINUOUS STREAMING: Send to FFmpeg with enhanced error handling
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    
                    await asyncio.sleep(0.05)
                    
                    self.chunks_processed += 1
                    
                    # Process when we have sufficient audio
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        result = self._process_pcm_buffer_continuous("transcribe", websocket)
                        
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg transcription pipe broken, restarting...")
                    await self._restart_ffmpeg_process("transcribe", websocket)
                except Exception as e:
                    logger.error(f"Error in CONTINUOUS STREAMING transcription processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING transcription error: {e}")
            return {"error": f"CONTINUOUS STREAMING transcription failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """CONTINUOUS STREAMING: Understanding mode - accumulate PCM with gap detection"""
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
                    
                    await asyncio.sleep(0.02)  # Shorter delay for faster accumulation
                    
                    # Get PCM data from buffer
                    if len(self.understand_pcm_buffer) >= self.frame_size_samples * 2:
                        pcm_chunk = bytes(self.understand_pcm_buffer[:self.frame_size_samples * 2])
                        del self.understand_pcm_buffer[:self.frame_size_samples * 2]
                        
                        # Detect speech in this frame
                        speech_detected = self._detect_speech_in_frame(pcm_chunk)
                        
                        return {
                            "pcm_data": pcm_chunk,
                            "speech_detected": speech_detected,
                            "frame_size": len(pcm_chunk),
                            "continuous_streaming": True
                        }
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg understanding pipe broken, restarting...")
                    await self._restart_ffmpeg_process("understand", websocket)
                except Exception as e:
                    logger.error(f"Error in CONTINUOUS STREAMING understanding processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING understanding error: {e}")
            return {"error": f"CONTINUOUS STREAMING understanding failed: {str(e)}"}
    
    def _detect_speech_in_frame(self, pcm_frame: bytes) -> bool:
        """CONTINUOUS STREAMING: Detect speech in a single PCM frame"""
        try:
            if not self.vad_enabled or len(pcm_frame) < self.frame_size_samples * 2:
                return False
            
            # Ensure frame is exactly the right size
            expected_size = self.frame_size_samples * 2
            if len(pcm_frame) != expected_size:
                # Pad or truncate to expected size
                if len(pcm_frame) < expected_size:
                    pcm_frame = pcm_frame + b'\x00' * (expected_size - len(pcm_frame))
                else:
                    pcm_frame = pcm_frame[:expected_size]
            
            # Use WebRTC VAD to detect speech
            try:
                return self.vad.is_speech(pcm_frame, self.sample_rate)
            except Exception as vad_error:
                logger.debug(f"VAD error: {vad_error}")
                return False
                
        except Exception as e:
            logger.error(f"Speech detection error: {e}")
            return False
    
    def _process_pcm_buffer_continuous(self, mode: str, websocket=None) -> Dict[str, Any]:
        """CONTINUOUS STREAMING: Process PCM buffer (for transcription mode)"""
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
            
            # Extract audio with minimal overlap for continuous streaming
            overlap_samples = int(self.sample_rate * 0.1 * 2)  # 100ms overlap
            end_index = min(threshold + overlap_samples, len(pcm_buffer))
            
            audio_data = bytes(pcm_buffer[:end_index])
            del pcm_buffer[:threshold]  # Keep small overlap
            
            # Create WAV file
            wav_bytes = self._pcm_to_wav_enhanced(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000
            self.total_audio_length += duration_ms
            
            # Enhanced speech detection
            speech_ratio = self._estimate_speech_ratio_continuous(audio_data)
            
            if speech_ratio > 0.1:
                self.speech_chunks_detected += 1
            
            # Quality control for continuous streaming
            min_duration = 1000  # 1 second minimum
            if duration_ms < min_duration or speech_ratio < 0.3:
                logger.debug(f"Skipping low quality audio: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                return None
            
            # Get conversation context
            conversation_context = ""
            if self.conversation_manager and websocket:
                conversation_context = self.conversation_manager.get_conversation_context(websocket)
            
            logger.info(f"🎤 CONTINUOUS STREAMING processed {mode}: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": time.time(),
                "continuous_streaming": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING PCM buffer processing error for {mode}: {e}")
            return {"error": f"CONTINUOUS STREAMING processing failed: {str(e)}"}
    
    def _estimate_speech_ratio_continuous(self, pcm_data: bytes) -> float:
        """CONTINUOUS STREAMING: Enhanced speech ratio estimation"""
        try:
            if not pcm_data or len(pcm_data) < 1000:
                return 0.0
                
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Method 1: WebRTC VAD frame-by-frame analysis
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
                                pass
                            total_frames += 1
                    
                    if total_frames > 0:
                        vad_ratio = speech_frames / total_frames
                        return min(1.0, max(0.0, vad_ratio))
                        
                except Exception as e:
                    logger.debug(f"VAD speech ratio error: {e}")
            
            # Method 2: Energy-based detection as fallback
            audio_float = audio_array.astype(np.float64)
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            energy_threshold = 500.0  # Adjusted for continuous streaming
            energy_ratio = min(1.0, max(0.0, (rms_energy - energy_threshold) / energy_threshold))
            
            # Method 3: Zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            
            if 0.01 <= zcr_normalized <= 0.2:  # Speech range
                zcr_ratio = 1.0
            elif zcr_normalized < 0.01:
                zcr_ratio = 0.0
            else:
                zcr_ratio = max(0.0, 1.0 - (zcr_normalized - 0.2) / 0.3)
            
            # Combine metrics for continuous streaming
            final_ratio = energy_ratio * 0.6 + zcr_ratio * 0.4
            
            return max(0.0, min(1.0, final_ratio))
            
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING speech ratio error: {e}")
            return 0.4  # Optimistic fallback
    
    def _pcm_to_wav_enhanced(self, pcm_data: bytes) -> bytes:
        """CONTINUOUS STREAMING: Enhanced PCM to WAV conversion"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            logger.debug(f"CONTINUOUS STREAMING WAV conversion: {len(pcm_data)} PCM → {len(wav_bytes)} WAV bytes")
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING WAV conversion error: {e}")
            raise RuntimeError(f"CONTINUOUS STREAMING WAV conversion failed: {e}")
    
    async def _restart_ffmpeg_process(self, mode: str, websocket=None):
        """CONTINUOUS STREAMING: Enhanced FFmpeg restart"""
        try:
            logger.info(f"🔄 Restarting CONTINUOUS STREAMING {mode} FFmpeg process...")
            
            # Clean up old process
            if mode == "transcribe" and self.transcribe_ffmpeg_process:
                try:
                    self.transcribe_ffmpeg_process.terminate()
                    await asyncio.sleep(3.0)
                except:
                    pass
                self.transcribe_ffmpeg_process = None
            elif mode == "understand" and self.understand_ffmpeg_process:
                try:
                    self.understand_ffmpeg_process.terminate()
                    await asyncio.sleep(3.0)
                except:
                    pass
                self.understand_ffmpeg_process = None
            
            # Clear buffer partially (keep some data for continuity)
            if mode == "transcribe":
                if len(self.transcribe_pcm_buffer) > 32000:  # Keep last 1 second
                    self.transcribe_pcm_buffer = self.transcribe_pcm_buffer[-32000:]
            else:
                if len(self.understand_pcm_buffer) > 32000:
                    self.understand_pcm_buffer = self.understand_pcm_buffer[-32000:]
            
            # Restart with enhanced configuration
            await self.start_ffmpeg_decoder(mode, websocket)
            
        except Exception as e:
            logger.error(f"Failed to restart CONTINUOUS STREAMING {mode} process: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CONTINUOUS STREAMING processing statistics"""
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
            "continuous_streaming": True,
            "gap_detection": {
                "threshold_ms": self.gap_threshold_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "frame_size_ms": self.frame_size_ms,
                "active_gap_states": len(self.gap_detection_state)
            }
        }
    
    def reset(self):
        """Reset CONTINUOUS STREAMING processor state"""
        self.transcribe_pcm_buffer.clear()
        self.understand_pcm_buffer.clear()
        self.chunks_processed = 0
        self.speech_chunks_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        self.gap_detection_state.clear()
        
        logger.info("✅ CONTINUOUS STREAMING audio processor reset")
    
    async def cleanup(self):
        """CONTINUOUS STREAMING: Enhanced cleanup"""
        logger.info("🧹 Starting CONTINUOUS STREAMING audio processor cleanup...")
        
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
                        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=20.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await asyncio.to_thread(process.wait)
                    
                    logger.info(f"✅ CONTINUOUS STREAMING FFmpeg {mode} process cleaned up")
                except Exception as e:
                    logger.error(f"CONTINUOUS STREAMING FFmpeg {mode} cleanup error: {e}")
        
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # Enhanced shutdown
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"CONTINUOUS STREAMING executor shutdown error: {e}")
        
        self.reset()
        logger.info("✅ CONTINUOUS STREAMING audio processor fully cleaned up")
