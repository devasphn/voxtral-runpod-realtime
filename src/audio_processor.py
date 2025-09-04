# FINAL COMPLETE FIX - audio_processor.py - ENHANCED FOR HUMAN SPEECH
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
    """FINAL FIXED: Audio processor optimized for human speech recognition"""
    
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
        
        # FINAL FIX: Enhanced FFmpeg processes for better speech processing
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # FINAL FIX: PCM buffers with enhanced management for speech
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # FINAL FIX: Optimized processing thresholds for human speech
        self.transcribe_threshold = int(sample_rate * 2.0 * 2)  # 2 seconds for better speech quality
        self.understand_threshold = int(sample_rate * 3.0 * 2)  # 3 seconds for understanding
        
        # FINAL FIX: Enhanced Voice Activity Detection for human speech
        try:
            self.vad = webrtcvad.Vad(0)  # Least aggressive for human speech
            self.vad_enabled = True
            logger.info("✅ Enhanced WebRTC VAD initialized for human speech (mode 0)")
        except:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available")
        
        # Statistics
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # FINAL FIX: ThreadPoolExecutor for enhanced processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SpeechProc")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        # FINAL FIX: Optimized audio quality control for human speech
        self.min_speech_duration = 1.0  # Minimum 1 second for good speech recognition
        self.speech_threshold = 0.3     # Lower threshold for human speech detection
        
        logger.info(f"✅ FINAL FIXED AudioProcessor for Human Speech: {sample_rate}Hz, {channels}ch, VAD: {self.vad_enabled}")
        logger.info(f"   Transcribe threshold: {self.transcribe_threshold} bytes ({self.transcribe_threshold/(sample_rate*2):.1f}s)")
        logger.info(f"   Understand threshold: {self.understand_threshold} bytes ({self.understand_threshold/(sample_rate*2):.1f}s)")
    
    async def start_ffmpeg_decoder(self, mode: str, websocket=None):
        """FINAL FIX: Start FFmpeg decoder optimized for human speech"""
        try:
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # FINAL FIX: Enhanced FFmpeg configuration optimized for human speech
            ffmpeg_process = (
                ffmpeg
                .input('pipe:0', format='webm', thread_queue_size=2048)
                .filter('highpass', f=80)  # Remove low-frequency noise
                .filter('lowpass', f=8000)  # Remove high-frequency noise above speech range
                .output(
                    'pipe:1', 
                    format='s16le', 
                    acodec='pcm_s16le', 
                    ac=self.channels, 
                    ar=str(self.sample_rate),
                    audio_bitrate='192k',  # Higher quality for speech
                    bufsize='2048k'
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
                asyncio.create_task(self._read_pcm_output_enhanced(mode, websocket))
                logger.info("✅ FINAL FIXED FFmpeg speech transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                asyncio.create_task(self._read_pcm_output_enhanced(mode, websocket))
                logger.info("✅ FINAL FIXED FFmpeg speech understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start FINAL FIXED FFmpeg speech decoder for {mode}: {e}")
            raise RuntimeError(f"FINAL FIXED FFmpeg speech {mode} initialization failed: {e}")
    
    async def _read_pcm_output_enhanced(self, mode: str, websocket=None):
        """FINAL FIX: Enhanced PCM reader optimized for human speech"""
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
        max_errors = 3  # Reduced for faster error detection
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout:
                try:
                    # FINAL FIX: Read larger chunks for better speech processing
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            ffmpeg_process.stdout.read, 
                            16384  # Larger chunks for speech
                        ),
                        timeout=3.0  # Faster timeout for speech
                    )
                    
                    if not chunk:
                        logger.warning(f"FINAL FIXED FFmpeg speech {mode} stdout closed")
                        break
                    
                    # Add to PCM buffer
                    pcm_buffer.extend(chunk)
                    
                    # FINAL FIX: Enhanced buffer size management for speech (30 seconds max)
                    max_buffer_size = int(self.sample_rate * 30 * 2)  # 30 seconds for speech
                    if len(pcm_buffer) > max_buffer_size:
                        excess = len(pcm_buffer) - max_buffer_size
                        del pcm_buffer[:excess]
                        logger.debug(f"Trimmed {mode} speech buffer by {excess} bytes")
                    
                    # Update activity
                    if conn_id:
                        self.last_activity[conn_id] = time.time()
                    
                    consecutive_errors = 0  # Reset on success
                    
                except asyncio.TimeoutError:
                    logger.debug(f"FINAL FIXED FFmpeg speech {mode} read timeout")
                    consecutive_errors += 1
                except Exception as e:
                    logger.error(f"Error reading FINAL FIXED FFmpeg speech {mode} output: {e}")
                    consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    logger.error(f"Too many consecutive errors in speech {mode}, stopping")
                    break
                    
        except Exception as e:
            logger.error(f"FINAL FIXED speech PCM reader failed for {mode}: {e}")
        finally:
            # Cleanup
            if conn_id and conn_id in self.active_connections:
                self.active_connections.discard(conn_id)
                self.last_activity.pop(conn_id, None)
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """FINAL FIX: Enhanced transcription processing optimized for human speech"""
        start_time = time.time()
        
        try:
            # FINAL FIX: Better input validation for speech
            if not webm_data or len(webm_data) < 500:  # Minimum for speech data
                logger.debug("Insufficient WebM speech data")
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe", websocket)
            
            # FINAL FIX: Send to FFmpeg with enhanced error handling for speech
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    
                    # FINAL FIX: Optimal processing delay for speech
                    await asyncio.sleep(0.05)
                    
                    self.chunks_processed += 1
                    
                    # FINAL FIX: Process when we have sufficient speech audio
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        result = self._process_pcm_buffer_enhanced("transcribe", websocket)
                        
                        # Record processing time
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg speech transcription pipe broken, restarting...")
                    await self._restart_ffmpeg_process("transcribe", websocket)
                except Exception as e:
                    logger.error(f"Error in FINAL FIXED speech transcription processing: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"FINAL FIXED speech transcription error: {e}")
            return {"error": f"FINAL FIXED speech processing failed: {str(e)}"}
    
    def _process_pcm_buffer_enhanced(self, mode: str, websocket=None) -> Dict[str, Any]:
        """FINAL FIX: Enhanced PCM buffer processing optimized for human speech"""
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
            
            # FINAL FIX: Extract audio with optimal overlap for speech continuity
            overlap_samples = int(self.sample_rate * 0.3 * 2)  # 300ms overlap for speech
            end_index = min(threshold + overlap_samples, len(pcm_buffer))
            
            audio_data = bytes(pcm_buffer[:end_index])
            del pcm_buffer[:threshold]  # Keep overlap for speech continuity
            
            # Create enhanced WAV file for speech
            wav_bytes = self._pcm_to_wav_enhanced(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000
            self.total_audio_length += duration_ms
            
            # FINAL FIX: Enhanced speech detection optimized for human voice
            speech_ratio = self._estimate_speech_ratio_enhanced(audio_data)
            
            # FINAL FIX: Quality control optimized for human speech
            if speech_ratio > 0.1:
                self.speech_chunks_detected += 1
            
            # Only process if meets minimum quality thresholds for speech
            min_duration = self.min_speech_duration * 1000  # Convert to ms
            if duration_ms < min_duration or speech_ratio < self.speech_threshold:
                logger.debug(f"Skipping low quality speech: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                return None
            
            # Get conversation context
            conversation_context = ""
            if self.conversation_manager and websocket:
                conversation_context = self.conversation_manager.get_conversation_context(websocket)
            
            logger.info(f"🎤 FINAL FIXED processed speech {mode}: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": time.time(),
                "final_fixed": True,
                "speech_optimized": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"FINAL FIXED speech PCM buffer processing error for {mode}: {e}")
            return {"error": f"FINAL FIXED speech processing failed: {str(e)}"}
    
    def _estimate_speech_ratio_enhanced(self, pcm_data: bytes) -> float:
        """FINAL FIX: Enhanced speech detection specifically optimized for human voice"""
        try:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Method 1: Enhanced WebRTC VAD optimized for human speech
            if self.vad_enabled and self.vad:
                try:
                    frame_size = int(self.sample_rate * 0.010)  # 10ms frames for better sensitivity
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
                                pass  # Skip invalid frames
                            total_frames += 1
                    
                    if total_frames > 0:
                        vad_ratio = speech_frames / total_frames
                        logger.debug(f"Enhanced VAD speech ratio: {vad_ratio:.3f}")
                        return vad_ratio
                        
                except Exception as e:
                    logger.debug(f"Enhanced VAD error: {e}")
            
            # Method 2: Enhanced energy-based detection for human speech
            audio_float = audio_array.astype(np.float64)
            
            # RMS energy with speech-optimized threshold
            rms_energy = np.sqrt(np.mean(audio_float ** 2))
            energy_threshold = 300.0  # Optimized for human speech
            energy_ratio = min(1.0, max(0.0, (rms_energy - energy_threshold) / energy_threshold))
            
            # Method 3: Enhanced Zero crossing rate for human speech
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            
            # Human speech typically has ZCR between 0.01 and 0.15
            if 0.005 <= zcr_normalized <= 0.2:
                zcr_ratio = 1.0
            elif zcr_normalized < 0.005:
                zcr_ratio = 0.0  # Too low - likely silence
            else:
                zcr_ratio = max(0.0, 1.0 - (zcr_normalized - 0.2) / 0.3)
            
            # Method 4: Enhanced spectral analysis for human speech
            try:
                fft = np.fft.rfft(audio_float)
                magnitude = np.abs(fft)
                freqs = np.fft.rfftfreq(len(audio_float), 1.0/self.sample_rate)
                
                if np.sum(magnitude) > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                    # Human speech typically between 100-500 Hz centroid
                    if 80 <= spectral_centroid <= 600:
                        spectral_ratio = 1.0
                    else:
                        spectral_ratio = 0.4
                else:
                    spectral_ratio = 0.0
            except:
                spectral_ratio = 0.5  # Default if spectral analysis fails
            
            # FINAL FIX: Optimized combination for human speech
            final_ratio = (
                energy_ratio * 0.5 +      # Energy is most important for speech
                zcr_ratio * 0.3 +         # ZCR for speech characteristics  
                spectral_ratio * 0.2      # Spectral for human voice frequency content
            )
            
            # Apply bounds and smoothing
            final_ratio = max(0.0, min(1.0, final_ratio))
            
            logger.debug(f"Enhanced speech detection - Energy: {energy_ratio:.3f}, ZCR: {zcr_ratio:.3f}, "
                        f"Spectral: {spectral_ratio:.3f}, Final: {final_ratio:.3f}")
            
            return final_ratio
            
        except Exception as e:
            logger.error(f"FINAL FIXED enhanced speech ratio error: {e}")
            return 0.3  # More optimistic fallback for speech
    
    def _pcm_to_wav_enhanced(self, pcm_data: bytes) -> bytes:
        """FINAL FIX: Enhanced PCM to WAV conversion optimized for speech"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            logger.debug(f"FINAL FIXED speech WAV conversion: {len(pcm_data)} PCM → {len(wav_bytes)} WAV bytes")
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"FINAL FIXED speech WAV conversion error: {e}")
            raise RuntimeError(f"FINAL FIXED speech WAV conversion failed: {e}")
    
    async def _restart_ffmpeg_process(self, mode: str, websocket=None):
        """FINAL FIX: Enhanced FFmpeg restart with proper cleanup for speech"""
        try:
            logger.info(f"🔄 Restarting FINAL FIXED speech {mode} FFmpeg process...")
            
            # Clean up old process
            if mode == "transcribe" and self.transcribe_ffmpeg_process:
                try:
                    self.transcribe_ffmpeg_process.terminate()
                    await asyncio.sleep(2.0)  # More time for speech processes
                except:
                    pass
                self.transcribe_ffmpeg_process = None
            elif mode == "understand" and self.understand_ffmpeg_process:
                try:
                    self.understand_ffmpeg_process.terminate()
                    await asyncio.sleep(2.0)
                except:
                    pass
                self.understand_ffmpeg_process = None
            
            # Clear buffer to start fresh
            if mode == "transcribe":
                self.transcribe_pcm_buffer.clear()
            else:
                self.understand_pcm_buffer.clear()
            
            # Restart with enhanced configuration
            await self.start_ffmpeg_decoder(mode, websocket)
            
        except Exception as e:
            logger.error(f"Failed to restart FINAL FIXED speech {mode} process: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics for speech"""
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
            "final_fixed": True,
            "speech_optimized": True,
            "transcribe_threshold_seconds": self.transcribe_threshold / (self.sample_rate * 2),
            "understand_threshold_seconds": self.understand_threshold / (self.sample_rate * 2),
            "speech_threshold": self.speech_threshold,
            "min_speech_duration": self.min_speech_duration
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
        
        logger.info("✅ FINAL FIXED speech audio processor reset")
    
    async def cleanup(self):
        """FINAL FIX: Enhanced cleanup for speech processing"""
        logger.info("🧹 Starting FINAL FIXED speech audio processor cleanup...")
        
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
                        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=15.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await asyncio.to_thread(process.wait)
                    
                    logger.info(f"✅ FINAL FIXED FFmpeg speech {mode} process cleaned up")
                except Exception as e:
                    logger.error(f"FINAL FIXED FFmpeg speech {mode} cleanup error: {e}")
        
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # FINAL FIX: Enhanced shutdown
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"FINAL FIXED speech executor shutdown error: {e}")
        
        self.reset()
        logger.info("✅ FINAL FIXED speech audio processor fully cleaned up")
