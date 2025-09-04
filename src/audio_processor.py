# FIXED ENHANCED AUDIO PROCESSOR - CORRECTED SLICE INDICES AND METHOD NAMES
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
import time

logger = logging.getLogger(__name__)

class EnhancedAudioProcessor:
    """FIXED: Enhanced WebM audio processing with corrected slice indices"""
    
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
        
        # Enhanced FFmpeg streaming with better error recovery
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # Enhanced PCM buffers with sliding window
        self.transcribe_pcm_buffer = bytearray()
        self.understand_pcm_buffer = bytearray()
        
        # FIXED: Enhanced thresholds converted to integers
        self.transcribe_threshold = int(sample_rate * 0.5)  # 500ms for transcription
        self.understand_threshold = int(sample_rate * 1.0)  # 1 second for understanding
        
        # Voice Activity Detection (optional)
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
            self.vad_enabled = True
            logger.info("✅ WebRTC VAD initialized")
        except ImportError:
            self.vad = None
            self.vad_enabled = False
            logger.warning("⚠️ WebRTC VAD not available, using fallback detection")
        except Exception as e:
            self.vad = None
            self.vad_enabled = False
            logger.warning(f"⚠️ WebRTC VAD initialization failed: {e}")
        
        # Enhanced statistics and monitoring
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # Thread pool with better resource management
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AudioProc")
        
        # Connection monitoring
        self.active_connections = set()
        self.last_activity = {}
        
        logger.info(f"✅ ENHANCED AudioProcessor initialized: {sample_rate}Hz, {channels}ch, VAD: {self.vad_enabled}")
    
    async def start_ffmpeg_decoder(self, mode: str, websocket=None):
        """Enhanced FFmpeg process startup with better error recovery"""
        try:
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # Enhanced FFmpeg configuration
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
                # Start enhanced background PCM reader
                asyncio.create_task(self._read_pcm_output_enhanced(mode, websocket))
                logger.info("✅ Enhanced FFmpeg transcription decoder started")
            else:
                self.understand_ffmpeg_process = ffmpeg_process
                # Start enhanced background PCM reader
                asyncio.create_task(self._read_pcm_output_enhanced(mode, websocket))
                logger.info("✅ Enhanced FFmpeg understanding decoder started")
                
        except Exception as e:
            logger.error(f"Failed to start enhanced FFmpeg decoder for {mode}: {e}")
            raise RuntimeError(f"Enhanced FFmpeg {mode} initialization failed: {e}")
    
    async def _read_pcm_output_enhanced(self, mode: str, websocket=None):
        """Enhanced background task to read PCM data with better error handling"""
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
        max_errors = 5
        
        try:
            while ffmpeg_process and ffmpeg_process.stdout:
                try:
                    # Enhanced PCM data reading with timeout
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor, 
                            ffmpeg_process.stdout.read, 
                            8192  # Larger buffer for better performance
                        ),
                        timeout=5.0  # 5 second timeout
                    )
                    
                    if not chunk:
                        logger.warning(f"Enhanced FFmpeg {mode} stdout closed")
                        break
                    
                    # Add to PCM buffer with size management
                    pcm_buffer.extend(chunk)
                    
                    # Prevent buffer overflow (keep last 30 seconds)
                    max_buffer_size = self.sample_rate * 30 * 2  # 30 seconds of 16-bit audio
                    if len(pcm_buffer) > max_buffer_size:
                        excess = len(pcm_buffer) - max_buffer_size
                        del pcm_buffer[:excess]
                        logger.debug(f"Trimmed {mode} buffer by {excess} bytes")
                    
                    # Update connection activity
                    if conn_id:
                        self.last_activity[conn_id] = time.time()
                    
                    consecutive_errors = 0  # Reset error counter on success
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Enhanced FFmpeg {mode} read timeout")
                    consecutive_errors += 1
                except Exception as e:
                    logger.error(f"Error reading enhanced FFmpeg {mode} output: {e}")
                    consecutive_errors += 1
                
                # Break if too many consecutive errors
                if consecutive_errors >= max_errors:
                    logger.error(f"Too many errors reading {mode} output, stopping")
                    break
                    
        except Exception as e:
            logger.error(f"Enhanced PCM reader task failed for {mode}: {e}")
        finally:
            # Cleanup connection tracking
            if conn_id and conn_id in self.active_connections:
                self.active_connections.discard(conn_id)
                self.last_activity.pop(conn_id, None)
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """Enhanced transcription processing with conversation context"""
        start_time = time.time()
        
        try:
            # Validate input
            if not webm_data or len(webm_data) < 10:
                return None
            
            if not self.transcribe_ffmpeg_process:
                await self.start_ffmpeg_decoder("transcribe", websocket)
            
            # Send WebM chunk to FFmpeg stdin with error recovery
            if self.transcribe_ffmpeg_process and self.transcribe_ffmpeg_process.stdin:
                try:
                    self.transcribe_ffmpeg_process.stdin.write(webm_data)
                    self.transcribe_ffmpeg_process.stdin.flush()
                    await asyncio.sleep(0.005)  # Small delay for processing
                    
                    self.chunks_processed += 1
                    
                    # Check if we have enough PCM data to process
                    if len(self.transcribe_pcm_buffer) >= self.transcribe_threshold:
                        result = self._process_pcm_buffer_enhanced("transcribe", websocket)
                        
                        # Record processing time
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg transcription process ended, restarting...")
                    await self._restart_ffmpeg_process("transcribe", websocket)
                except Exception as e:
                    logger.error(f"Error writing to enhanced FFmpeg transcription: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced WebM transcription chunk processing error: {e}")
            return {"error": f"Enhanced processing failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """Enhanced understanding processing with conversation context"""
        start_time = time.time()
        
        try:
            # Validate input
            if not webm_data or len(webm_data) < 10:
                return None
            
            if not self.understand_ffmpeg_process:
                await self.start_ffmpeg_decoder("understand", websocket)
            
            # Send WebM chunk to FFmpeg stdin with error recovery
            if self.understand_ffmpeg_process and self.understand_ffmpeg_process.stdin:
                try:
                    self.understand_ffmpeg_process.stdin.write(webm_data)
                    self.understand_ffmpeg_process.stdin.flush()
                    await asyncio.sleep(0.005)  # Small delay for processing
                    
                    # Check if we have enough PCM data to process
                    if len(self.understand_pcm_buffer) >= self.understand_threshold:
                        result = self._process_pcm_buffer_enhanced("understand", websocket)
                        
                        # Record processing time
                        processing_time = time.time() - start_time
                        self.processing_times.append(processing_time)
                        
                        return result
                        
                except BrokenPipeError:
                    logger.warning("FFmpeg understanding process ended, restarting...")
                    await self._restart_ffmpeg_process("understand", websocket)
                except Exception as e:
                    logger.error(f"Error writing to enhanced FFmpeg understanding: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced WebM understanding chunk processing error: {e}")
            return {"error": f"Enhanced processing failed: {str(e)}"}
    
    def _process_pcm_buffer_enhanced(self, mode: str, websocket=None) -> Dict[str, Any]:
        """FIXED: Enhanced PCM buffer processing with corrected slice indices"""
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
            
            # FIXED: Extract audio data with proper integer indices
            overlap_samples = int(self.sample_rate * 0.1)  # 100ms overlap
            extract_end = min(threshold + overlap_samples, len(pcm_buffer))  # Ensure we don't exceed buffer
            audio_data = bytes(pcm_buffer[:extract_end])
            del pcm_buffer[:threshold]  # Keep overlap in buffer
            
            # Create WAV file from PCM data
            wav_bytes = self._pcm_to_wav_enhanced(audio_data)
            
            # Calculate duration
            duration_ms = (len(audio_data) / 2) / self.sample_rate * 1000  # 16-bit PCM
            
            self.total_audio_length += duration_ms
            
            # Enhanced speech detection
            speech_ratio = self._estimate_speech_ratio_enhanced(audio_data)
            
            # Update statistics
            if speech_ratio > 0.05:  # Lowered threshold
                self.speech_chunks_detected += 1
            
            # Get conversation context if available
            conversation_context = ""
            if self.conversation_manager and websocket:
                try:
                    conversation_context = self.conversation_manager.get_conversation_context(websocket)
                except:
                    conversation_context = ""  # Fallback on error
            
            logger.info(f"🎤 Enhanced processed {mode} PCM audio: {duration_ms:.0f}ms (Total: {self.total_audio_length:.1f}ms) - Speech: {speech_ratio:.3f} - Context: {bool(conversation_context)}")
            
            return {
                "audio_data": wav_bytes,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": mode,
                "processed_at": time.time(),
                "enhanced": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"Enhanced PCM buffer processing error for {mode}: {e}")
            return {"error": f"Enhanced processing failed: {str(e)}"}
    
    def _estimate_speech_ratio_enhanced(self, pcm_data: bytes) -> float:
        """Enhanced speech detection with improved fallbacks"""
        try:
            # Convert PCM to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0
            
            # Method 1: WebRTC VAD (if available)
            if self.vad_enabled and self.vad:
                try:
                    # VAD works with specific frame sizes (10ms, 20ms, 30ms)
                    frame_duration = 30  # 30ms frames
                    frame_size = int(self.sample_rate * frame_duration / 1000)
                    
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
                        vad_ratio = speech_frames / total_frames
                        logger.debug(f"WebRTC VAD ratio: {vad_ratio:.3f}")
                        return vad_ratio
                except Exception as e:
                    logger.debug(f"WebRTC VAD failed, using fallback: {e}")
            
            # Method 2: Enhanced energy-based detection
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            
            # Adaptive threshold
            silence_threshold = 150  # Base threshold
            
            # Method 3: Frame-based analysis
            frame_size = int(self.sample_rate * 0.025)  # 25ms frames
            hop_size = int(self.sample_rate * 0.010)   # 10ms hop
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_array) - frame_size, hop_size):
                frame = audio_array[i:i + frame_size]
                frame_rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
                
                if frame_rms > silence_threshold:
                    # Additional check: zero crossing rate
                    zero_crossings = np.sum(np.diff(np.signbit(frame)))
                    zcr_normalized = zero_crossings / len(frame)
                    
                    # Speech typically has ZCR between 0.01 and 0.5
                    if 0.01 <= zcr_normalized <= 0.5:
                        speech_frames += 1
                total_frames += 1
            
            frame_ratio = speech_frames / max(total_frames, 1) if total_frames > 0 else 0.0
            
            # Combine methods
            rms_ratio = min(1.0, rms_energy / 1000.0)  # Normalize to 0-1
            
            # Weighted combination
            final_ratio = (
                frame_ratio * 0.6 +          # Energy-based frames (60%)
                rms_ratio * 0.4              # Overall energy (40%)
            )
            
            logger.debug(f"Enhanced speech detection: Frame={frame_ratio:.3f}, RMS={rms_ratio:.3f}, Final={final_ratio:.3f}")
            
            return min(1.0, final_ratio)
            
        except Exception as e:
            logger.error(f"Enhanced speech ratio estimation error: {e}")
            return 0.3  # Conservative fallback value
    
    def _pcm_to_wav_enhanced(self, pcm_data: bytes) -> bytes:
        """Enhanced PCM to WAV conversion with better error handling"""
        try:
            wav_io = io.BytesIO()
            
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_data)
            
            wav_bytes = wav_io.getvalue()
            logger.debug(f"Enhanced WAV conversion: {len(pcm_data)} PCM bytes → {len(wav_bytes)} WAV bytes")
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Enhanced PCM to WAV conversion error: {e}")
            raise RuntimeError(f"Enhanced WAV conversion failed: {e}")
    
    async def _restart_ffmpeg_process(self, mode: str, websocket=None):
        """Enhanced FFmpeg process restart with exponential backoff"""
        try:
            logger.info(f"🔄 Restarting enhanced {mode} FFmpeg process...")
            
            # Clean up old process
            if mode == "transcribe" and self.transcribe_ffmpeg_process:
                try:
                    self.transcribe_ffmpeg_process.terminate()
                    await asyncio.sleep(1)
                except:
                    pass
                self.transcribe_ffmpeg_process = None
            elif mode == "understand" and self.understand_ffmpeg_process:
                try:
                    self.understand_ffmpeg_process.terminate()
                    await asyncio.sleep(1)
                except:
                    pass
                self.understand_ffmpeg_process = None
            
            # Wait before restart
            await asyncio.sleep(0.5)
            
            # Restart process
            await self.start_ffmpeg_decoder(mode, websocket)
            
        except Exception as e:
            logger.error(f"Failed to restart enhanced {mode} process: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced audio processing statistics"""
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
            "transcribe_pcm_buffer_size": len(self.transcribe_pcm_buffer),
            "understand_pcm_buffer_size": len(self.understand_pcm_buffer),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcribe_ffmpeg_running": self.transcribe_ffmpeg_process is not None,
            "understand_ffmpeg_running": self.understand_ffmpeg_process is not None,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.active_connections),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "conversation_manager_enabled": self.conversation_manager is not None,
            "enhanced_features": [
                "✅ Fixed Slice Indices Processing",
                "✅ Enhanced Speech Detection", 
                "✅ Conversation Context Integration",
                "✅ Enhanced Buffer Management",
                "✅ Automatic Process Recovery",
                "✅ Performance Monitoring"
            ]
        }
    
    def reset(self):
        """Enhanced reset with comprehensive cleanup"""
        self.transcribe_pcm_buffer.clear()
        self.understand_pcm_buffer.clear()
        self.chunks_processed = 0
        self.speech_chunks_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        
        logger.info("✅ Enhanced audio processor reset completed")
    
    async def cleanup(self):
        """FIXED: Renamed from cleanup_enhanced to cleanup for compatibility"""
        logger.info("🧹 Starting enhanced audio processor cleanup...")
        
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
                    
                    # Graceful termination
                    process.terminate()
                    try:
                        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await asyncio.to_thread(process.wait)
                    
                    logger.info(f"✅ Enhanced FFmpeg {mode} process cleaned up")
                except Exception as e:
                    logger.error(f"Enhanced FFmpeg {mode} cleanup error: {e}")
        
        self.transcribe_ffmpeg_process = None
        self.understand_ffmpeg_process = None
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=10)
        
        # Final cleanup
        self.reset()
        
        logger.info("✅ Enhanced audio processor fully cleaned up")

# Backwards compatibility alias
AudioProcessor = EnhancedAudioProcessor
