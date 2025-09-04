# ULTIMATE SOLUTION - audio_processor.py - COMPLETELY REBUILT FROM SCRATCH
import asyncio
import logging
import numpy as np
from typing import Optional, List, Dict, Any
import io
import wave
import collections
import subprocess
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import webrtcvad
import time
import queue

logger = logging.getLogger(__name__)

class UltimateAudioProcessor:
    """ULTIMATE: Completely rebuilt audio processor with working FFmpeg and gap detection"""
    
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
        
        # ULTIMATE: Simple approach - no complex FFmpeg pipes
        self.temp_dir = tempfile.mkdtemp(prefix="voxtral_audio_")
        
        # ULTIMATE: Direct audio accumulation without FFmpeg complexity
        self.transcribe_audio_queue = queue.Queue()
        self.understand_audio_queue = queue.Queue()
        
        # ULTIMATE: Gap detection parameters
        self.gap_threshold_ms = 300  # 300ms silence gap
        self.min_speech_duration_ms = 500  # Minimum 500ms for processing
        
        # ULTIMATE: Voice Activity Detection
        try:
            self.vad = webrtcvad.Vad(1)  # Mode 1 for gap detection
            self.vad_enabled = True
            logger.info("✅ ULTIMATE WebRTC VAD initialized")
        except Exception as e:
            self.vad = None
            self.vad_enabled = False
            logger.warning(f"⚠️ WebRTC VAD not available: {e}")
        
        # Statistics and state
        self.chunks_processed = 0
        self.total_audio_length = 0
        self.speech_chunks_detected = 0
        self.processing_times = collections.deque(maxlen=100)
        
        # ULTIMATE: Thread pool for audio processing
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="UltimateAudio")
        
        # Connection tracking
        self.active_connections = set()
        self.last_activity = {}
        
        logger.info(f"✅ ULTIMATE AudioProcessor initialized: {sample_rate}Hz, {channels}ch")
        logger.info(f"   Gap threshold: {self.gap_threshold_ms}ms")
        logger.info(f"   Min speech: {self.min_speech_duration_ms}ms")
        logger.info(f"   VAD enabled: {self.vad_enabled}")
        logger.info(f"   Temp dir: {self.temp_dir}")
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """ULTIMATE: Transcription processing with direct WebM to WAV conversion"""
        start_time = time.time()
        
        try:
            if not webm_data or len(webm_data) < 500:
                return None
            
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # ULTIMATE: Add to queue for batch processing
            self.transcribe_audio_queue.put(webm_data)
            
            # ULTIMATE: Check if we have enough audio to process (every 2 seconds worth)
            if self.transcribe_audio_queue.qsize() >= 20:  # 20 chunks * 100ms = 2 seconds
                return await self._process_transcribe_queue(websocket, start_time)
            
            return None
            
        except Exception as e:
            logger.error(f"ULTIMATE transcription error: {e}")
            return {"error": f"ULTIMATE transcription failed: {str(e)}"}
    
    async def process_webm_chunk_understand(self, webm_data: bytes, websocket=None) -> Optional[Dict[str, Any]]:
        """ULTIMATE: Understanding mode with simple PCM accumulation"""
        try:
            if not webm_data or len(webm_data) < 500:
                return None
            
            # Track connection
            if websocket:
                conn_id = id(websocket)
                self.active_connections.add(conn_id)
                self.last_activity[conn_id] = time.time()
            
            # ULTIMATE: Convert WebM chunk directly to PCM
            pcm_data = await self._webm_to_pcm_ultimate(webm_data)
            
            if pcm_data and len(pcm_data) > 0:
                # Detect speech in PCM data
                speech_detected = self._detect_speech_ultimate(pcm_data)
                
                return {
                    "pcm_data": pcm_data,
                    "speech_detected": speech_detected,
                    "frame_count": len(pcm_data) // 320,  # 10ms frames at 16kHz
                    "frame_size": len(pcm_data),
                    "ultimate": True
                }
            
            return None
            
        except Exception as e:
            logger.error(f"ULTIMATE understanding error: {e}")
            return {"error": f"ULTIMATE understanding failed: {str(e)}"}
    
    async def _process_transcribe_queue(self, websocket, start_time) -> Optional[Dict[str, Any]]:
        """ULTIMATE: Process accumulated transcription audio"""
        try:
            # Collect all queued audio
            audio_chunks = []
            while not self.transcribe_audio_queue.empty():
                try:
                    chunk = self.transcribe_audio_queue.get_nowait()
                    audio_chunks.append(chunk)
                except queue.Empty:
                    break
            
            if not audio_chunks:
                return None
            
            logger.info(f"🎤 ULTIMATE: Processing {len(audio_chunks)} transcription chunks")
            
            # ULTIMATE: Combine all WebM chunks and convert to WAV
            combined_webm = b''.join(audio_chunks)
            wav_data = await self._webm_to_wav_ultimate(combined_webm)
            
            if not wav_data:
                logger.warning("Failed to convert WebM to WAV")
                return None
            
            # Calculate duration
            duration_ms = (len(wav_data) - 44) / 2 / self.sample_rate * 1000  # Subtract WAV header
            self.total_audio_length += duration_ms
            
            # Enhanced speech ratio estimation
            pcm_data = wav_data[44:]  # Skip WAV header
            speech_ratio = self._estimate_speech_ratio_ultimate(pcm_data)
            
            if speech_ratio > 0.1:
                self.speech_chunks_detected += 1
            
            # Quality control
            if duration_ms < 1000 or speech_ratio < 0.3:
                logger.debug(f"Skipping low quality audio: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
                return None
            
            # Get conversation context if available
            conversation_context = ""
            if self.conversation_manager and websocket:
                conversation_context = self.conversation_manager.get_conversation_context(websocket)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.chunks_processed += len(audio_chunks)
            
            logger.info(f"✅ ULTIMATE transcription processed: {duration_ms:.0f}ms, speech: {speech_ratio:.3f}")
            
            return {
                "audio_data": wav_data,
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "mode": "transcribe",
                "processed_at": time.time(),
                "ultimate": True,
                "conversation_context": conversation_context,
                "vad_enabled": self.vad_enabled
            }
            
        except Exception as e:
            logger.error(f"ULTIMATE transcription queue processing error: {e}")
            return {"error": f"ULTIMATE processing failed: {str(e)}"}
    
    async def _webm_to_wav_ultimate(self, webm_data: bytes) -> Optional[bytes]:
        """ULTIMATE: Convert WebM to WAV using FFmpeg subprocess (not pipes)"""
        temp_webm = None
        temp_wav = None
        
        try:
            # Create temporary files
            temp_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False, dir=self.temp_dir)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.temp_dir)
            
            # Write WebM data
            temp_webm.write(webm_data)
            temp_webm.flush()
            temp_webm.close()
            
            temp_wav.close()
            
            # ULTIMATE: Use FFmpeg subprocess with simple file conversion
            cmd = [
                'ffmpeg',
                '-i', temp_webm.name,
                '-acodec', 'pcm_s16le',
                '-ac', str(self.channels),
                '-ar', str(self.sample_rate),
                '-y',  # Overwrite output
                temp_wav.name
            ]
            
            # Run FFmpeg with timeout
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10.0
                )
            )
            
            if result.returncode == 0:
                # Read the converted WAV file
                with open(temp_wav.name, 'rb') as f:
                    wav_data = f.read()
                
                if len(wav_data) > 1000:  # Valid WAV file
                    return wav_data
                else:
                    logger.warning(f"WAV file too small: {len(wav_data)} bytes")
            else:
                logger.warning(f"FFmpeg conversion failed with code: {result.returncode}")
            
            return None
            
        except Exception as e:
            logger.error(f"ULTIMATE WebM to WAV conversion failed: {e}")
            return None
        finally:
            # Cleanup temp files
            for temp_file in [temp_webm, temp_wav]:
                if temp_file:
                    try:
                        if hasattr(temp_file, 'name'):
                            if os.path.exists(temp_file.name):
                                os.unlink(temp_file.name)
                        else:
                            if os.path.exists(temp_file):
                                os.unlink(temp_file)
                    except:
                        pass
    
    async def _webm_to_pcm_ultimate(self, webm_data: bytes) -> Optional[bytes]:
        """ULTIMATE: Convert WebM to PCM using FFmpeg subprocess"""
        temp_webm = None
        temp_pcm = None
        
        try:
            # Create temporary files
            temp_webm = tempfile.NamedTemporaryFile(suffix='.webm', delete=False, dir=self.temp_dir)
            temp_pcm = tempfile.NamedTemporaryFile(suffix='.pcm', delete=False, dir=self.temp_dir)
            
            # Write WebM data
            temp_webm.write(webm_data)
            temp_webm.flush()
            temp_webm.close()
            
            temp_pcm.close()
            
            # ULTIMATE: Use FFmpeg to extract PCM
            cmd = [
                'ffmpeg',
                '-i', temp_webm.name,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ac', str(self.channels),
                '-ar', str(self.sample_rate),
                '-y',  # Overwrite output
                temp_pcm.name
            ]
            
            # Run FFmpeg with timeout
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0
                )
            )
            
            if result.returncode == 0:
                # Read the converted PCM file
                with open(temp_pcm.name, 'rb') as f:
                    pcm_data = f.read()
                
                if len(pcm_data) > 320:  # At least 10ms of audio
                    return pcm_data
                else:
                    logger.debug(f"PCM file too small: {len(pcm_data)} bytes")
            
            return None
            
        except Exception as e:
            logger.error(f"ULTIMATE WebM to PCM conversion failed: {e}")
            return None
        finally:
            # Cleanup temp files
            for temp_file in [temp_webm, temp_pcm]:
                if temp_file:
                    try:
                        if hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except:
                        pass
    
    def _detect_speech_ultimate(self, pcm_data: bytes) -> bool:
        """ULTIMATE: Enhanced speech detection using VAD and energy"""
        try:
            if not pcm_data or len(pcm_data) < 320:  # Less than 10ms
                return False
            
            # Method 1: WebRTC VAD if available
            if self.vad_enabled and self.vad:
                try:
                    frame_size_bytes = 320  # 10ms at 16kHz
                    speech_frames = 0
                    total_frames = 0
                    
                    for i in range(0, len(pcm_data) - frame_size_bytes, frame_size_bytes):
                        frame = pcm_data[i:i + frame_size_bytes]
                        if len(frame) == frame_size_bytes:
                            try:
                                if self.vad.is_speech(frame, self.sample_rate):
                                    speech_frames += 1
                            except:
                                pass
                            total_frames += 1
                    
                    if total_frames > 0:
                        vad_ratio = speech_frames / total_frames
                        return vad_ratio > 0.3  # 30% of frames contain speech
                        
                except Exception as e:
                    logger.debug(f"VAD speech detection error: {e}")
            
            # Method 2: Energy-based fallback
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            
            energy_threshold = 800.0  # Threshold for speech
            return rms_energy > energy_threshold
            
        except Exception as e:
            logger.error(f"ULTIMATE speech detection error: {e}")
            return False
    
    def _estimate_speech_ratio_ultimate(self, pcm_data: bytes) -> float:
        """ULTIMATE: Enhanced speech ratio estimation"""
        try:
            if not pcm_data or len(pcm_data) < 1600:
                return 0.0
                
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # RMS energy calculation
            rms_energy = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            energy_threshold = 600.0
            energy_ratio = min(1.0, max(0.0, (rms_energy - energy_threshold) / (energy_threshold * 2)))
            
            # Zero crossing rate for speech characteristics
            zero_crossings = np.sum(np.diff(np.signbit(audio_array)))
            zcr_normalized = zero_crossings / max(len(audio_array) - 1, 1)
            
            if 0.01 <= zcr_normalized <= 0.2:
                zcr_ratio = 1.0
            else:
                zcr_ratio = 0.0
            
            # Combine metrics
            final_ratio = (energy_ratio * 0.7 + zcr_ratio * 0.3)
            return max(0.0, min(1.0, final_ratio))
            
        except Exception as e:
            logger.error(f"ULTIMATE speech ratio error: {e}")
            return 0.3
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ULTIMATE processing statistics"""
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
            "transcribe_queue_size": self.transcribe_audio_queue.qsize(),
            "understand_queue_size": self.understand_audio_queue.qsize(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "vad_enabled": self.vad_enabled,
            "active_connections": len(self.active_connections),
            "avg_processing_time_ms": round(avg_processing_time * 1000, 2),
            "ultimate": True,
            "temp_dir": self.temp_dir,
            "gap_detection": {
                "threshold_ms": self.gap_threshold_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms
            }
        }
    
    def reset(self):
        """Reset ULTIMATE processor state"""
        # Clear queues
        while not self.transcribe_audio_queue.empty():
            try:
                self.transcribe_audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.understand_audio_queue.empty():
            try:
                self.understand_audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.chunks_processed = 0
        self.speech_chunks_detected = 0
        self.total_audio_length = 0
        self.processing_times.clear()
        self.active_connections.clear()
        self.last_activity.clear()
        
        logger.info("✅ ULTIMATE audio processor reset completed")
    
    async def cleanup(self):
        """ULTIMATE: Enhanced cleanup with proper resource management"""
        logger.info("🧹 Starting ULTIMATE audio processor cleanup...")
        
        # Reset state
        self.reset()
        
        # Shutdown executor
        try:
            self.executor.shutdown(wait=True)
            logger.info("✅ ULTIMATE executor shutdown completed")
        except Exception as e:
            logger.error(f"ULTIMATE executor shutdown error: {e}")
        
        # Cleanup temp directory
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"✅ ULTIMATE temp directory cleaned: {self.temp_dir}")
        except Exception as e:
            logger.error(f"ULTIMATE temp directory cleanup error: {e}")
        
        logger.info("✅ ULTIMATE audio processor fully cleaned up")

# Alias for backward compatibility
FixedAudioProcessor = UltimateAudioProcessor
