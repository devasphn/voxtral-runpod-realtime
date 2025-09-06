import asyncio, logging, tempfile, os, subprocess, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mistral_common.audio import Audio
import webrtcvad

logger = logging.getLogger(__name__)

class UnderstandingAudioProcessor:
    def __init__(self, sample_rate=16000, channels=1, gap_threshold_ms=300, conversation_manager=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.gap_threshold_ms = gap_threshold_ms
        self.conversation_manager = conversation_manager

        # Gap detection parameters
        self.min_speech_duration_ms = gap_threshold_ms
        self.max_speech_duration_ms = 30000
        self.gap_threshold_samples = int(sample_rate * (gap_threshold_ms / 1000.0))

        # VAD configuration
        try:
            self.vad = webrtcvad.Vad(1)  # Aggressiveness level 1 (0-3)
            self.vad_enabled = True
            logger.info("✅ WebRTC VAD initialized")
        except Exception as e:
            logger.warning(f"WebRTC VAD initialization failed: {e}, using fallback")
            self.vad = None
            self.vad_enabled = False

        # Energy-based speech detection fallback
        self.energy_threshold = 100.0
        self.zcr_min = 0.001
        self.zcr_max = 0.5

        # Connection-specific buffers
        self.audio_segments = {}
        self.speech_buffers = {}
        self.silence_counters = {}
        self.last_speech_times = {}

        # Thread pool for audio processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"✅ AudioProcessor initialized: gap={gap_threshold_ms}ms, min_speech={self.min_speech_duration_ms}ms")

    async def process_audio_understanding(self, audio_data: bytes, websocket=None):
        """Process audio data for understanding mode with gap detection"""
        start_time = time.time()
        conn_id = id(websocket)
        
        # Initialize connection-specific data structures
        self.audio_segments.setdefault(conn_id, bytearray())
        self.speech_buffers.setdefault(conn_id, [])
        self.silence_counters.setdefault(conn_id, 0)
        self.last_speech_times.setdefault(conn_id, 0)

        # Convert audio data to PCM
        try:
            pcm_data = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._convert_to_pcm, audio_data
            )
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return {"audio_received": False, "error": "Audio conversion failed"}

        if not pcm_data or len(pcm_data) < 32:
            return {"audio_received": True, "speech_complete": False, "duration_ms": 0}

        # Add PCM data to segment buffer
        self.audio_segments[conn_id].extend(pcm_data)

        # Perform VAD analysis
        speech_detected = self._detect_speech(pcm_data)
        current_time = time.time()

        # Update speech state
        if speech_detected:
            self.last_speech_times[conn_id] = current_time
            self.silence_counters[conn_id] = 0
        else:
            self.silence_counters[conn_id] += 1

        # Calculate durations
        total_duration_ms = len(self.audio_segments[conn_id]) / 2 / self.sample_rate * 1000
        silence_duration_ms = (current_time - self.last_speech_times.get(conn_id, current_time)) * 1000

        # Check minimum duration requirement
        if total_duration_ms < self.min_speech_duration_ms:
            return {
                "audio_received": True,
                "speech_complete": False,
                "duration_ms": total_duration_ms,
                "speech_detected": speech_detected,
                "silence_duration_ms": silence_duration_ms
            }

        # Check gap detection conditions
        gap_detected = (
            silence_duration_ms >= self.gap_threshold_ms or 
            total_duration_ms >= self.max_speech_duration_ms
        )

        if gap_detected and total_duration_ms >= self.min_speech_duration_ms:
            # Process and finalize the segment
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._finalize_segment, conn_id
            )
            
            # Reset buffers for next segment
            self.silence_counters[conn_id] = 0
            self.last_speech_times[conn_id] = current_time
            
            result.update({
                "speech_complete": True,
                "gap_detected": True,
                "total_duration_ms": total_duration_ms,
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            
            logger.info(f"Speech segment finalized: {total_duration_ms:.0f}ms, gap: {silence_duration_ms:.0f}ms")
            return result

        # Return intermediate status
        return {
            "audio_received": True,
            "speech_complete": False,
            "duration_ms": total_duration_ms,
            "speech_detected": speech_detected,
            "silence_duration_ms": silence_duration_ms,
            "remaining_to_gap_ms": max(0, self.gap_threshold_ms - silence_duration_ms)
        }

    def _convert_to_pcm(self, audio_data: bytes) -> bytes:
        """Convert WebM/various audio formats to 16kHz mono PCM"""
        if not audio_data or len(audio_data) < 10:
            return b""

        temp_input = None
        temp_output = None
        
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_data)
                temp_input = f.name

            # Create temporary output file
            temp_output = temp_input + ".pcm"

            # Multiple FFmpeg strategies
            conversion_commands = [
                # Strategy 1: Direct WebM to PCM
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", temp_input,
                    "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", "16000",
                    "-f", "s16le",
                    temp_output
                ],
                # Strategy 2: Force WebM container
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "webm", "-i", temp_input,
                    "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", "16000",
                    "-f", "s16le",
                    temp_output
                ],
                # Strategy 3: Try as generic audio
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", temp_input,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", "16000",
                    "-f", "s16le",
                    temp_output
                ]
            ]

            # Try each conversion strategy
            for i, cmd in enumerate(conversion_commands):
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        timeout=10,
                        check=True
                    )
                    
                    # Read converted PCM data
                    if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                        with open(temp_output, 'rb') as f:
                            pcm_data = f.read()
                        if len(pcm_data) > 32:  # Valid audio data
                            logger.debug(f"Audio conversion successful with strategy {i+1}")
                            return pcm_data
                except subprocess.CalledProcessError as e:
                    logger.debug(f"Conversion strategy {i+1} failed: {e}")
                    continue
                except subprocess.TimeoutExpired:
                    logger.warning(f"Conversion strategy {i+1} timed out")
                    continue

            # If all strategies failed, try to interpret as raw PCM
            logger.warning("All FFmpeg strategies failed, trying as raw audio")
            if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
                # Might be WAV format
                return audio_data[44:]  # Skip WAV header
            elif len(audio_data) % 2 == 0:
                # Might already be PCM
                return audio_data

            return b""

        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return b""
        finally:
            # Clean up temporary files
            for temp_file in [temp_input, temp_output]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.debug(f"Failed to clean up {temp_file}: {e}")

    def _detect_speech(self, pcm_data: bytes) -> bool:
        """Detect speech in PCM data using WebRTC VAD or energy-based fallback"""
        if not pcm_data or len(pcm_data) < 320:  # Need at least 20ms of audio
            return False

        try:
            if self.vad_enabled and self.vad:
                # Use WebRTC VAD
                frame_size = 320  # 20ms at 16kHz (320 bytes = 160 samples * 2 bytes)
                speech_frames = 0
                total_frames = 0

                for i in range(0, len(pcm_data) - frame_size + 1, frame_size):
                    frame = pcm_data[i:i + frame_size]
                    if len(frame) == frame_size:
                        try:
                            if self.vad.is_speech(frame, self.sample_rate):
                                speech_frames += 1
                            total_frames += 1
                        except Exception:
                            # WebRTC VAD failed, use energy fallback for this frame
                            if self._energy_based_vad(frame):
                                speech_frames += 1
                            total_frames += 1

                # Return True if at least 30% of frames contain speech
                return total_frames > 0 and (speech_frames / total_frames) >= 0.3

            else:
                # Energy-based fallback
                return self._energy_based_vad(pcm_data)

        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return self._energy_based_vad(pcm_data)

    def _energy_based_vad(self, pcm_data: bytes) -> bool:
        """Energy-based voice activity detection fallback"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            if len(audio_array) == 0:
                return False

            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_array ** 2))
            
            # Calculate zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(audio_array)))[0]
            zcr = len(zero_crossings) / len(audio_array) if len(audio_array) > 1 else 0

            # Speech detection based on energy and zero crossing rate
            has_energy = rms_energy > self.energy_threshold
            good_zcr = self.zcr_min <= zcr <= self.zcr_max

            return has_energy and good_zcr

        except Exception as e:
            logger.debug(f"Energy-based VAD error: {e}")
            return False

    def _finalize_segment(self, conn_id: int):
        """Finalize audio segment and prepare for processing"""
        try:
            # Get accumulated PCM data
            pcm_data = bytes(self.audio_segments.get(conn_id, bytearray()))
            
            # Clear the segment buffer for next audio
            if conn_id in self.audio_segments:
                self.audio_segments[conn_id].clear()

            return {
                "audio_data": pcm_data,
                "segment_length": len(pcm_data),
                "duration_ms": len(pcm_data) / 2 / self.sample_rate * 1000 if pcm_data else 0
            }

        except Exception as e:
            logger.error(f"Segment finalization error: {e}")
            return {"audio_data": b"", "segment_length": 0, "duration_ms": 0}

    def cleanup_connection(self, websocket):
        """Clean up connection-specific data"""
        conn_id = id(websocket)
        
        # Remove connection-specific data
        self.audio_segments.pop(conn_id, None)
        self.speech_buffers.pop(conn_id, None)
        self.silence_counters.pop(conn_id, None)
        self.last_speech_times.pop(conn_id, None)
        
        logger.debug(f"Cleaned up audio processor data for connection {conn_id}")

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up audio processor...")
        
        # Clear all buffers
        self.audio_segments.clear()
        self.speech_buffers.clear()
        self.silence_counters.clear()
        self.last_speech_times.clear()
        
        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("✅ Audio processor cleanup completed")

    def get_stats(self):
        """Get processor statistics"""
        return {
            "gap_threshold_ms": self.gap_threshold_ms,
            "energy_threshold": self.energy_threshold,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "active_connections": len(self.audio_segments),
            "vad_enabled": self.vad_enabled,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "max_speech_duration_ms": self.max_speech_duration_ms
        }
