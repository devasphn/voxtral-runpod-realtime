
import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import webrtcvad

logger = logging.getLogger(__name__)

class PerfectAudioProcessor:
    """FINAL PERFECTED: Audio processor with frame-perfect VAD and universally compatible timeout handling."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        try:
            self.vad = webrtcvad.Vad(3)  # Mode 3: most aggressive for responsiveness
            self.vad_enabled = True
            self.vad_frame_duration_ms = 30 
            self.vad_frame_size_bytes = int(sample_rate * (self.vad_frame_duration_ms / 1000.0) * 2) # Samples * bytes_per_sample (16-bit)
            logger.info(f"✅ FINAL PERFECTED WebRTC VAD initialized. Frame size: {self.vad_frame_size_bytes} bytes ({self.vad_frame_duration_ms}ms)")
        except Exception as e:
            self.vad = None
            self.vad_enabled = False
            logger.warning(f"⚠️ WebRTC VAD not available, falling back to energy detection: {e}")

    async def process_webm_chunk_understand(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        if not (webm_data and len(webm_data) > 100): return None
        
        pcm_data = await self._webm_to_pcm_perfect(webm_data)
        if pcm_data:
            speech_detected = self._detect_speech_perfect(pcm_data)
            return {"pcm_data": pcm_data, "speech_detected": speech_detected}
        return None

    async def _webm_to_pcm_perfect(self, webm_data: bytes) -> Optional[bytes]:
        """THE FIX: Replaced incompatible timeout with the universally supported asyncio.wait_for pattern."""
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate), 'pipe:1']
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            pcm_data, stderr = await asyncio.wait_for(proc.communicate(input=webm_data), timeout=2.0)
            
            if proc.returncode != 0:
                # This is expected for tiny, invalid chunks from the browser, so we don't log it as a major warning.
                if "Invalid data" not in stderr.decode():
                    logger.warning(f"FFmpeg failed to convert to PCM: {stderr.decode()}")
                return None
            return pcm_data
        except asyncio.TimeoutError:
            logger.error("FFmpeg process timed out during PCM conversion.")
            if proc.returncode is None: proc.kill()
            await proc.wait()
            return None
        except Exception as e:
            logger.error(f"Error converting WebM to PCM: {e}", exc_info=True)
            return None

    def _detect_speech_perfect(self, pcm_data: bytes) -> bool:
        """THE FIX: Correctly iterates through PCM data in valid VAD frame sizes."""
        if not self.vad_enabled or not self.vad:
            return self._detect_speech_by_energy(pcm_data)

        num_frames = len(pcm_data) // self.vad_frame_size_bytes
        if num_frames == 0: return False
            
        try:
            for i in range(num_frames):
                start = i * self.vad_frame_size_bytes
                end = start + self.vad_frame_size_bytes
                if self.vad.is_speech(pcm_data[start:end], self.sample_rate):
                    return True # Found speech, return immediately
        except Exception as e:
            logger.warning(f"VAD frame error: {e}. Falling back to energy check.")
            return self._detect_speech_by_energy(pcm_data)
        
        return False

    def _detect_speech_by_energy(self, pcm_data: bytes) -> bool:
        if not pcm_data: return False
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        if audio_array.size == 0: return False
        rms_energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))
        return rms_energy > 350 # Tuned energy threshold
