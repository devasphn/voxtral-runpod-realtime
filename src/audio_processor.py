import asyncio
import logging
import numpy as np
from typing import Optional, Dict, Any
import tempfile
import os
import subprocess
import queue
import webrtcvad

logger = logging.getLogger(__name__)

class PerfectAudioProcessor:
    """FINAL PERFECTED: Audio processor with bulletproof WebM handling and frame-perfect VAD."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        conversation_manager=None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.conversation_manager = conversation_manager
        self.transcribe_audio_queue = queue.Queue()
        
        try:
            self.vad = webrtcvad.Vad(3)  # Mode 3: most aggressive
            self.vad_enabled = True
            # VAD requires specific frame lengths (10, 20, or 30 ms)
            self.vad_frame_duration_ms = 30 
            self.vad_frame_size_bytes = int(sample_rate * (self.vad_frame_duration_ms / 1000.0) * 2) # 16-bit PCM
            logger.info(f"✅ FINAL PERFECTED WebRTC VAD initialized with {self.vad_frame_duration_ms}ms frames.")
        except Exception as e:
            self.vad = None
            self.vad_enabled = False
            logger.warning(f"⚠️ WebRTC VAD not available, falling back to energy detection: {e}")
    
    async def process_webm_chunk_transcribe(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """PERFECT: Transcription with robust, queued WebM processing"""
        if not self._is_valid_webm_chunk(webm_data):
            return None
        
        self.transcribe_audio_queue.put(webm_data)
        
        if self.transcribe_audio_queue.qsize() >= 15: # Process in ~1.5-second batches
            return await self._process_transcribe_queue()
        
        return None
    
    async def process_webm_chunk_understand(self, webm_data: bytes) -> Optional[Dict[str, Any]]:
        """FINAL PERFECTED: Understanding mode with direct WebM to PCM conversion for gap detection"""
        if not self._is_valid_webm_chunk(webm_data):
            return None
            
        pcm_data = await self._webm_to_pcm_perfect(webm_data)
        
        if pcm_data:
            speech_detected = self._detect_speech_perfect(pcm_data)
            return {
                "pcm_data": pcm_data,
                "speech_detected": speech_detected,
                "perfect": True
            }
        return None
            
    def _is_valid_webm_chunk(self, webm_data: bytes) -> bool:
        if not webm_data or len(webm_data) < 100:
            return False
        if webm_data.startswith(b'\x1aE\xdf\xa3') or b'webm' in webm_data[:10].lower():
            return True
        return False
    
    async def _process_transcribe_queue(self) -> Optional[Dict[str, Any]]:
        """PERFECT: Process accumulated transcription audio from the queue"""
        audio_chunks = []
        while not self.transcribe_audio_queue.empty():
            try:
                audio_chunks.append(self.transcribe_audio_queue.get_nowait())
            except queue.Empty:
                break
        
        if not audio_chunks: return None
            
        combined_webm = b''.join(audio_chunks)
        wav_data = await self._webm_to_wav_bulletproof(combined_webm)
        
        if not wav_data:
            logger.warning("Failed to convert WebM to WAV for transcription")
            return None
        
        return {"audio_data": wav_data, "perfect": True}
    
    async def _webm_to_wav_bulletproof(self, webm_data: bytes) -> Optional[bytes]:
        """PERFECT: Bulletproof WebM to WAV conversion with multiple FFmpeg strategies"""
        # This function remains highly effective and doesn't need changes.
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            
            temp_webm_path = temp_webm.name
            temp_wav_path = temp_wav.name
            temp_webm.write(webm_data)
            temp_webm.close()
            
            ffmpeg_commands = [
                ['ffmpeg', '-loglevel', 'error', '-i', temp_webm_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate), '-f', 'wav', '-y', temp_wav_path],
                ['ffmpeg', '-loglevel', 'error', '-err_detect', 'ignore_err', '-i', temp_webm_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate), '-f', 'wav', '-y', temp_wav_path]
            ]
            
            success = False
            for cmd in ffmpeg_commands:
                proc = await asyncio.create_subprocess_exec(*cmd, stderr=asyncio.subprocess.PIPE)
                _, stderr = await proc.communicate()
                if proc.returncode == 0 and os.path.getsize(temp_wav_path) > 44:
                    success = True
                    break
            
            wav_data = None
            if success:
                with open(temp_wav_path, 'rb') as f:
                    wav_data = f.read()

            os.unlink(temp_webm_path)
            os.unlink(temp_wav_path)
            return wav_data

    async def _webm_to_pcm_perfect(self, webm_data: bytes) -> Optional[bytes]:
        """PERFECT: Convert a single WebM chunk to raw PCM data for VAD"""
        # This function is also highly effective.
        cmd = ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate), 'pipe:1']
        proc = await asyncio.create_subprocess_exec(*cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pcm_data, stderr = await proc.communicate(input=webm_data)
        if proc.returncode != 0:
            logger.warning(f"FFmpeg failed to convert to PCM: {stderr.decode()}")
            return None
        return pcm_data

    def _detect_speech_perfect(self, pcm_data: bytes) -> bool:
        """FINAL PERFECTED: The CRITICAL FIX for VAD by correctly chunking audio into valid frame sizes."""
        if self.vad_enabled and self.vad:
            # THE FIX: Iterate through the PCM data in correctly sized chunks that VAD expects.
            num_frames = len(pcm_data) // self.vad_frame_size_bytes
            if num_frames == 0:
                return False # Not enough data for even one frame
                
            is_speech_found = False
            try:
                for i in range(num_frames):
                    start_byte = i * self.vad_frame_size_bytes
                    end_byte = start_byte + self.vad_frame_size_bytes
                    frame = pcm_data[start_byte:end_byte]
                    
                    if self.vad.is_speech(frame, self.sample_rate):
                        is_speech_found = True
                        break # Found speech, no need to check further
            except Exception as e:
                # This can happen if a frame is corrupted, fallback to energy.
                logger.warning(f"VAD frame error, falling back to energy check: {e}")
                return self._detect_speech_by_energy(pcm_data)
            
            return is_speech_found

        # Fallback for when VAD is disabled
        return self._detect_speech_by_energy(pcm_data)

    def _detect_speech_by_energy(self, pcm_data: bytes) -> bool:
        """Energy-based speech detection as a fallback."""
        if not pcm_data: return False
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        # Prevent overflow with very loud audio
        rms_energy = np.sqrt(np.mean(np.square(audio_array.astype(np.float64))))
        # Tuned threshold for typical speech
        return rms_energy > 350
