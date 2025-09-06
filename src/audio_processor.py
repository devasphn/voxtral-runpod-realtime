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

        # Lowered to match gap threshold
        self.min_speech_duration_ms = gap_threshold_ms
        self.max_speech_duration_ms = 30000
        self.gap_threshold_samples = int(sample_rate * (gap_threshold_ms / 1000.0))

        # VAD
        self.vad = webrtcvad.Vad(1)
        self.vad_enabled = True

        # Energy thresholds lowered
        self.energy_threshold = 50.0
        self.zcr_min = 0.001
        self.zcr_max = 0.5

        # Buffers
        self.audio_segments = {}
        self.speech_buffers = {}
        self.silence_counters = {}

        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info(f"AudioProcessor initialized: gap={gap_threshold_ms}ms, min_speech={self.min_speech_duration_ms}ms")

    async def process_audio_understanding(self, audio_data: bytes, websocket=None):
        start_time = time.time()
        conn_id = id(websocket)
        # initialize
        self.audio_segments.setdefault(conn_id, bytearray())
        self.speech_buffers.setdefault(conn_id, [])
        self.silence_counters.setdefault(conn_id, 0)

        # convert
        pcm = await asyncio.get_event_loop().run_in_executor(self.executor, self._convert_to_pcm, audio_data)
        if not pcm:
            return {"audio_received":True,"speech_complete":False}

        self.audio_segments[conn_id].extend(pcm)
        # VAD on frames including final
        frame_bytes = 320
        speech_frames = total_frames = 0
        for i in range(0, len(pcm), frame_bytes):
            frame = pcm[i:i+frame_bytes]
            if len(frame) < frame_bytes:
                frame = frame.ljust(frame_bytes, b'\0')
            total_frames += 1
            if self.vad.is_speech(frame, self.sample_rate):
                speech_frames += 1
        speech_ratio = speech_frames/total_frames if total_frames else 0
        is_speech = speech_ratio > 0.15

        duration_ms = len(self.audio_segments[conn_id])/2/self.sample_rate*1000
        silence_ms = self.silence_counters[conn_id]* (frame_bytes/2/self.sample_rate*1000)

        if is_speech:
            self.speech_buffers[conn_id].append(time.time())
            self.silence_counters[conn_id] = 0
        else:
            self.silence_counters[conn_id] += 1

        if duration_ms < self.min_speech_duration_ms:
            return {"audio_received":True,"speech_complete":False,"duration_ms":duration_ms}

        # gap detection
        if silence_ms >= self.gap_threshold_ms or duration_ms >= self.max_speech_duration_ms:
            result = await asyncio.get_event_loop().run_in_executor(self.executor, self._finalize_segment, conn_id)
            # preserve buffers for next utterance
            self.silence_counters[conn_id] = 0
            return result

        return {"audio_received":True,"speech_complete":False,"duration_ms":duration_ms}

    def _convert_to_pcm(self, data: bytes):
        try:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(data)
                in_path = f.name
            cmd = ["ffmpeg","-y","-loglevel","error","-i",in_path,
                   "-acodec","pcm_s16le","-ac","1","-ar","16000","-f","s16le","-"]
            proc = subprocess.run(cmd, capture_output=True, timeout=10)
            pcm = proc.stdout if proc.returncode==0 else None
            return pcm
        finally:
            try: os.unlink(in_path)
            except: pass

    def _finalize_segment(self, conn_id):
        pcm = bytes(self.audio_segments.pop(conn_id, b""))
        return {"speech_complete":True,"audio_data":pcm}

    def cleanup_connection(self, websocket):
        cid = id(websocket)
        self.audio_segments.pop(cid,None)
        self.speech_buffers.pop(cid,None)
        self.silence_counters.pop(cid,None)

    async def cleanup(self):
        self.executor.shutdown(wait=True)

    def get_stats(self):
        return {"gap_threshold_ms":self.gap_threshold_ms,"energy_threshold":self.energy_threshold}
