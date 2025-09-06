import asyncio, logging, time, tempfile, os, base64
import torch
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage, AudioChunk, TextChunk
from transformers import VoxtralForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)

class VoxtralUnderstandingManager:
    def __init__(self, model_name="mistralai/Voxtral-Mini-3B-2507", device="cuda", torch_dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.is_loaded = False
        self.model_info = {}
        self.target_response_ms = 200

    async def load_model(self):
        logger.info(f"Loading model {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype,
            device_map="auto", trust_remote_code=True,
            low_cpu_mem_usage=True, attn_implementation="eager"
        )
        self.model.eval()
        self.is_loaded = True

    def _bytes_to_audio(self, data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            wav = wave.open(f.name,'wb')
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
            wav.writeframes(data); wav.close()
            audio = Audio.from_file(f.name)
        os.unlink(f.name)
        return audio

    async def understand_audio(self, message):
        if not self.is_loaded:
            return {"error":"Model not loaded"}

        start = time.time()
        audio_data = message.get("audio")
        if isinstance(audio_data, str):
            audio_data = base64.b64decode(audio_data)

        audio = self._bytes_to_audio(audio_data)
        conversation = [
            UserMessage(content=[AudioChunk(audio=audio), TextChunk(text=message.get("text",""))])
        ]
        inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        inputs = {k:v.to(self.device, dtype=self.torch_dtype) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=120, temperature=0.2, top_p=0.8, use_cache=True)
        response = self.processor.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        total = (time.time()-start)*1000
        return {"response":response,"processing_time_ms":total,"sub_200ms":total<200}
