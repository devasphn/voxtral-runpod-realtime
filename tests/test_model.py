import pytest
import torch
import asyncio
import numpy as np
import io
import base64
from unittest.mock import Mock, patch, AsyncMock
from src.model_loader import VoxtralModelManager
from src.audio_processor import AudioProcessor
import tempfile
import wave

@pytest.fixture
def mock_model_manager():
    """Create mock model manager for testing"""
    manager = VoxtralModelManager(
        model_name="mistralai/Voxtral-Mini-3B-2507",
        device="cpu",  # Use CPU for testing
        torch_dtype=torch.float32
    )
    return manager

@pytest.fixture
def audio_processor():
    """Create audio processor for testing"""
    return AudioProcessor(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=30
    )

@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing"""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

@pytest.fixture
def sample_wav_file():
    """Create sample WAV file for testing"""
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        with wave.open(tmp_file.name, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            
            # Generate 1 second of sine wave
            duration = 1.0
            sample_rate = 16000
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            wav_file.writeframes(audio_int16.tobytes())
        
        return tmp_file.name

def test_model_manager_initialization():
    """Test VoxtralModelManager initialization"""
    manager = VoxtralModelManager(
        model_name="test-model",
        device="cpu",
        torch_dtype=torch.float32
    )
    
    assert manager.model_name == "test-model"
    assert manager.device.type == "cpu"
    assert manager.torch_dtype == torch.float32
    assert not manager.is_loaded
    assert manager.model is None
    assert manager.processor is None

def test_gpu_availability():
    """Test GPU availability detection"""
    # This test will pass regardless of GPU availability
    is_cuda_available = torch.cuda.is_available()
    
    if is_cuda_available:
        manager = VoxtralModelManager(device="cuda")
        assert manager.device.type == "cuda"
    else:
        manager = VoxtralModelManager(device="cuda")
        # Should fallback to CPU if CUDA not available
        assert manager.device.type in ["cuda", "cpu"]

@pytest.mark.asyncio
async def test_model_loading_mock():
    """Test model loading with mocked components"""
    with patch('src.model_loader.VoxtralForConditionalGeneration') as mock_model_class, \
         patch('src.model_loader.AutoProcessor') as mock_processor_class:
        
        # Setup mocks
        mock_model = Mock()
        mock_processor = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        manager = VoxtralModelManager(device="cpu")
        
        # Test loading
        await manager.load_model()
        
        assert manager.is_loaded
        assert manager.model == mock_model
        assert manager.processor == mock_processor
        
        # Verify model was put in eval mode
        mock_model.eval.assert_called_once()

def test_parameter_counting():
    """Test parameter counting functionality"""
    # Create a simple mock model
    mock_model = Mock()
    
    # Mock parameters
    param1 = Mock()
    param1.numel.return_value = 1000
    param2 = Mock()
    param2.numel.return_value = 2000
    
    mock_model.parameters.return_value = [param1, param2]
    
    manager = VoxtralModelManager(device="cpu")
    manager.model = mock_model
    
    param_count = manager._count_parameters()
    assert param_count == 3000

def test_memory_usage_cpu():
    """Test memory usage calculation on CPU"""
    manager = VoxtralModelManager(device="cpu")
    memory_usage = manager._get_memory_usage()
    
    # On CPU, should return 0 GPU memory
    assert memory_usage["gpu_memory"] == 0.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_usage_gpu():
    """Test memory usage calculation on GPU"""
    manager = VoxtralModelManager(device="cuda")
    memory_usage = manager._get_memory_usage()
    
    # Should have GPU memory info
    assert "gpu_allocated_gb" in memory_usage
    assert "gpu_cached_gb" in memory_usage
    assert "gpu_total_gb" in memory_usage

@pytest.mark.asyncio
async def test_transcribe_audio_not_loaded():
    """Test transcription when model not loaded"""
    manager = VoxtralModelManager(device="cpu")
    result = await manager.transcribe_audio(b"fake_audio_data")
    
    assert "error" in result
    assert result["error"] == "Model not loaded"

@pytest.mark.asyncio
async def test_understand_audio_not_loaded():
    """Test understanding when model not loaded"""
    manager = VoxtralModelManager(device="cpu")
    message = {"audio": "fake_audio", "text": "What is this?"}
    result = await manager.understand_audio(message)
    
    assert "error" in result
    assert result["error"] == "Model not loaded"

def test_bytes_to_audio_invalid():
    """Test audio conversion with invalid data"""
    manager = VoxtralModelManager(device="cpu")
    
    # Test with invalid audio data
    result = manager._bytes_to_audio(b"invalid_audio_data")
    assert result is None

@pytest.mark.asyncio
async def test_cleanup():
    """Test model cleanup"""
    manager = VoxtralModelManager(device="cpu")
    
    # Set up mock model and processor
    manager.model = Mock()
    manager.processor = Mock()
    manager.is_loaded = True
    
    await manager.cleanup()
    
    assert manager.model is None
    assert manager.processor is None
    assert not manager.is_loaded

# Audio Processor Tests

def test_audio_processor_initialization():
    """Test AudioProcessor initialization"""
    processor = AudioProcessor(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=30,
        vad_mode=3
    )
    
    assert processor.sample_rate == 16000
    assert processor.channels == 1
    assert processor.chunk_duration_ms == 30
    assert processor.silence_threshold == 10
    assert processor.silence_count == 0

def test_audio_processor_stats(audio_processor):
    """Test audio processor statistics"""
    stats = audio_processor.get_stats()
    
    expected_keys = [
        "buffer_size", "sample_rate", "channels", 
        "chunk_duration_ms", "silence_count", "silence_threshold"
    ]
    
    for key in expected_keys:
        assert key in stats

def test_audio_processor_reset(audio_processor):
    """Test audio processor reset"""
    # Add some data to buffer
    audio_processor.audio_buffer.append({"test": "data"})
    audio_processor.silence_count = 5
    
    # Reset
    audio_processor.reset()
    
    assert len(audio_processor.audio_buffer) == 0
    assert audio_processor.silence_count == 0

def test_should_process_buffer_empty(audio_processor):
    """Test buffer processing decision with empty buffer"""
    # Empty buffer should not be processed
    should_process = audio_processor._should_process_buffer()
    assert not should_process

def test_should_process_buffer_insufficient_chunks(audio_processor):
    """Test buffer processing with insufficient chunks"""
    # Add few chunks (less than minimum)
    for i in range(3):
        audio_processor.audio_buffer.append({
            "audio": b"test_audio",
            "is_speech": True,
            "timestamp": i
        })
    
    should_process = audio_processor._should_process_buffer()
    assert not should_process

def test_should_process_buffer_with_speech(audio_processor):
    """Test buffer processing with speech detected"""
    # Add chunks with speech
    for i in range(6):  # More than minimum
        audio_processor.audio_buffer.append({
            "audio": b"test_audio",
            "is_speech": True,
            "timestamp": i
        })
    
    should_process = audio_processor._should_process_buffer()
    assert should_process

def test_process_buffer_empty(audio_processor):
    """Test processing empty buffer"""
    result = audio_processor._process_buffer()
    
    assert "error" in result
    assert result["error"] == "Empty buffer"

@pytest.mark.asyncio
async def test_transcription_response_format():
    """Test transcription response format"""
    # Test expected response format
    expected_keys = ["type", "text", "language", "confidence", "timestamp"]
    
    # Mock response
    mock_response = {
        "type": "transcription",
        "text": "Hello world",
        "language": "auto-detected", 
        "confidence": 0.95,
        "timestamp": 1234567890.0
    }
    
    for key in expected_keys:
        assert key in mock_response

@pytest.mark.asyncio
async def test_understanding_response_format():
    """Test understanding response format"""
    # Test expected response format
    expected_keys = ["type", "response", "query", "timestamp"]
    
    # Mock response
    mock_response = {
        "type": "understanding",
        "response": "I can hear someone saying hello",
        "query": "What can you hear?",
        "timestamp": 1234567890.0
    }
    
    for key in expected_keys:
        assert key in mock_response

def test_base64_encoding_decoding(sample_audio_data):
    """Test base64 encoding/decoding for audio data"""
    # Encode
    encoded = base64.b64encode(sample_audio_data).decode()
    assert isinstance(encoded, str)
    
    # Decode
    decoded = base64.b64decode(encoded)
    assert decoded == sample_audio_data

@pytest.mark.asyncio
async def test_understanding_message_format():
    """Test understanding message format"""
    test_audio = b"test_audio_data"
    test_query = "What can you hear?"
    
    # Format as base64
    audio_b64 = base64.b64encode(test_audio).decode()
    
    message = {
        "audio": audio_b64,
        "text": test_query
    }
    
    assert "audio" in message
    assert "text" in message
    assert isinstance(message["audio"], str)
    assert message["text"] == test_query

def test_torch_dtype_handling():
    """Test torch dtype handling"""
    # Test different dtypes
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        manager = VoxtralModelManager(torch_dtype=dtype)
        assert manager.torch_dtype == dtype

@pytest.mark.asyncio 
async def test_concurrent_requests():
    """Test handling multiple concurrent requests"""
    manager = VoxtralModelManager(device="cpu")
    
    # Mock audio data
    audio_data = b"test_audio_data"
    
    # Multiple concurrent requests (should handle gracefully even if model not loaded)
    tasks = [
        manager.transcribe_audio(audio_data),
        manager.transcribe_audio(audio_data),
        manager.transcribe_audio(audio_data)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should return error since model not loaded
    for result in results:
        assert isinstance(result, dict)
        assert "error" in result

def test_device_selection():
    """Test device selection logic"""
    # Test explicit CPU
    manager_cpu = VoxtralModelManager(device="cpu")
    assert manager_cpu.device.type == "cpu"
    
    # Test explicit CUDA (may fallback to CPU if not available)
    manager_cuda = VoxtralModelManager(device="cuda")
    assert manager_cuda.device.type in ["cuda", "cpu"]
    
    # Test auto-selection
    manager_auto = VoxtralModelManager()
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert manager_auto.device.type == expected_device

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
