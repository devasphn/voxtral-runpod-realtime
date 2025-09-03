set -e

echo "📦 Installing Voxtral Mini 3B Dependencies..."
echo "=============================================="

# Update system packages
echo "🔄 Updating system packages..."
apt-get update
apt-get install -y --no-install-recommends \\
    ffmpeg \\
    libsndfile1 \\
    wget \\
    curl \\
    git \\
    build-essential \\
    software-properties-common

# Clean apt cache
rm -rf /var/lib/apt/lists/*

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip install --no-cache-dir -r requirements.txt

# Verify installations
echo "✅ Verifying installations..."

# Check PyTorch GPU support
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"

# Check Transformers
python -c "
import transformers
print(f'Transformers Version: {transformers.__version__}')
"

# Check vLLM
python -c "
try:
    import vllm
    print(f'vLLM Version: {vllm.__version__}')
except ImportError:
    print('vLLM not installed')
"

# Check mistral-common
python -c "
try:
    import mistral_common
    print(f'Mistral Common Version: {mistral_common.__version__}')
except ImportError:
    print('mistral-common not installed')
"

# Check FastAPI
python -c "
import fastapi
print(f'FastAPI Version: {fastapi.__version__}')
"

# Check audio libraries
python -c "
import librosa, soundfile, pydub
print('Audio libraries: librosa, soundfile, pydub - OK')
"

echo "✅ All dependencies installed successfully!"

# Pre-download model (optional)
if [ "${PRELOAD_MODEL:-false}" = "true" ]; then
    echo "📥 Pre-downloading Voxtral model..."
    python -c "
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import os

model_name = 'mistralai/Voxtral-Mini-3B-2507'
cache_dir = '/app/models'

print(f'Downloading {model_name} to {cache_dir}...')

try:
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    print('✅ Processor downloaded')
    
    # Download model config only (not full weights to save space)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    print('✅ Model config downloaded')
    
    print('Model will be fully downloaded on first use')
except Exception as e:
    print(f'Warning: Failed to pre-download model: {e}')
    print('Model will be downloaded on first use')
"
fi

echo "🎉 Installation completed successfully!"
