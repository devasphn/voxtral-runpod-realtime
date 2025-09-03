set -e

echo "🚀 Starting Voxtral Mini 3B Real-Time Server..."
echo "=================================================="

# Set environment variables
export PYTHONPATH="/app:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="/app/models"
export TRANSFORMERS_CACHE="/app/models"
export TORCH_HOME="/app/models"

# Log system information
echo "📊 System Information:"
echo "- GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "- GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- Working Directory: $(pwd)"

# Create necessary directories
mkdir -p /app/logs /app/temp /app/models

# Set permissions
chmod 755 /app/scripts/*.sh

# Health check endpoint (background)
echo "🔍 Starting health check server..."
python -c "
import uvicorn
from fastapi import FastAPI
from datetime import datetime
import asyncio
import torch

app = FastAPI()

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'service': 'voxtral-realtime'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, log_level='warning')
" &

# Wait a moment for health server to start
sleep 2

# Start main application
echo "🎤 Starting Voxtral Real-Time Server..."
cd /app

# Run with uvicorn for production
exec uvicorn src.main:app \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --workers 1 \\
    --loop uvloop \\
    --http h11 \\
    --ws websockets \\
    --log-level info \\
    --access-log \\
    --no-use-colors
