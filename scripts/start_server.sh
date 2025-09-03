#!/bin/bash
# Voxtral Mini 3B Server Startup Script - RunPod Fixed Version

set -e

echo "🚀 Starting Voxtral Mini 3B Real-Time Server..."
echo "=================================================="

# Get current working directory
WORK_DIR=$(pwd)
echo "Working Directory: $WORK_DIR"

# Set environment variables relative to current directory
export PYTHONPATH="$WORK_DIR:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="$WORK_DIR/models"
export TRANSFORMERS_CACHE="$WORK_DIR/models"
export TORCH_HOME="$WORK_DIR/models"

# Log system information
echo "📊 System Information:"
echo "- GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "- GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- Working Directory: $WORK_DIR"

# Create necessary directories relative to current path
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/temp" "$WORK_DIR/models"
echo "✅ Created directories: logs, temp, models"

# Set permissions for scripts in current directory
chmod 755 "$WORK_DIR/scripts"/*.sh
echo "✅ Set script permissions"

# Check if port 8005 is available
if netstat -tuln | grep ":8005 " > /dev/null; then
    echo "⚠️ Port 8005 is already in use, skipping health server"
else
    # Health check endpoint (background) on port 8005
    echo "🔍 Starting health check server on port 8005..."
    python -c "
import uvicorn
from fastapi import FastAPI
from datetime import datetime
import torch
import sys

# Add current directory to Python path
sys.path.insert(0, '$WORK_DIR')

app = FastAPI()

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'service': 'voxtral-realtime',
        'working_directory': '$WORK_DIR'
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8005, log_level='warning')
" &
    
    # Wait for health server to start
    sleep 3
    echo "✅ Health check server started on port 8005"
fi

# Start main application
echo "🎤 Starting Voxtral Real-Time Server on port 8000..."

# Change to working directory and start main app
cd "$WORK_DIR"

# Run with uvicorn for production
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --http h11 \
    --ws websockets \
    --log-level info \
    --access-log \
    --no-use-colors
