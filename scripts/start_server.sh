#!/bin/bash
# FIXED ENHANCED VOXTRAL STARTUP SCRIPT - WITH PROPER PORT HANDLING AND CLEANUP

set -e

# Enhanced signal handlers for proper cleanup
cleanup_enhanced() {
    echo "🛑 Received shutdown signal, performing enhanced cleanup..."
    echo "🧹 Cleaning up enhanced server processes..."
    
    # Kill any running processes
    if [[ -n "${HEALTH_SERVER_PID:-}" ]]; then
        kill -TERM "$HEALTH_SERVER_PID" 2>/dev/null || true
        wait "$HEALTH_SERVER_PID" 2>/dev/null || true
        echo "✅ Health server stopped (PID: $HEALTH_SERVER_PID)"
    fi
    
    if [[ -n "${MAIN_SERVER_PID:-}" ]]; then
        kill -TERM "$MAIN_SERVER_PID" 2>/dev/null || true
        wait "$MAIN_SERVER_PID" 2>/dev/null || true
        echo "✅ Main server stopped (PID: $MAIN_SERVER_PID)"
    fi
    
    # Kill any lingering FFmpeg processes
    pkill -f "ffmpeg.*webm" 2>/dev/null || true
    
    # Kill any lingering Python processes related to our app
    pkill -f "uvicorn.*src.main:app" 2>/dev/null || true
    
    echo "✅ Enhanced cleanup completed"
    echo "✅ Enhanced server shutdown completed"
    exit 0
}

# Set up signal handlers
trap cleanup_enhanced SIGINT SIGTERM EXIT

echo "🚀 Starting ENHANCED Voxtral Mini 3B Real-Time Server..."
echo "=========================================================="

# Get current working directory
WORK_DIR=$(pwd)
echo "Working Directory: $WORK_DIR"

# Set environment variables relative to current directory
export PYTHONPATH="$WORK_DIR:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="$WORK_DIR/models"
export TRANSFORMERS_CACHE="$WORK_DIR/models"
export TORCH_HOME="$WORK_DIR/models"

# Log enhanced system information
echo "📊 Enhanced System Information:"
echo "- GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "- GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- Working Directory: $WORK_DIR"

# Create necessary directories relative to current path
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/temp" "$WORK_DIR/models" "$WORK_DIR/conversations"
echo "✅ Created directories: logs, temp, models, conversations"

# Set permissions for scripts in current directory
chmod 755 "$WORK_DIR/scripts"/*.sh
echo "✅ Set script permissions"

# FIXED: Better health check server with port conflict handling
echo "🔍 Starting enhanced health check server on port 8005..."

# Check if port is already in use and kill existing process
if lsof -ti:8005 >/dev/null 2>&1; then
    echo "⚠️ Port 8005 is in use, killing existing processes..."
    lsof -ti:8005 | xargs kill -TERM 2>/dev/null || true
    sleep 2
fi

# Start health check server
python -c "
import uvicorn
import asyncio
import signal
import sys
import os
from fastapi import FastAPI
from datetime import datetime
import torch

# Set up working directory
sys.path.insert(0, '$WORK_DIR')

app = FastAPI()

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'service': 'voxtral-realtime-enhanced',
        'working_directory': '$WORK_DIR',
        'features': [
            'Conversation Memory',
            'Enhanced Speech Detection',
            'Robust Shutdown',
            'Multilingual Support'
        ]
    }

def signal_handler(signum, frame):
    print(f'Enhanced health server received signal {signum}, shutting down...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    try:
        uvicorn.run(app, host='0.0.0.0', port=8005, log_level='warning', access_log=False)
    except Exception as e:
        print(f'Health server failed: {e}')
        sys.exit(1)
" &

HEALTH_SERVER_PID=$!

# Wait for health server to start
sleep 3
echo "✅ Enhanced health check server started on port 8005 (PID: $HEALTH_SERVER_PID)"

# Start main application
echo "🎤 Starting Enhanced Voxtral Real-Time Server on port 8000..."
echo "Features: Conversation Memory | Robust Shutdown | Multilingual Support | Enhanced Detection"

# Change to working directory and start main app
cd "$WORK_DIR"

# Start main server with enhanced features
uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --http h11 \
    --ws websockets \
    --log-level info \
    --access-log \
    --no-use-colors &

MAIN_SERVER_PID=$!
echo "🚀 Enhanced main server started (PID: $MAIN_SERVER_PID)"

# Wait for main server process
wait $MAIN_SERVER_PID

# Check if main server died unexpectedly
if ! kill -0 $MAIN_SERVER_PID 2>/dev/null; then
    echo "❌ Main server process died unexpectedly"
    cleanup_enhanced
fi
