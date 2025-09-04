#!/bin/bash
# FIXED: Enhanced Voxtral Mini 3B Server Startup Script with Robust Shutdown

set -e

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

# Enhanced system information
echo "📊 Enhanced System Information:"
echo "- GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "- GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- Working Directory: $WORK_DIR"

# Create necessary directories with enhanced structure
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/temp" "$WORK_DIR/models" "$WORK_DIR/conversations"
echo "✅ Created directories: logs, temp, models, conversations"

# Set permissions for scripts
chmod 755 "$WORK_DIR/scripts"/*.sh
echo "✅ Set script permissions"

# Create cleanup script for proper shutdown
cat > "$WORK_DIR/cleanup.sh" << 'EOF'
#!/bin/bash
echo "🧹 Cleaning up enhanced server processes..."

# Kill uvicorn processes gracefully
pkill -f "uvicorn.*src.main:app" -SIGTERM 2>/dev/null || true

# Wait for graceful shutdown
sleep 5

# Force kill if still running
pkill -f "uvicorn.*src.main:app" -SIGKILL 2>/dev/null || true

# Kill any remaining FFmpeg processes
pkill -f "ffmpeg.*webm" 2>/dev/null || true

# Clean up temp files
rm -rf /tmp/tmp*.wav 2>/dev/null || true
rm -f "$WORK_DIR/cleanup.sh" 2>/dev/null || true

echo "✅ Enhanced cleanup completed"
EOF

chmod +x "$WORK_DIR/cleanup.sh"

# Create signal handler for proper cleanup
cleanup_and_exit() {
    echo "🛑 Received shutdown signal, performing enhanced cleanup..."
    
    # Run cleanup script
    if [ -f "$WORK_DIR/cleanup.sh" ]; then
        "$WORK_DIR/cleanup.sh"
    fi
    
    echo "✅ Enhanced server shutdown completed"
    exit 0
}

# Register signal handlers for robust shutdown
trap cleanup_and_exit SIGINT SIGTERM EXIT

# Check if port 8005 is available for health server
if command -v netstat >/dev/null 2>&1; then
    if netstat -tuln 2>/dev/null | grep ":8005 " > /dev/null; then
        echo "⚠️ Port 8005 is already in use, skipping health server"
    else
        start_health_server=true
    fi
else
    # netstat not available, try to start health server anyway
    start_health_server=true
fi

if [ "$start_health_server" = true ]; then
    # Enhanced health check endpoint (background) on port 8005
    echo "🔍 Starting enhanced health check server on port 8005..."
    python3 -c "
import uvicorn
import asyncio
from fastapi import FastAPI
from datetime import datetime
import torch
import sys
import signal
import os

# Add current directory to Python path
sys.path.insert(0, '$WORK_DIR')

app = FastAPI()

# Shutdown handler for health server
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    print('🛑 Health server received shutdown signal')
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'service': 'voxtral-realtime-enhanced',
        'working_directory': '$WORK_DIR',
        'enhancements': [
            'Conversation Memory',
            'Robust Shutdown',
            'Multilingual Support',
            'Enhanced Speech Detection'
        ]
    }

@app.get('/shutdown')
async def trigger_shutdown():
    print('🛑 Health server shutdown requested')
    shutdown_event.set()
    return {'status': 'shutting_down'}

async def run_server():
    config = uvicorn.Config(app, host='0.0.0.0', port=8005, log_level='warning')
    server = uvicorn.Server(config)
    
    # Run until shutdown signal
    task = asyncio.create_task(server.serve())
    await shutdown_event.wait()
    
    # Graceful shutdown
    server.should_exit = True
    await task
    print('✅ Health server shutdown completed')

if __name__ == '__main__':
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
" &
    
    HEALTH_PID=$!
    sleep 3
    echo "✅ Enhanced health check server started on port 8005 (PID: $HEALTH_PID)"
fi

# Enhanced server start with better error handling
echo "🎤 Starting Enhanced Voxtral Real-Time Server on port 8000..."
echo "Features: Conversation Memory | Robust Shutdown | Multilingual Support | Enhanced Detection"

# Change to working directory
cd "$WORK_DIR"

# Function to check if server is still running
check_server() {
    if ! kill -0 $MAIN_PID 2>/dev/null; then
        echo "❌ Main server process died unexpectedly"
        cleanup_and_exit
    fi
}

# Start main server with enhanced configuration
python3 -c "
import sys
import asyncio
import signal
import logging
sys.path.insert(0, '$WORK_DIR')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced main
from src.main import app
import uvicorn

def signal_handler(signum, frame):
    logger.info(f'🛑 Received signal {signum}, shutting down enhanced server...')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    try:
        uvicorn.run(
            'src.main:app',
            host='0.0.0.0',
            port=8000,
            workers=1,
            loop='uvloop',
            http='h11',
            ws='websockets',
            log_level='info',
            access_log=True,
            timeout_graceful_shutdown=30,
            timeout_keep_alive=65
        )
    except KeyboardInterrupt:
        logger.info('🛑 Enhanced server stopped by user')
    except Exception as e:
        logger.error(f'❌ Enhanced server error: {e}')
        sys.exit(1)
" &

MAIN_PID=$!
echo "🚀 Enhanced main server started (PID: $MAIN_PID)"

# Monitor server health
while true; do
    sleep 10
    check_server
done
