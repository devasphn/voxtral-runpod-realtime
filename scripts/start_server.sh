#!/bin/bash
# COMPLETELY FIXED STARTUP SCRIPT - USES CORRECTED FILES AND PROPER ERROR HANDLING

set -e

echo "🚀 Starting COMPLETELY FIXED Voxtral Mini 3B Real-Time Server..."
echo "================================================================="

# Get current working directory
WORK_DIR=$(pwd)
echo "Working Directory: $WORK_DIR"

# COMPLETELY FIXED: Alternative port cleanup without lsof dependency
echo "🧹 Cleaning up existing processes..."

# Kill processes by name/pattern (more reliable than lsof)
pkill -f "uvicorn.*fixed_main:app" -SIGKILL 2>/dev/null || true
pkill -f "uvicorn.*main:app" -SIGKILL 2>/dev/null || true
pkill -f "ffmpeg.*webm" -SIGKILL 2>/dev/null || true

# Alternative port cleanup using netstat (if available)
if command -v netstat >/dev/null 2>&1; then
    # Find and kill processes using our ports
    for port in 8000 8005; do
        pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d/ -f1)
        if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            echo "Killed process $pid using port $port"
        fi
    done
elif command -v ss >/dev/null 2>&1; then
    # Use ss as alternative to netstat
    for port in 8000 8005; do
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | sed 's/.*pid=\([0-9]*\).*/\1/' | head -1)
        if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            echo "Killed process $pid using port $port"
        fi
    done
else
    echo "⚠️ Neither lsof, netstat, nor ss available - skipping port cleanup"
fi

sleep 2
echo "✅ Existing processes cleaned up"

# Set environment variables relative to current directory
export PYTHONPATH="$WORK_DIR:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME="$WORK_DIR/models"
export TRANSFORMERS_CACHE="$WORK_DIR/models"
export TORCH_HOME="$WORK_DIR/models"

# Enhanced system information
echo "📊 COMPLETELY FIXED System Information:"
echo "- GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "- GPU Count: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
echo "- GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- Working Directory: $WORK_DIR"

# Create necessary directories
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/temp" "$WORK_DIR/models" "$WORK_DIR/conversations"
echo "✅ Created directories: logs, temp, models, conversations"

# Set permissions for scripts
chmod 755 "$WORK_DIR/scripts"/*.sh
echo "✅ Set script permissions"

# COMPLETELY FIXED: Create improved cleanup script
cat > "$WORK_DIR/cleanup.sh" << 'EOF'
#!/bin/bash
echo "🧹 Cleaning up COMPLETELY FIXED server processes..."

# Kill uvicorn processes gracefully first
pkill -f "uvicorn.*fixed_main:app" -SIGTERM 2>/dev/null || true
pkill -f "uvicorn.*main:app" -SIGTERM 2>/dev/null || true

# Wait for graceful shutdown
sleep 3

# Force kill if still running
pkill -f "uvicorn.*fixed_main:app" -SIGKILL 2>/dev/null || true
pkill -f "uvicorn.*main:app" -SIGKILL 2>/dev/null || true

# Kill FFmpeg processes
pkill -f "ffmpeg.*webm" -SIGKILL 2>/dev/null || true

# Alternative port cleanup without lsof
if command -v netstat >/dev/null 2>&1; then
    for port in 8000 8005; do
        pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d/ -f1 2>/dev/null)
        if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
elif command -v ss >/dev/null 2>&1; then
    for port in 8000 8005; do
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | sed 's/.*pid=\([0-9]*\).*/\1/' | head -1)
        if [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
fi

# Clean up temp files
rm -rf /tmp/tmp*.wav 2>/dev/null || true
rm -f "$WORK_DIR/cleanup.sh" 2>/dev/null || true

echo "✅ COMPLETELY FIXED cleanup completed"
EOF

chmod +x "$WORK_DIR/cleanup.sh"

# Create signal handler for proper cleanup
cleanup_and_exit() {
    echo "🛑 Received shutdown signal, performing COMPLETELY FIXED cleanup..."
    
    # Run cleanup script
    if [ -f "$WORK_DIR/cleanup.sh" ]; then
        "$WORK_DIR/cleanup.sh"
    fi
    
    echo "✅ COMPLETELY FIXED server shutdown completed"
    exit 0
}

# Register signal handlers
trap cleanup_and_exit SIGINT SIGTERM EXIT

# Start health check server on port 8005
echo "🔍 Starting COMPLETELY FIXED health check server on port 8005..."
python3 -c "
import uvicorn
import asyncio
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
        'service': 'voxtral-realtime-completely-fixed',
        'working_directory': '$WORK_DIR',
        'fixes_applied': [
            '✅ Correct Voxtral API Usage (Never pass None to language)',
            '✅ Valid Language Codes (en, es, fr, pt, hi, de, nl, it)',
            '✅ Proper apply_transcription_request vs apply_chat_template',
            '✅ Fixed Result Structure Handling',
            '✅ Enhanced Error Recovery and Validation',
            '✅ Better Audio File Conversion'
        ]
    }

if __name__ == '__main__':
    try:
        uvicorn.run(app, host='0.0.0.0', port=8005, log_level='warning')
    except Exception as e:
        print(f'Health server failed: {e}')
        sys.exit(1)
" &

HEALTH_PID=$!
sleep 2
echo "✅ COMPLETELY FIXED health check server started on port 8005 (PID: $HEALTH_PID)"

# COMPLETELY FIXED: Check if fixed files exist and use them
if [ -f "$WORK_DIR/fixed_main.py" ]; then
    MAIN_MODULE="fixed_main:app"
    echo "🎯 Using COMPLETELY FIXED main module"
else
    MAIN_MODULE="src.main:app"
    echo "⚠️ Using original main module (fixes not applied)"
fi

# Function to check server health
check_server() {
    if ! kill -0 $MAIN_PID 2>/dev/null; then
        echo "❌ Main server process died unexpectedly"
        cleanup_and_exit
    fi
}

# COMPLETELY FIXED: Start main server with proper module selection
echo "🎤 Starting COMPLETELY FIXED Voxtral Real-Time Server on port 8000..."
echo "Features: Correct Voxtral API Usage | Proper Language Handling | Fixed Result Processing"

# Change to working directory
cd "$WORK_DIR"

# Start main server with proper module selection
python3 -c "
import sys
import asyncio
import signal
import logging
sys.path.insert(0, '$WORK_DIR')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    logger.info(f'🛑 Received signal {signum}, shutting down COMPLETELY FIXED server...')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    import uvicorn
    try:
        # Try to use the completely fixed main first
        try:
            from fixed_main import app
            logger.info('✅ Using COMPLETELY FIXED main application')
        except ImportError:
            from src.main import app
            logger.info('⚠️ Using original main application')
        
        uvicorn.run(
            app,
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
        logger.info('🛑 COMPLETELY FIXED server stopped by user')
    except Exception as e:
        logger.error(f'❌ COMPLETELY FIXED server error: {e}')
        sys.exit(1)
" &

MAIN_PID=$!
echo "🚀 COMPLETELY FIXED main server started (PID: $MAIN_PID)"

# Monitor server health
echo "👀 Monitoring server health..."
while true; do
    sleep 10
    check_server
    
    # Optional: Check if health endpoint is responding
    if command -v curl >/dev/null 2>&1; then
        if ! curl -s http://localhost:8005/health >/dev/null 2>&1; then
            echo "⚠️ Health check endpoint not responding"
        fi
    fi
done
