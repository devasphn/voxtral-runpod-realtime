#!/bin/bash
# VOXTRAL UNDERSTANDING-ONLY DEPLOYMENT SCRIPT

set -e

echo "🧠 Deploying Voxtral UNDERSTANDING-ONLY Real-Time System..."
echo "============================================================="

# Get current directory
WORK_DIR=$(pwd)
echo "Working Directory: $WORK_DIR"

# Function to backup original files
backup_files() {
    echo "📦 Creating backup of original files..."
    
    BACKUP_DIR="$WORK_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup original files
    [ -f "src/main.py" ] && cp "src/main.py" "$BACKUP_DIR/main.py.backup"
    [ -f "src/model_loader.py" ] && cp "src/model_loader.py" "$BACKUP_DIR/model_loader.py.backup"
    [ -f "src/audio_processor.py" ] && cp "src/audio_processor.py" "$BACKUP_DIR/audio_processor.py.backup"
    [ -f "config/settings.py" ] && cp "config/settings.py" "$BACKUP_DIR/settings.py.backup"
    [ -f "static/index.html" ] && cp "static/index.html" "$BACKUP_DIR/index.html.backup"
    [ -f "static/websocket_client.js" ] && cp "static/websocket_client.js" "$BACKUP_DIR/websocket_client.js.backup"
    [ -f "static/style.css" ] && cp "static/style.css" "$BACKUP_DIR/style.css.backup"
    
    echo "✅ Backup created at: $BACKUP_DIR"
}

# Function to deploy understanding-only files
deploy_understanding_files() {
    echo "🚀 Deploying UNDERSTANDING-ONLY files..."
    
    # Check if understanding-only files exist
    ASSETS_DIR="$WORK_DIR"
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-main.py" ]; then
        echo "📄 Deploying main.py..."
        cp "$ASSETS_DIR/voxtral-understanding-main.py" "src/main.py"
        echo "✅ Updated src/main.py"
    else
        echo "❌ voxtral-understanding-main.py not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-model.py" ]; then
        echo "📄 Deploying model_loader.py..."
        cp "$ASSETS_DIR/voxtral-understanding-model.py" "src/model_loader.py"
        echo "✅ Updated src/model_loader.py"
    else
        echo "❌ voxtral-understanding-model.py not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-audio.py" ]; then
        echo "📄 Deploying audio_processor.py..."
        cp "$ASSETS_DIR/voxtral-understanding-audio.py" "src/audio_processor.py"
        echo "✅ Updated src/audio_processor.py"
    else
        echo "❌ voxtral-understanding-audio.py not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-settings.py" ]; then
        echo "📄 Deploying settings.py..."
        cp "$ASSETS_DIR/voxtral-understanding-settings.py" "config/settings.py"
        echo "✅ Updated config/settings.py"
    else
        echo "❌ voxtral-understanding-settings.py not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-index.html" ]; then
        echo "📄 Deploying index.html..."
        cp "$ASSETS_DIR/voxtral-understanding-index.html" "static/index.html"
        echo "✅ Updated static/index.html"
    else
        echo "❌ voxtral-understanding-index.html not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-client.js" ]; then
        echo "📄 Deploying websocket_client.js..."
        cp "$ASSETS_DIR/voxtral-understanding-client.js" "static/websocket_client.js"
        echo "✅ Updated static/websocket_client.js"
    else
        echo "❌ voxtral-understanding-client.js not found"
        exit 1
    fi
    
    if [ -f "$ASSETS_DIR/voxtral-understanding-style.css" ]; then
        echo "📄 Deploying style.css..."
        cp "$ASSETS_DIR/voxtral-understanding-style.css" "static/style.css"
        echo "✅ Updated static/style.css"
    else
        echo "❌ voxtral-understanding-style.css not found"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing UNDERSTANDING-ONLY dependencies..."
    
    # Check if requirements.txt exists and install
    if [ -f "requirements.txt" ]; then
        pip install --no-cache-dir -r requirements.txt
        echo "✅ Requirements installed"
    fi
    
    # Install additional dependencies for understanding-only mode
    pip install --no-cache-dir webrtcvad numpy scipy
    echo "✅ Additional dependencies installed"
}

# Function to update ports configuration
update_ports() {
    echo "🔧 Updating port configuration..."
    
    # Update deployment config if it exists
    if [ -f "deployment/runpod_config.json" ]; then
        # Backup original
        cp "deployment/runpod_config.json" "deployment/runpod_config.json.backup"
        
        # Update to understanding-only ports
        cat > "deployment/runpod_config.json" << 'EOF'
{
    "name": "voxtral-understanding-only-realtime",
    "description": "Voxtral Mini 3B UNDERSTANDING-ONLY Real-Time with 0.3s Gap Detection",
    "version": "2.0.0-UNDERSTANDING-ONLY",
    
    "hardware": {
        "gpu_type": "NVIDIA A40",
        "gpu_count": 1,
        "vram_gb": 48,
        "cpu_cores": 8,
        "memory_gb": 32,
        "container_disk_gb": 50,
        "volume_disk_gb": 30
    },
    
    "ports": {
        "http": [
            {
                "internal": 8000,
                "description": "FastAPI UNDERSTANDING-ONLY Application"
            },
            {
                "internal": 8005,
                "description": "Health Check Endpoint"
            }
        ],
        "tcp": [
            {
                "internal": 8766,
                "description": "WebSocket UNDERSTANDING-ONLY Service"
            }
        ]
    },
    
    "environment": {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/app",
        "HF_HOME": "/app/models",
        "TRANSFORMERS_CACHE": "/app/models",
        "TORCH_HOME": "/app/models",
        "LOG_LEVEL": "INFO",
        "MAX_CONCURRENT_CONNECTIONS": "10",
        "MODEL_NAME": "mistralai/Voxtral-Mini-3B-2507",
        "GAP_THRESHOLD_MS": "300",
        "TARGET_RESPONSE_MS": "200",
        "UNDERSTANDING_ONLY": "true"
    },
    
    "deployment_notes": [
        "✅ UNDERSTANDING-ONLY mode with 0.3s gap detection",
        "✅ Sub-200ms response time optimization",
        "✅ Single WebSocket endpoint for conversational AI",
        "✅ Enhanced audio processing for human speech",
        "✅ Context-aware conversation memory",
        "✅ Multilingual support with auto-detection",
        "✅ Real-time gap detection using WebRTC VAD"
    ]
}
EOF
        echo "✅ Updated runpod_config.json for UNDERSTANDING-ONLY"
    fi
}

# Function to cleanup and restart
restart_services() {
    echo "🔄 Restarting services..."
    
    # Stop existing processes
    echo "🛑 Stopping existing processes..."
    pkill -f "uvicorn.*main:app" -SIGKILL 2>/dev/null || true
    pkill -f "python.*main.py" -SIGKILL 2>/dev/null || true
    sleep 2
    
    # Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# Function to start understanding-only server
start_server() {
    echo "🚀 Starting UNDERSTANDING-ONLY server..."
    
    # Set environment variables
    export PYTHONPATH="$WORK_DIR:${PYTHONPATH}"
    export UNDERSTANDING_ONLY="true"
    export GAP_THRESHOLD_MS="300"
    export TARGET_RESPONSE_MS="200"
    
    # Start the server
    if [ -f "scripts/start_server.sh" ]; then
        echo "Using existing start script..."
        chmod +x scripts/start_server.sh
        ./scripts/start_server.sh
    else
        echo "Starting server directly..."
        cd "$WORK_DIR"
        python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload &
        
        # Wait a bit for server to start
        sleep 5
        
        # Check if server is running
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ UNDERSTANDING-ONLY server started successfully!"
            echo "🌐 Access at: http://localhost:8000"
            echo "🧠 Understanding endpoint: ws://localhost:8000/ws/understand"
        else
            echo "❌ Server failed to start"
            exit 1
        fi
    fi
}

# Main deployment function
main() {
    echo "Starting UNDERSTANDING-ONLY deployment..."
    
    # Check if we're in the right directory
    if [ ! -f "src/main.py" ] && [ ! -f "config/settings.py" ]; then
        echo "❌ Not in Voxtral project directory. Please run from project root."
        exit 1
    fi
    
    # Create backup
    backup_files
    
    # Deploy understanding-only files
    deploy_understanding_files
    
    # Install dependencies
    install_dependencies
    
    # Update configuration
    update_ports
    
    # Restart services
    restart_services
    
    # Start server
    start_server
    
    echo ""
    echo "🎉 UNDERSTANDING-ONLY deployment completed successfully!"
    echo "============================================================="
    echo "✅ Mode: UNDERSTANDING-ONLY conversational AI"
    echo "✅ Gap Detection: 0.3 seconds"
    echo "✅ Response Target: Sub-200ms"
    echo "✅ WebSocket: /ws/understand"
    echo "✅ Web Interface: http://localhost:8000"
    echo ""
    echo "🧠 Ready for real-time conversational AI interaction!"
}

# Run main function
main "$@"
