# Voxtral Mini 3B Real-Time Streaming on RunPod

## Overview
This project deploys Mistral's Voxtral Mini 3B model on RunPod with real-time WebSocket audio streaming capabilities. The model combines state-of-the-art speech recognition and understanding in a 3B parameter package optimized for edge deployment.

## Hardware Requirements
- **GPU**: NVIDIA A40 (48GB VRAM)
- **Container Disk**: 50GB
- **Volume Disk**: 30GB  
- **Template**: PyTorch 2.8.0 + CUDA 12.4

## Port Configuration
- **HTTP Ports**: 8000 (FastAPI), 8080 (Health Check)
- **TCP Ports**: 8765 (WebSocket Streaming), 8766 (WebSocket Control)

## Quick Deploy on RunPod

### 1. Create Pod with Specifications
```bash
# Select GPU: NVIDIA A40
# Container Disk: 50GB
# Volume Disk: 30GB
# Template: PyTorch 2.8.0
# Expose Ports: 8000/http, 8080/http, 8765/tcp, 8766/tcp
```

### 2. Upload and Install
```bash
git clone <your-repo-url>
cd voxtral-runpod-realtime
chmod +x scripts/*.sh
./scripts/install_dependencies.sh
```

### 3. Start Services
```bash
./scripts/start_server.sh
```

### 4. Test Connection
- HTTP Health Check: `https://[POD-ID]-8080.proxy.runpod.net/health`  
- WebSocket Test: Open `https://[POD-ID]-8000.proxy.runpod.net` for client interface

## Features
- ✅ Real-time audio transcription
- ✅ Multi-language support (8 languages)
- ✅ 32K token context window
- ✅ Concurrent WebSocket connections
- ✅ GPU memory optimization
- ✅ Automatic model loading
- ✅ Health monitoring

## API Endpoints
- `GET /health` - Health check
- `GET /model/info` - Model information
- `WS /ws/transcribe` - Real-time transcription
- `WS /ws/understand` - Audio understanding
- `GET /` - Test client interface

## Performance
- **Model Size**: ~9.5GB GPU memory
- **Concurrent Streams**: 4-6 simultaneous connections
- **Latency**: <200ms for real-time processing
- **Supported Audio**: Up to 30 minutes for transcription
