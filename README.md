# Voxtral Mini 3B Real-Time Streaming on RunPod

## Overview
This project deploys Mistral's Voxtral Mini 3B model on RunPod with real-time WebSocket audio streaming capabilities. The model combines state-of-the-art speech recognition and understanding in a 3B parameter package optimized for edge deployment.

## Hardware Requirements
- **GPU**: NVIDIA A40 (48GB VRAM)
- **Container Disk**: 50GB
- **Volume Disk**: 30GB  
- **Template**: PyTorch 2.8.0 + CUDA 12.4

## Port Configuration (Corrected)
- **HTTP Ports**: 8000 (FastAPI), 8005 (Health Check)
- **TCP Ports**: 8765 (WebSocket Streaming), 8766 (WebSocket Control)

## Quick Deploy on RunPod

### 1. Create Pod with Specifications
```bash
# Select GPU: NVIDIA A40
# Container Disk: 50GB
# Volume Disk: 30GB
# Template: PyTorch 2.8.0
# Expose Ports: 8000/http, 8005/http, 8765/tcp, 8766/tcp
