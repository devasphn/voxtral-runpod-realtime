# Voxtral Mini 3B RunPod Deployment
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    ffmpeg \\
    libsndfile1 \\
    wget \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/temp /app/models

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Expose ports
EXPOSE 8000 8080 8765 8766

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./scripts/start_server.sh"]
