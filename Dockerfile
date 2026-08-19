# WanVideo 2.1 RunPod Serverless Worker
#
# IMPORTANT: Model is NOT downloaded at build time.
# It downloads on first worker startup (cold start).
# Use RunPod Network Volumes to cache the model across restarts.
#
# Base: RunPod PyTorch CUDA 12.1 image (pre-installed torch, CUDA, cuDNN)
# Model: Wan-AI/Wan2.1-T2V-1.3B (default) or Wan2.1-T2V-14B

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies (ffmpeg for video encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — install in one layer, no model download
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler files
COPY handler.py .
COPY download_model.py .

# Model will be downloaded at runtime on first cold start.
# To cache across restarts, mount a RunPod Network Volume at /app/models

CMD ["python", "-u", "handler.py"]
