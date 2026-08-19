# WanVideo 2.1 RunPod Serverless Worker
#
# Base: RunPod PyTorch CUDA 12.1 image
# Model: Wan-AI/Wan2.1-T2V-14B (text-to-video, 14B params)
# VRAM: Requires 24GB+ (A100/H100 recommended for 14B; use 1.3B for smaller GPUs)

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .
COPY download_model.py .

# Download model weights at build time (bakes into image, avoids cold start)
# Comment this out if you want to download at runtime instead (saves image size)
RUN python download_model.py

CMD ["python", "-u", "handler.py"]
