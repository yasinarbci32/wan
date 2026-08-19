"""
download_model.py — Pre-downloads WanVideo 2.1 model weights at Docker build time.
Run: python download_model.py
This bakes the weights into the image to eliminate cold-start download delays.

Model variants:
  Wan-AI/Wan2.1-T2V-14B  — 14B params, requires 80GB VRAM (A100/H100)
  Wan-AI/Wan2.1-T2V-1.3B — 1.3B params, runs on 16GB VRAM (A10G, RTX 4090)

Set WAN_MODEL_ID env var to override the default.
"""

import os
from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B")
MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/app/models/wan2.1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Downloading model: {MODEL_ID}")
print(f"Destination: {MODEL_DIR}")

os.makedirs(MODEL_DIR, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    token=HF_TOKEN or None,
    ignore_patterns=["*.md", "*.txt", "*.png", "*.jpg", ".gitattributes"],
)

print(f"Model downloaded to {MODEL_DIR}")
