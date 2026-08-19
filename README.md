# WanVideo 2.1 — RunPod Serverless Worker

Text-to-video generation using [Wan-AI/Wan2.1](https://huggingface.co/Wan-AI) deployed as a RunPod serverless endpoint.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container definition |
| `requirements.txt` | Python dependencies |
| `download_model.py` | Downloads model weights at build time |
| `handler.py` | RunPod serverless handler (inference logic) |

## Model Variants

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| `Wan-AI/Wan2.1-T2V-1.3B` | 16GB | Fast | Good |
| `Wan-AI/Wan2.1-T2V-14B` | 80GB | Slow | Best |

Default is **1.3B** (works on A10G / RTX 4090). Set `WAN_MODEL_ID` env var to use 14B.

## Build & Push

```bash
# 1. Build the image (downloads model weights inside — ~10GB for 1.3B)
cd runpod-worker
docker build -t your-dockerhub-username/wan2-worker:latest .

# 2. Push to Docker Hub (or any registry RunPod supports)
docker push your-dockerhub-username/wan2-worker:latest
```

If you want to skip baking the model into the image (smaller image, longer cold start):
- Comment out the `RUN python download_model.py` line in `Dockerfile`
- Set `WAN_MODEL_ID` env var in the RunPod template to download at runtime

## RunPod Setup

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Custom** → paste your Docker image URL
4. Recommended GPU: **A10G (24GB)** for 1.3B, **A100 (80GB)** for 14B
5. Set environment variables if needed:
   ```
   WAN_MODEL_ID=Wan-AI/Wan2.1-T2V-1.3B
   WAN_MODEL_DIR=/app/models/wan2.1
   HF_TOKEN=hf_...  # only needed for gated models
   ```
6. Copy the **Endpoint ID** → paste into `.env` as `RUNPOD_ENDPOINT_ID`
7. Generate a RunPod API key → paste into `.env` as `RUNPOD_API_KEY`

## Input Schema

```json
{
  "input": {
    "prompt": "A cinematic vertical shot of a runner at dawn, golden light, motion blur",
    "negative_prompt": "blurry, low quality, watermark",
    "duration_seconds": 5,
    "width": 576,
    "height": 1024,
    "fps": 24,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": -1
  }
}
```

## Output Schema

```json
{
  "video_base64": "<base64 MP4 string>",
  "width": 576,
  "height": 1024,
  "fps": 24,
  "num_frames": 120,
  "duration_seconds": 5,
  "seed": 98765
}
```

The server decodes `video_base64` and streams the MP4 back to the client.

## Notes

- Width and height are automatically rounded to the nearest multiple of 16
- Maximum duration is 10 seconds (257 frames limit in WanVideo)
- The model is loaded once at cold start and reused across jobs (warm instances)
- VRAM is freed after each job with `torch.cuda.empty_cache()`
