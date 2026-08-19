"""
handler.py — RunPod Serverless Worker for WanVideo 2.1 Text-to-Video

Input schema:
{
  "prompt": "A cinematic shot of...",
  "negative_prompt": "blurry, low quality",   # optional
  "duration_seconds": 5,                       # 1–10
  "width": 576,                                # must be divisible by 16
  "height": 1024,                              # must be divisible by 16
  "fps": 24,                                   # frames per second
  "num_inference_steps": 50,                   # default 50
  "guidance_scale": 7.5,                       # default 7.5
  "seed": -1                                   # -1 = random
}

Output:
{
  "video_base64": "<base64-encoded mp4>",
  "width": 576,
  "height": 1024,
  "fps": 24,
  "num_frames": 120,
  "duration_seconds": 5,
  "seed": 12345
}
"""

import os
import sys
import gc
import random
import base64
import tempfile

import runpod
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR    = os.environ.get("WAN_MODEL_DIR", "/app/models/wan2.1")
MODEL_ID     = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_DURATION = 10   # seconds
MAX_FRAMES   = 257  # WanVideo hard limit

print(f"[init] DEVICE={DEVICE}, DTYPE={DTYPE}, MODEL_DIR={MODEL_DIR}")

# ---------------------------------------------------------------------------
# Download model on cold start if not cached
# ---------------------------------------------------------------------------

def ensure_model_downloaded():
    """Download model weights if not already present (e.g. first cold start or no network volume)."""
    marker = os.path.join(MODEL_DIR, ".download_complete")
    if os.path.exists(marker):
        print(f"[init] Model already cached at {MODEL_DIR}")
        return

    print(f"[init] Model not found. Downloading {MODEL_ID} to {MODEL_DIR}...")
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        token=HF_TOKEN or None,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    # Write marker so we skip download on next warm start
    with open(marker, "w") as f:
        f.write("ok")
    print(f"[init] Download complete.")

# Run download check at import time (before first job)
ensure_model_downloaded()

# ---------------------------------------------------------------------------
# Lazy model loading — loaded once at cold start, reused across jobs
# ---------------------------------------------------------------------------

_pipeline = None

def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("[init] Loading WanVideo pipeline...")

    try:
        # Try loading via the official wan package if present
        from wan.pipeline import WanT2V
        _pipeline = WanT2V.from_pretrained(
            MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_ID,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        print("[init] Loaded via wan.pipeline")
    except ImportError:
        # Fallback: use diffusers pipeline (works with Wan2.1 checkpoints)
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from diffusers.pipelines.wan.pipeline_wan import WanPipeline
        from transformers import CLIPTextModel, AutoTokenizer

        model_path = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "model_index.json")) else MODEL_ID
        _pipeline = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        print("[init] Loaded via diffusers WanPipeline")

    if hasattr(_pipeline, 'enable_model_cpu_offload'):
        _pipeline.enable_model_cpu_offload()

    _pipeline.set_progress_bar_config(disable=True)
    print("[init] Pipeline ready")
    return _pipeline


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def round16(x):
    """Round to nearest multiple of 16 (required by WanVideo)."""
    return max(16, int(x) // 16 * 16)

def validate_input(job_input: dict) -> tuple[dict, str | None]:
    """Returns (validated_params, error_string_or_None)."""
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        return {}, "prompt is required and must be non-empty"

    duration = clamp(float(job_input.get("duration_seconds", 5)), 1, MAX_DURATION)
    fps      = clamp(int(job_input.get("fps", 24)), 8, 60)
    width    = round16(job_input.get("width", 576))
    height   = round16(job_input.get("height", 1024))
    steps    = clamp(int(job_input.get("num_inference_steps", 50)), 10, 100)
    guidance = clamp(float(job_input.get("guidance_scale", 7.5)), 1.0, 20.0)
    seed     = int(job_input.get("seed", -1))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    num_frames = min(int(duration * fps), MAX_FRAMES)

    return {
        "prompt":               prompt,
        "negative_prompt":      job_input.get("negative_prompt", "blurry, low quality, watermark, text, distorted"),
        "duration_seconds":     duration,
        "fps":                  fps,
        "width":                width,
        "height":               height,
        "num_inference_steps":  steps,
        "guidance_scale":       guidance,
        "seed":                 seed,
        "num_frames":           num_frames,
    }, None


# ---------------------------------------------------------------------------
# Video encoding helper
# ---------------------------------------------------------------------------

def frames_to_mp4_base64(frames: list, fps: int, width: int, height: int) -> str:
    """Convert list of numpy/PIL frames to base64-encoded MP4."""
    import imageio
    from PIL import Image as PILImage

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            output_params=["-vf", f"scale={width}:{height}", "-movflags", "+faststart"],
        )
        for frame in frames:
            if hasattr(frame, 'numpy'):
                frame = frame.numpy()
            if isinstance(frame, np.ndarray):
                frame = (frame * 255).clip(0, 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            else:
                frame = np.array(frame)
            writer.append_data(frame)
        writer.close()

        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    # Validate
    params, error = validate_input(job_input)
    if error:
        return {"error": error}

    print(f"[job] prompt={params['prompt'][:80]}... frames={params['num_frames']} "
          f"size={params['width']}x{params['height']} steps={params['num_inference_steps']} "
          f"seed={params['seed']}")

    try:
        pipe = load_pipeline()

        # Set seed for reproducibility
        generator = torch.Generator(device=DEVICE).manual_seed(params["seed"])

        # Run inference
        with torch.inference_mode():
            output = pipe(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                height=params["height"],
                width=params["width"],
                num_frames=params["num_frames"],
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                generator=generator,
            )

        # Extract frames — diffusers returns output.frames as list of PIL or tensors
        frames = output.frames
        if isinstance(frames, torch.Tensor):
            # Shape: (1, T, H, W, C) or (T, H, W, C)
            frames = frames.squeeze(0).cpu().numpy()
            frames = [frames[i] for i in range(frames.shape[0])]
        elif isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], list):
            frames = frames[0]  # unwrap batch dimension

        video_b64 = frames_to_mp4_base64(frames, params["fps"], params["width"], params["height"])

        result = {
            "video_base64":   video_b64,
            "width":          params["width"],
            "height":         params["height"],
            "fps":            params["fps"],
            "num_frames":     len(frames),
            "duration_seconds": params["duration_seconds"],
            "seed":           params["seed"],
        }

        print(f"[job] Done. Frames: {len(frames)}, video size: {len(video_b64) // 1024}KB (b64)")
        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": "CUDA out of memory. Try reducing resolution, duration, or switching to 1.3B model."}

    except Exception as e:
        import traceback
        print(f"[job] ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Free VRAM after each job
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[worker] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
