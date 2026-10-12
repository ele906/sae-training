"""Replay SDXL samples and save the spatial SAE-input residual for each image."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLOCK = "unet.down_blocks.2.attentions.1"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "backprop/manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "residuals")
    args = parser.parse_args()
    settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    if settings["block"] != BLOCK:
        raise ValueError(f"Expected SAE block {BLOCK}, got {settings['block']}")
    checkpoint = Path(settings["sae_checkpoint"])
    weights = checkpoint / "state_dict.pth"
    if sha256(weights) != settings["sae_weights_sha256"]:
        raise ValueError("Pretrained SAE checkpoint hash mismatch")
    with args.manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1200 or len({r["sample_id"] for r in rows}) != 1200:
        raise ValueError("Expected 1,200 unique samples")
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    sys.path.insert(0, str(ROOT / "sdxl-unbox/scripts"))
    import diffusers
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder

    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().cuda().eval()
    sae.requires_grad_(False)

    started = time.monotonic()
    complete = 0
    spatial_shape = None
    for i, row in enumerate(rows):
        image_path = ROOT / "backprop" / row["image"]
        output_path = args.output / f"{row['sample_id']}.npy"
        if output_path.exists():
            complete += 1
            continue
        seed = int(row["seed"])
        noise_rng = torch.Generator("cuda").manual_seed(seed)
        scheduler_rng = torch.Generator("cuda").manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=noise_rng, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            result, cache = pipe.run_with_cache(
                prompt=row["prompt"], latents=latents, generator=scheduler_rng,
                num_inference_steps=1, guidance_scale=0.0, height=512, width=512,
                positions_to_cache=[BLOCK], save_input=True)
            original = np.asarray(Image.open(image_path).convert("RGB"))
            replay = np.asarray(result.images[0].convert("RGB"))
            if original.shape != replay.shape or not np.array_equal(original, replay):
                mae = float(np.abs(original.astype(np.float32) - replay.astype(np.float32)).mean())
                raise RuntimeError(f"Replay mismatch {row['sample_id']}: RGB MAE={mae}")
            residual = (cache["output"][BLOCK] - cache["input"][BLOCK])[:, -1].permute(0, 2, 3, 1)[0]
            residual = residual.contiguous().to(dtype=torch.float16).cpu().numpy()
        if residual.ndim != 3 or residual.shape[-1] != sae.d_model:
            raise ValueError(f"Unexpected residual shape {residual.shape} for {row['sample_id']}")
        if spatial_shape is None:
            spatial_shape = list(residual.shape)
        elif list(residual.shape) != spatial_shape:
            raise ValueError(f"Residual shape changed at {row['sample_id']}: {residual.shape}")
        np.save(output_path, residual, allow_pickle=False)
        complete += 1
        if complete == 1 or complete % 25 == 0:
            print(f"{complete}/{len(rows)} residuals saved; {time.monotonic()-started:.1f}s elapsed", flush=True)
        del cache, residual, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    meta = dict(samples=complete, shape=spatial_shape, dtype="float16", block=BLOCK,
                checkpoint=settings["sae_checkpoint"], checkpoint_sha256=settings["sae_weights_sha256"],
                manifest_sha256=sha256(args.manifest), replay="Exact RGB equality checked against source image",
                torch=torch.__version__, diffusers=diffusers.__version__)
    (args.output / "settings.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
