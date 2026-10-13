"""Ablate flower-supporting features in the diffusion model using the trained SAE."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLOCK = "unet.down_blocks.2.attentions.1"
GROUP = ("east asian", "woman")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE / "c2c_box_targets.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "backprop/manifest.csv")
    parser.add_argument("--residuals", type=Path, default=HERE / "residuals")
    parser.add_argument("--sae-checkpoint", type=Path, default=HERE / "runs/c2c_box_sae_003/sae")
    parser.add_argument("--head", type=Path, default=HERE / "runs/c2c_box_sae_003/box_head.pth")
    parser.add_argument("--output", type=Path, default=HERE / "runs/flower_intervention_001")
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--negative-limit", type=int)
    args = parser.parse_args()
    if args.features < 1:
        parser.error("--features must be positive")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh directory: {args.output}")

    import torch
    from PIL import Image, ImageDraw
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    sys.path.insert(0, str(ROOT / "sdxl-unbox/scripts"))
    from SDLens import HookedStableDiffusionXLPipeline
    from SAE import SparseAutoencoder
    from sae_experiment import intervention

    torch.set_num_threads(4)
    device = torch.device("cuda")
    sae = SparseAutoencoder.load_from_disk(str(args.sae_checkpoint)).float().to(device).eval()
    weights = torch.load(args.head, map_location=device, weights_only=True)
    head = torch.nn.Conv2d(sae.n_dirs, 10, kernel_size=1).to(device)
    head.load_state_dict(weights)
    head.eval()
    flower_weight = head.weight[5, :, 0, 0]

    targets = read_csv(args.targets)
    train_rows = [r for r in targets if r["concept"] == "flower" and r["split"] == "train"
                  and (r["race_prompt"], r["gender_prompt"]) == GROUP]
    target_by_id = {r["sample_id"]: r for r in train_rows}
    positive = [r for r in train_rows if int(r["detector_present"]) == 1]
    negative = [r for r in train_rows if int(r["detector_present"]) == 0]
    if not positive or not negative:
        raise ValueError("Need C2C-positive and negative training images in the selected prompt group")

    contribution = {0: [], 1: []}
    activation_sum = torch.zeros(sae.n_dirs, device=device)
    active_cells = torch.zeros(sae.n_dirs, device=device)
    total_cells = 0
    for row in train_rows:
        residual = torch.from_numpy(np.load(args.residuals / f"{row['sample_id']}.npy", allow_pickle=False).astype(np.float32)).to(device)
        h, w, d = residual.shape
        with torch.no_grad():
            z = sae.encode(residual.reshape(-1, d)).reshape(h, w, -1)
            activation_sum += z.sum((0, 1))
            active_cells += (z > 0).sum((0, 1))
            total_cells += h*w
            score_map = (z * flower_weight).sum(-1)
            flat_index = int(score_map.argmax())
            gy, gx = divmod(flat_index, w)
            contrib = z[gy, gx] * flower_weight
            cls = int(row["detector_present"])
            contribution[cls].append(contrib.cpu().numpy())
        del residual, z
    mean_pos = np.stack(contribution[1]).mean(0)
    mean_neg = np.stack(contribution[0]).mean(0)
    contrast = mean_pos - mean_neg
    ranked = np.argsort(-contrast)
    chosen = [int(i) for i in ranked if contrast[i] > 0 and mean_pos[i] > 0][:args.features]
    if not chosen:
        raise ValueError("No positive flower-supporting SAE features found in training data")

    act_mean = (activation_sum / total_cells).cpu().numpy()
    act_freq = (active_cells / total_cells).cpu().numpy()
    target_mean = float(np.mean(act_mean[chosen]))
    target_freq = float(np.mean(act_freq[chosen]))
    eligible = [i for i in range(sae.n_dirs) if i not in chosen and act_mean[i] > 0
                and abs(act_freq[i]-target_freq) <= 0.1
                and 0.5*target_mean <= act_mean[i] <= 2*target_mean]
    if not eligible:
        eligible = [i for i in range(sae.n_dirs) if i not in chosen and act_mean[i] > 0]
    rng = np.random.default_rng(args.seed)
    control = int(rng.choice(eligible))

    with (ROOT / "backprop/generation_settings.json").open() as f:
        settings = json.load(f)
    if settings["block"] != BLOCK:
        raise ValueError("Diffusion block does not match the trained SAE")
    manifest = {r["sample_id"]: r for r in read_csv(args.manifest)}
    test_rows = [r for r in targets if r["concept"] == "flower" and r["split"] == "test"
                 and (r["race_prompt"], r["gender_prompt"]) == GROUP]
    positives_test = [r for r in test_rows if int(r["detector_present"]) == 1]
    negatives_test = [r for r in test_rows if int(r["detector_present"]) == 0]
    if not positives_test or not negatives_test:
        raise ValueError("Need held-out positive and negative images in the selected prompt group")
    # All matched positives and negatives keep the test set independent of feature selection.
    if args.positive_limit is not None:
        positives_test = positives_test[:args.positive_limit]
    if args.negative_limit is not None:
        negatives_test = negatives_test[:args.negative_limit]
    selected = positives_test + negatives_test

    load = dict(torch_dtype=torch.float16, cache_dir=str(ROOT / "ReNO/hf_cache"), local_files_only=True)
    vae = AutoencoderKL.from_pretrained(settings["vae"], **load)
    pipe = HookedStableDiffusionXLPipeline.from_pretrained(settings["model"], vae=vae, variant="fp16", **load)
    pipe.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.pipe.scheduler.config, timestep_spacing="trailing")
    pipe.pipe.to("cuda")
    pipe.pipe.set_progress_bar_config(disable=True)

    args.output.mkdir(parents=True)
    features_csv = args.output / "feature_ranking.csv"
    with features_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "feature", "positive_contribution", "negative_contribution", "contrast", "mean_activation", "active_fraction"])
        writer.writeheader()
        for rank, j in enumerate(chosen, 1):
            writer.writerow(dict(rank=rank, feature=j, positive_contribution=mean_pos[j], negative_contribution=mean_neg[j], contrast=contrast[j], mean_activation=act_mean[j], active_fraction=act_freq[j]))
    (args.output / "settings.json").write_text(json.dumps(dict(
        sae_checkpoint=str(args.sae_checkpoint), box_head=str(args.head), block=BLOCK,
        selection_group=GROUP, train_positive=len(positive), train_negative=len(negative),
        selected_features=chosen, random_control=control, strengths=[0, 0.5, 1],
        num_inference_steps=args.num_inference_steps,
        test_positive=sum(int(r["detector_present"]) for r in selected),
        test_negative=sum(not int(r["detector_present"]) for r in selected),
        target='C2C flower boxes', interpretation='Fixed SAE feature ablation during generation; inspect pictures for object removal and collateral changes'), indent=2) + "\n")

    metrics, sheets = [], []
    for ix, target_row in enumerate(selected, 1):
        row = manifest[target_row["sample_id"]]
        seed = int(row["seed"])
        def gen_args():
            return dict(prompt=row["prompt"],
                        latents=torch.randn((1, 4, 64, 64), generator=torch.Generator("cuda").manual_seed(seed), device="cuda", dtype=torch.float16),
                        generator=torch.Generator("cuda").manual_seed(seed), num_inference_steps=args.num_inference_steps,
                        guidance_scale=0.0, height=512, width=512)
        original = Image.open(ROOT / "backprop" / row["image"]).convert("RGB")
        replay = pipe.pipe(**gen_args()).images[0]
        replay_mae = float(np.abs(np.asarray(original).astype(float)-np.asarray(replay).astype(float)).mean())
        if args.num_inference_steps == 1 and replay_mae != 0:
            raise RuntimeError(f"Source replay mismatch for {row['sample_id']}: MAE={replay_mae}")
        variants = [("source", original), ("replay", replay)]
        for alpha, label in [(0.0, "features_zero"), (0.5, "features_half"), (1.0, "features_noop")]:
            edited = pipe.run_with_hooks(**gen_args(), position_hook_dict={BLOCK: intervention(sae, chosen, alpha)}).images[0]
            delta = float(np.abs(np.asarray(edited).astype(float)-np.asarray(replay).astype(float)).mean())
            if alpha == 1 and delta != 0:
                raise RuntimeError("Strength 1 was not an exact no-op")
            edited.save(args.output / f"{row['sample_id']}_{label}.png")
            variants.append((label, edited))
            metrics.append(dict(sample_id=row["sample_id"], requested_c2c_present=target_row["detector_present"],
                                intervention="flower_features", features=";".join(map(str, chosen)), strength=alpha,
                                pixel_mae=delta, source_replay_mae=replay_mae))
        control_image = pipe.run_with_hooks(**gen_args(), position_hook_dict={BLOCK: intervention(sae, [control], 0.0)}).images[0]
        control_image.save(args.output / f"{row['sample_id']}_random_control.png")
        variants.append((f"random f{control} zero", control_image))
        control_delta = float(np.abs(np.asarray(control_image).astype(float)-np.asarray(replay).astype(float)).mean())
        metrics.append(dict(sample_id=row["sample_id"], requested_c2c_present=target_row["detector_present"],
                            intervention="matched_random_control", features=str(control), strength=0,
                            pixel_mae=control_delta, source_replay_mae=replay_mae))
        sheet = Image.new("RGB", (len(variants)*256, 300), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), f"{row['sample_id']} | C2C flower {['absent','present'][int(target_row['detector_present'])]}", fill="black")
        for j, (name, image) in enumerate(variants):
            sheet.paste(image.resize((256, 256)), (j*256, 40))
            draw.text((j*256+3, 22), name, fill="black")
        sheet.save(args.output / f"{row['sample_id']}_grid.jpg", quality=92)
        sheets.append(args.output / f"{row['sample_id']}_grid.jpg")
        if ix == 1 or ix % 5 == 0:
            print(f"{ix}/{len(selected)} completed; {row['sample_id']}", flush=True)

    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    page_rows = []
    for target_row in selected:
        path = args.output / f"{manifest[target_row['sample_id']]['sample_id']}_grid.jpg"
        page_rows.append((target_row["sample_id"], Image.open(path).convert("RGB")))
    cell_h = 300
    for page, start in enumerate(range(0, len(page_rows), 4), 1):
        subset = page_rows[start:start+4]
        overview = Image.new("RGB", (1536, len(subset)*cell_h), "white")
        for i, (_, img) in enumerate(subset):
            overview.paste(img.resize((1536, cell_h)), (0, i*cell_h))
        overview.save(args.output / f"all_test_comparisons_page_{page}.jpg", quality=90)
    test_positive_count = sum(int(r["detector_present"]) for r in selected)
    print(json.dumps(dict(output=str(args.output), features=chosen, random_control=control,
                          test_samples=len(selected), positive=test_positive_count,
                          negative=len(selected)-test_positive_count)), flush=True)


if __name__ == "__main__":
    main()
