"""Render held-out pretrained-SAE feature maps with face and part boxes."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS, face_boxes, read_csv
from render_face_region_heatmaps import load_region_boxes

DEFAULT_MODEL = ROOT / "sdxl-unbox/checkpoints/unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
EXAMPLES = ((1664, "nose"), (527, "eyes"), (527, "ears"), (3725, "eyebrows"))
MAX_BOX_AREA = 0.12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--images", type=Path, default=ROOT / "backprop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mask_for_box(box, height, width, device="cpu"):
    x1, y1, x2, y2 = box
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (512 / width)
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (512 / height)
    return ((xs[None, :] >= x1) & (xs[None, :] <= x2)
            & (ys[:, None] >= y1) & (ys[:, None] <= y2))


def eligible_boxes(boxes, region):
    return [item for item in boxes.get(region, [])
            if (item["xyxy"][2] - item["xyxy"][0])
            * (item["xyxy"][3] - item["xyxy"][1]) / (512 * 512) <= MAX_BOX_AREA]


def choose_sample(rows, part_boxes, faces, residual_dir, region, feature, model, device, cache):
    eligible = []
    for row in rows:
        sample_id = row["sample_id"]
        part = eligible_boxes(part_boxes.get(sample_id, {}), region)
        if row["split"] == "test" and part and sample_id in faces:
            if (residual_dir / f"{sample_id}.npy").is_file():
                eligible.append((row, part))
    if not eligible:
        raise ValueError(f"No held-out example with a usable {region} box")
    best = None
    for row, target_boxes in eligible:
        sample_id = row["sample_id"]
        key = (sample_id, feature)
        if key not in cache:
            residual_np = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
            residual = torch.from_numpy(residual_np.astype(np.float32)).to(device)
            with torch.no_grad():
                cache[key] = model.encode(residual)[..., feature].float().cpu().numpy()
        values = cache[key]
        target_mask = np.zeros(values.shape, dtype=bool)
        for item in target_boxes:
            target_mask |= mask_for_box(item["xyxy"], *values.shape).numpy()
        score = float(values[target_mask].mean()) if target_mask.any() else 0.0
        if best is None or score > best[0]:
            best = (score, row, target_boxes, values)
    # A high-activation example is illustrative only; all quantitative estimates use the full test set.
    return best[1], best[2], best[3], best[0]


def overlay(image, values, face_box, part_boxes):
    width, height = image.size
    face_box = [coordinate * 512 for coordinate in face_box]
    face_mask = mask_for_box(face_box, values.shape[0], values.shape[1]).numpy()
    region_values = values[face_mask]
    vmax = max(float(region_values.max()) if region_values.size else 0.0, 1e-8)
    normalized = np.clip(values / vmax, 0, 1) * face_mask
    red = np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1)
    rgba = np.uint8(np.stack((red, green, blue, normalized * 0 + 1), axis=-1) * 255)
    rgba[..., 3] = np.uint8(normalized * 180)
    heatmap = Image.fromarray(rgba, mode="RGBA").resize(image.size, Image.Resampling.NEAREST)
    result = Image.alpha_composite(image.convert("RGBA"), heatmap).convert("RGB")
    draw = ImageDraw.Draw(result)
    sx, sy = width / 512, height / 512
    x1, y1, x2, y2 = face_box
    draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline="#00e5ff", width=3)
    for item in part_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline="#ffff00", width=3)
    return result


def main():
    args = parse_args()
    device = torch.device(args.device)
    rows = read_csv(args.manifest)
    part_boxes = load_region_boxes(args.boxes)
    faces = face_boxes(args.boxes)
    model = SparseAutoencoder.load_from_disk(str(args.model_path)).float().to(device).eval()
    model.requires_grad_(False)
    activation_cache = {}

    panel_width, image_width, image_height, header_height = 1040, 480, 480, 48
    cell_width, cell_height = 520, 540
    sheet = Image.new("RGB", (panel_width, header_height + 2 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((20, 20), "Held-out examples · jet = feature activation · cyan = face box · yellow = part box",
              fill=(30, 45, 60), font=font)
    for index, (feature, region) in enumerate(EXAMPLES):
        row, target_boxes, values, part_score = choose_sample(
            rows, part_boxes, faces, args.residuals, region, feature, model, device, activation_cache
        )
        sample_id = row["sample_id"]
        with Image.open(args.images / row["image"]) as source:
            image = source.convert("RGB")
        rendered = overlay(image, values, faces[sample_id], target_boxes)
        rendered.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
        column, row_index = index % 2, index // 2
        x = column * cell_width
        y = header_height + row_index * cell_height
        title = (f"Feature {feature} · {region.title()} box · target mean {part_score:.2f} · {sample_id}"
                 f" · {row['race_prompt'].title()} {row['gender_prompt'].title()}")
        draw.text((x + 12, y + 4), title, fill=(30, 45, 60), font=font)
        sheet.paste(rendered, (x + 20, y + 30))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)
    print(args.output)


if __name__ == "__main__":
    main()
