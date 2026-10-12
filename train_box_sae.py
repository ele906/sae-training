"""Fine-tune a Top-K SAE with a C2C box prediction head."""
import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLOCK = "unet.down_blocks.2.attentions.1"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def target_box(row):
    boxes = json.loads(row["boxes_xyxy_normalized"])
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def box_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(area_a+area_b-inter, 1e-9)


def average_precision(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float((precision * ranked).sum() / ranked.sum())


def make_head(features, concepts):
    import torch.nn as nn
    return nn.Conv2d(features, concepts * 5, kernel_size=1)


def encode_and_reconstruct(sae, x):
    # x is [spatial cells, residual dimension]. Preserve sparse reconstruction,
    # and materialize sparse latents only for the small detection head.
    reconstruction, info = sae(x)
    indices = info["inds"].reshape(x.shape[0], sae.k)
    values = info["vals"].reshape(x.shape[0], sae.k)
    codes = torch.zeros((x.shape[0], sae.n_dirs), dtype=x.dtype, device=x.device)
    codes.scatter_(1, indices, values)
    return reconstruction, codes


def forward_one(sae, head, residual):
    h, w, d = residual.shape
    flat = residual.reshape(h*w, d)
    reconstruction, codes = encode_and_reconstruct(sae, flat)
    predictions = head(codes.reshape(1, h, w, -1).permute(0, 3, 1, 2))
    predictions = predictions.reshape(1, 2, 5, h, w)
    return reconstruction, predictions


def losses(predictions, residual, boxes, pos_weight):
    _, concepts, _, h, w = predictions.shape
    objectness = predictions[:, :, 0]
    box_values = predictions[:, :, 1:5].sigmoid()
    object_target = torch.zeros_like(objectness)
    box_target = torch.zeros((1, concepts, 4, h, w), device=predictions.device)
    for c, box in enumerate(boxes):
        if box is None:
            continue
        cx, cy, bw, bh = box_to_cxcywh(box)
        gx = min(w-1, max(0, int(cx*w)))
        gy = min(h-1, max(0, int(cy*h)))
        object_target[0, c, gy, gx] = 1
        box_target[0, c, :, gy, gx] = torch.tensor([cx, cy, bw, bh], device=predictions.device)
    weights = 1 + (pos_weight-1) * object_target
    object_loss = (F.binary_cross_entropy_with_logits(objectness, object_target, reduction="none") * weights).mean()
    positive = object_target.bool()
    if positive.any():
        predicted = box_values.permute(0, 1, 3, 4, 2)[positive]
        expected = box_target.permute(0, 1, 3, 4, 2)[positive]
        box_loss = F.smooth_l1_loss(predicted, expected)
    else:
        box_loss = object_loss.new_zeros(())
    return object_loss, box_loss, object_target


@torch.no_grad()
def evaluate(sae, head, rows, boxes_by_id, residual_dir, device, recon_weight, object_weight, box_weight, pos_weight):
    head.eval()
    sae.eval()
    total_loss = 0.0
    records = {c: {"scores": [], "labels": [], "ious": [], "detected_iou50": []} for c in range(2)}
    for row in rows:
        residual = torch.from_numpy(np.load(residual_dir / f"{row['sample_id']}.npy", allow_pickle=False).astype(np.float32)).to(device)
        recons, pred = forward_one(sae, head, residual)
        recon = F.mse_loss(recons, residual.reshape(-1, residual.shape[-1])) / residual.var().clamp_min(1e-6)
        boxes = boxes_by_id[row["sample_id"]]
        obj_loss, box_loss, _ = losses(pred, residual, boxes, pos_weight)
        total_loss += float(recon_weight*recon + object_weight*obj_loss + box_weight*box_loss)
        h, w = residual.shape[:2]
        for c, gt in enumerate(boxes):
            scores = pred[0, c, 0].sigmoid().flatten()
            best = int(scores.argmax())
            gy, gx = divmod(best, w)
            raw = pred[0, c, 1:5, gy, gx].sigmoid().tolist()
            pred_box = cxcywh_to_xyxy(raw)
            score = float(scores[best])
            records[c]["scores"].append(score)
            records[c]["labels"].append(gt is not None)
            if gt is not None:
                overlap = iou(pred_box, gt)
                records[c]["ious"].append(overlap)
                records[c]["detected_iou50"].append(score >= 0.5 and overlap >= 0.5)
    result = {"mean_validation_loss": total_loss/max(len(rows), 1)}
    for c, name in enumerate(("kimono", "flower")):
        r = records[c]
        result[name] = {
            "images": len(r["labels"]),
            "c2c_positive_images": int(sum(r["labels"])),
            "objectness_average_precision": average_precision(r["scores"], r["labels"]),
            "mean_iou_on_c2c_positive": float(np.mean(r["ious"])) if r["ious"] else None,
            "recall_at_score_0p5_and_iou_0p5": float(np.mean(r["detected_iou50"])) if r["detected_iou50"] else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE / "c2c_box_targets.csv")
    parser.add_argument("--residuals", type=Path, default=HERE / "residuals")
    parser.add_argument("--output", type=Path, default=HERE / "runs/c2c_box_sae_001")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sae-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--positive-weight", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh --output: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    rows = read_csv(args.targets)
    pairs = [(r["sample_id"], r["concept"]) for r in rows]
    if len(pairs) != len(set(pairs)) or any(r["split"] not in ("train", "validation", "test") for r in rows):
        raise ValueError("Expected one row per image/concept pair with valid splits")
    concepts = ("kimono", "flower")
    boxes_by_id = {}
    sample_meta = {}
    for row in rows:
        if row["concept"] not in concepts:
            raise ValueError(f"Unknown target concept {row['concept']}")
        boxes_by_id.setdefault(row["sample_id"], [None, None])[concepts.index(row["concept"])] = target_box(row)
        sample_meta[row["sample_id"]] = row
    if set(boxes_by_id) != {p.stem for p in args.residuals.glob("*.npy")}:
        missing = sorted(set(boxes_by_id) - {p.stem for p in args.residuals.glob("*.npy")})
        if missing:
            raise ValueError(f"Missing residual files, first IDs: {missing[:5]}")
    train = [r for r in rows if r["concept"] == "kimono" and r["split"] == "train"]
    val = [r for r in rows if r["concept"] == "kimono" and r["split"] == "validation"]
    test = [r for r in rows if r["concept"] == "kimono" and r["split"] == "test"]
    if not train or not val or not test:
        raise ValueError("Need train, validation, and test samples")

    import sys
    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from SAE import SparseAutoencoder
    from SAE.sae import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
    gen_settings = json.loads((ROOT / "backprop/generation_settings.json").read_text())
    checkpoint = Path(gen_settings["sae_checkpoint"])
    sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device)
    sae.train()
    example_path = next(args.residuals.glob("*.npy"))
    example = np.load(example_path, mmap_mode="r", allow_pickle=False)
    spatial_shape = list(example.shape[:2])
    if example.shape[-1] != sae.d_model:
        raise ValueError(f"Residual dimension {example.shape[-1]} != SAE input {sae.d_model}")
    del example
    head = make_head(sae.n_dirs, len(concepts)).to(device)
    optimizer = torch.optim.Adam([
        {"params": sae.parameters(), "lr": args.sae_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ])
    train_ids = [r["sample_id"] for r in train]
    val_ids = [r["sample_id"] for r in val]
    test_ids = [r["sample_id"] for r in test]
    best_loss = float("inf")
    best_sae, best_head = None, None
    history = []
    args.output.mkdir(parents=True)
    for epoch in range(args.epochs):
        sae.train()
        head.train()
        order = list(train_ids)
        random.shuffle(order)
        epoch_loss = 0.0
        for step, sid in enumerate(order, 1):
            residual = torch.from_numpy(np.load(args.residuals / f"{sid}.npy", allow_pickle=False).astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, pred = forward_one(sae, head, residual)
            recon_loss = F.mse_loss(reconstruction, residual.reshape(-1, residual.shape[-1])) / residual.var().clamp_min(1e-6)
            obj_loss, box_loss, _ = losses(pred, residual, boxes_by_id[sid], args.positive_weight)
            loss = args.reconstruction_weight*recon_loss + args.objectness_weight*obj_loss + args.box_weight*box_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, sample {sid}")
            loss.backward()
            unit_norm_decoder_grad_adjustment_(sae)
            optimizer.step()
            with torch.no_grad():
                unit_norm_decoder_(sae)
            epoch_loss += float(loss.detach())
            if step % 100 == 0:
                print(f"epoch={epoch+1} step={step}/{len(order)} loss={float(loss.detach()):.5f}", flush=True)
            del residual, reconstruction, pred, loss
        validation = evaluate(sae, head, val, boxes_by_id, args.residuals, device,
                              args.reconstruction_weight, args.objectness_weight,
                              args.box_weight, args.positive_weight)
        record = {"epoch": epoch+1, "train_loss": epoch_loss/len(order), **validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["mean_validation_loss"] < best_loss:
            best_loss = validation["mean_validation_loss"]
            best_sae, best_head = copy.deepcopy(sae).cpu(), copy.deepcopy(head).cpu()
    best_sae.save_to_disk(str(args.output / "sae"))
    torch.save(best_head.state_dict(), args.output / "box_head.pth")
    final_sae = best_sae.to(device).eval()
    final_head = best_head.to(device).eval()
    test_metrics = evaluate(final_sae, final_head, test, boxes_by_id, args.residuals, device,
                            args.reconstruction_weight, args.objectness_weight,
                            args.box_weight, args.positive_weight)
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    (args.output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, allow_nan=False) + "\n")
    settings = dict(targets=str(args.targets.resolve()), residuals=str(args.residuals.resolve()),
                    epochs=args.epochs, sae_lr=args.sae_lr, head_lr=args.head_lr,
                    reconstruction_weight=args.reconstruction_weight, objectness_weight=args.objectness_weight,
                    box_weight=args.box_weight, positive_weight=args.positive_weight,
                    seed=args.seed, block=BLOCK, spatial_shape=spatial_shape,
                    train_images=len(train_ids), validation_images=len(val_ids), test_images=len(test_ids),
                    best_validation_loss=best_loss,
                    interpretation="Held-out box agreement with C2C teacher labels; not independent box accuracy")
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    report = ["# C2C-supervised SAE box prediction", "", "Held-out test performance against C2C boxes.", "",
              "```json", json.dumps(test_metrics, indent=2, allow_nan=False), "```", "",
              "This measures imitation of C2C box labels, not independent localization accuracy."]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"test_metrics": test_metrics, "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
