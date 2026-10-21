"""Select facial-region SAE features on validation data and check them on test data."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sdxl-unbox"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from SAE import SparseAutoencoder
from fine_tune_face_sae import DEFAULT_BOXES, DEFAULT_MANIFEST, DEFAULT_RESIDUALS, face_boxes, read_csv
from render_face_region_heatmaps import load_region_boxes

REGIONS = ("eyes", "eyebrows", "nose", "ears")
MAX_BOX_AREA = 0.12
MIN_REGION_CELLS = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Eyes-only SAE run with its selected SAE")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    parser.add_argument("--output", type=Path, help="Default: RUN/part_feature_validation")
    parser.add_argument("--model-path", type=Path,
                        help="SAE checkpoint to evaluate; defaults to RUN/sae")
    parser.add_argument(
        "--feature-ids", type=str,
        help="Comma-separated fixed SAE feature IDs to test for every region; keeps this list fixed on test data.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--resume", action="store_true", help="Overwrite an existing results folder")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.model_path is not None:
        args.model_path = args.model_path.resolve()
    args.output = (args.output or args.run / "part_feature_validation").resolve()
    if args.output.exists() and not args.resume:
        raise ValueError(f"Output exists; choose a new --output or use --resume: {args.output}")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    return args


def region_mask(boxes, height, width):
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) * (512 / width)
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) * (512 / height)
    mask = torch.zeros((height, width), dtype=torch.bool)
    for item in boxes:
        x1, y1, x2, y2 = item["xyxy"]
        area_fraction = (x2 - x1) * (y2 - y1) / (512 * 512)
        if area_fraction > MAX_BOX_AREA:
            continue
        mask |= ((xs[None, :] >= x1) & (xs[None, :] <= x2)
                 & (ys[:, None] >= y1) & (ys[:, None] <= y2))
    return mask


def category_name(row):
    return f"{row['race_prompt'].replace(' ', '_')}_{row['gender_prompt']}"


def selected_rows(rows, region_boxes, face_box_map, residual_dir):
    selected = {region: {"validation": [], "test": []} for region in REGIONS}
    for row in rows:
        split = row["split"]
        sample_id = row["sample_id"]
        if split not in ("validation", "test") or sample_id not in face_box_map:
            continue
        residual_path = residual_dir / f"{sample_id}.npy"
        if not residual_path.is_file():
            continue
        shape = np.load(residual_path, mmap_mode="r", allow_pickle=False).shape
        face = torch.zeros(shape[:2], dtype=torch.bool)
        fx1, fy1, fx2, fy2 = face_box_map[sample_id]
        xs = (torch.arange(shape[1], dtype=torch.float32) + 0.5) / shape[1]
        ys = (torch.arange(shape[0], dtype=torch.float32) + 0.5) / shape[0]
        face = ((xs[None, :] >= fx1) & (xs[None, :] <= fx2)
                & (ys[:, None] >= fy1) & (ys[:, None] <= fy2))
        for region in REGIONS:
            boxes = region_boxes.get(sample_id, {}).get(region, [])
            reference = torch.ones_like(face) if region == "ears" else face
            part = region_mask(boxes, *shape[:2]) & reference
            rest = reference & ~part
            if int(part.sum()) < MIN_REGION_CELLS or int(rest.sum()) < MIN_REGION_CELLS:
                continue
            selected[region][split].append({**row, "category": category_name(row)})
    return selected


@torch.no_grad()
def collect_activations(model, sample_rows, region_boxes, face_box_map, residual_dir, device):
    vectors = {}
    model.eval()
    for row in sample_rows:
        sample_id = row["sample_id"]
        array = np.load(residual_dir / f"{sample_id}.npy", allow_pickle=False)
        residual = torch.from_numpy(array.astype(np.float32)).to(device)
        height, width = residual.shape[:2]
        codes = model.encode(residual)

        fx1, fy1, fx2, fy2 = face_box_map[sample_id]
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        face = ((xs[None, :] >= fx1) & (xs[None, :] <= fx2)
                & (ys[:, None] >= fy1) & (ys[:, None] <= fy2))
        for region in REGIONS:
            boxes = region_boxes.get(sample_id, {}).get(region, [])
            reference = torch.ones_like(face) if region == "ears" else face
            part = region_mask(boxes, height, width).to(device) & reference
            rest = reference & ~part
            if int(part.sum()) < MIN_REGION_CELLS or int(rest.sum()) < MIN_REGION_CELLS:
                continue
            vectors[region, row["split"], sample_id] = {
                "part_mean": codes[part].mean(dim=0).float().cpu().numpy(),
                "rest_mean": codes[rest].mean(dim=0).float().cpu().numpy(),
                "category": row["category"],
            }
        del residual, codes
    return vectors


def category_contrasts(entries, feature_ids=None):
    by_category = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry["part_mean"] - entry["rest_mean"])
    categories = {
        category: np.mean(values, axis=0)
        for category, values in by_category.items()
    }
    if feature_ids is not None:
        categories = {category: values[feature_ids] for category, values in categories.items()}
    return categories


def load_candidates(run_dir, validation_vectors):
    with (run_dir / "eye_localized_features.csv").open(newline="") as source:
        eye_rows = list(csv.DictReader(source))
    eye_features = [int(row["feature"]) for row in eye_rows[:3]]
    candidates = {"eyes": eye_features}
    validation_means = {}
    for region in REGIONS:
        entries = [value for (part, split, _sample), value in validation_vectors.items()
                   if part == region and split == "validation"]
        macro = category_contrasts(entries)
        score = np.mean(list(macro.values()), axis=0)
        ranked = np.argsort(score)[::-1][:3].astype(int).tolist()
        if region != "eyes":
            candidates[region] = ranked
        validation_means[region] = score
    return candidates, validation_means


def bootstrap_macro_ci(entries, feature_ids, iterations, seed):
    grouped = defaultdict(list)
    for entry in entries:
        difference = entry["part_mean"][feature_ids] - entry["rest_mean"][feature_ids]
        grouped[entry["category"]].append(difference)
    arrays = {category: np.stack(values) for category, values in grouped.items()}
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(feature_ids)), dtype=np.float64)
    for iteration in range(iterations):
        category_means = []
        for values in arrays.values():
            indices = rng.integers(0, len(values), size=len(values))
            category_means.append(values[indices].mean(axis=0))
        draws[iteration] = np.mean(category_means, axis=0)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def validate_candidates(candidates, validation_vectors, test_vectors, bootstrap_samples, seed):
    summary_rows = []
    details = {}
    for region in REGIONS:
        feature_ids = candidates[region]
        validation_entries = [value for (part, split, _sample), value in validation_vectors.items()
                              if part == region and split == "validation"]
        test_entries = [value for (part, split, _sample), value in test_vectors.items()
                        if part == region and split == "test"]
        validation_macro = category_contrasts(validation_entries, feature_ids)
        test_macro = category_contrasts(test_entries, feature_ids)
        low, high = bootstrap_macro_ci(test_entries, feature_ids, bootstrap_samples,
                                       seed + REGIONS.index(region))
        categories = sorted(test_macro)
        details[region] = {
            "candidate_features": feature_ids,
            "validation_categories": len(validation_macro),
            "test_categories": len(test_macro),
            "test_images": len(test_entries),
            "bootstrap_method": "stratified image bootstrap within race/gender categories; macro-average category contrast",
            "bootstrap_samples": bootstrap_samples,
        }
        for index, feature in enumerate(feature_ids):
            test_image_diffs = np.array([
                entry["part_mean"][feature] - entry["rest_mean"][feature]
                for entry in test_entries
            ])
            image_pooled = float(test_image_diffs.mean())
            macro_diff = float(np.mean([test_macro[category][index] for category in categories]))
            part_mean = float(np.mean([entry["part_mean"][feature] for entry in test_entries]))
            rest_mean = float(np.mean([entry["rest_mean"][feature] for entry in test_entries]))
            summary_rows.append({
                "region": region,
                "selection_split": "validation",
                "feature": feature,
                "validation_macro_contrast": float(np.mean([
                    validation_macro[category][index] for category in validation_macro
                ])),
                "test_macro_contrast": macro_diff,
                "test_bootstrap_ci_low": float(low[index]),
                "test_bootstrap_ci_high": float(high[index]),
                "test_pooled_image_contrast": image_pooled,
                "test_mean_activation_in_region": part_mean,
                "reference_region": "rest_of_image" if region == "ears" else "rest_of_face",
                "test_mean_activation_reference": rest_mean,
                "test_images": len(test_entries),
                "test_categories": len(categories),
                "test_images_with_positive_contrast_fraction": float((test_image_diffs > 0).mean()),
            })
    return summary_rows, details


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    rows = read_csv(args.manifest)
    region_boxes = load_region_boxes(args.boxes)
    face_box_map = face_boxes(args.boxes)
    selected = selected_rows(rows, region_boxes, face_box_map, args.residuals)
    if any(not selected[region][split] for region in REGIONS for split in ("validation", "test")):
        raise ValueError("Each region needs usable validation and test examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path or (args.run / "sae")
    model = SparseAutoencoder.load_from_disk(str(model_path)).float().to(device).eval()
    validation_rows = list({
        row["sample_id"]: row
        for region in REGIONS for row in selected[region]["validation"]
    }.values())
    test_rows = list({
        row["sample_id"]: row
        for region in REGIONS for row in selected[region]["test"]
    }.values())
    validation_vectors = collect_activations(model, validation_rows, region_boxes,
                                             face_box_map, args.residuals, device)
    test_vectors = collect_activations(model, test_rows, region_boxes,
                                       face_box_map, args.residuals, device)
    if args.feature_ids:
        feature_ids = [int(value.strip()) for value in args.feature_ids.split(",") if value.strip()]
        if not feature_ids or len(set(feature_ids)) != len(feature_ids):
            raise ValueError("--feature-ids must contain unique integer IDs")
        candidates = {region: feature_ids for region in REGIONS}
        candidate_selection = (
            "fixed feature list supplied on command line; features ranked by validation eye contrast; "
            "no test data used for feature selection"
        )
    else:
        candidates, _ = load_candidates(args.run, validation_vectors)
        candidate_selection = (
            "eye candidates from eye fine-tuning validation ranking; eyebrow, nose, and ear candidates "
            "ranked by validation category activation contrast"
        )
    summary_rows, details = validate_candidates(
        candidates, validation_vectors, test_vectors, args.bootstrap_samples, args.seed
    )

    summary_path = args.output / "heldout_feature_validation.csv"
    with summary_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "model": str(model_path),
        "candidate_selection": candidate_selection,
        "test_evaluation": "candidate features evaluated only on test split; eyes and eyebrows contrasted with other face cells, ears contrasted with other image cells",
        "box_filter": {"maximum_area_fraction": MAX_BOX_AREA,
                       "minimum_region_cells": MIN_REGION_CELLS,
                       "earrings_excluded": True},
        "regions": details,
        "interpretation": "Positive held-out contrast supports spatial association with the annotated box, not a semantic feature label. Boxes are automatic and unreviewed.",
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved held-out feature validation to {summary_path}")
    for row in summary_rows:
        print(f"{row['region']} feature={row['feature']} test_macro_contrast="
              f"{row['test_macro_contrast']:.4f} 95%CI="
              f"[{row['test_bootstrap_ci_low']:.4f}, {row['test_bootstrap_ci_high']:.4f}] "
              f"images={row['test_images']} categories={row['test_categories']}")


if __name__ == "__main__":
    main()
