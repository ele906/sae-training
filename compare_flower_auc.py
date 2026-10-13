"""Compare flower feature separability in frozen and box-fine-tuned SAEs."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    # Pairwise form handles ties explicitly and avoids an extra dependency.
    return float(((pos[:, None] > neg[None, :]).sum() +
                  0.5 * (pos[:, None] == neg[None, :]).sum()) /
                 (len(pos) * len(neg)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE / "c2c_box_targets.csv")
    parser.add_argument("--residuals", type=Path, default=HERE / "residuals")
    parser.add_argument("--output", type=Path, default=HERE / "runs/flower_auc_frozen_vs_finetuned_001")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Output exists; choose a fresh directory: {args.output}")

    sys.path.insert(0, str(ROOT / "sdxl-unbox"))
    from SAE import SparseAutoencoder

    device = torch.device("cuda")
    torch.set_num_threads(4)
    rows = [r for r in read_csv(args.targets)
            if r["concept"] == "flower" and r["race_prompt"] == "east asian"
            and r["gender_prompt"] == "woman" and r["split"] in ("train", "test")]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    labels_train = np.array([int(r["detector_present"]) for r in train])
    labels_test = np.array([int(r["detector_present"]) for r in test])
    if not labels_train.any() or labels_train.all() or not labels_test.any() or labels_test.all():
        raise ValueError("Need positive and negative C2C labels in train and held-out test")

    models = {
        "frozen": HERE / "runs/frozen_sae_box_baseline_002/sae",
        "finetuned": HERE / "runs/c2c_box_sae_003/sae",
    }
    all_scores = {}
    for name, checkpoint in models.items():
        sae = SparseAutoencoder.load_from_disk(str(checkpoint)).float().to(device).eval()
        split_scores = {}
        with torch.inference_mode():
            for split, subset in (("train", train), ("test", test)):
                scores = []
                for i, row in enumerate(subset, 1):
                    residual = torch.from_numpy(np.load(
                        args.residuals / f"{row['sample_id']}.npy", allow_pickle=False
                    ).astype(np.float32)).to(device)
                    h, w, d = residual.shape
                    z = sae.encode(residual.reshape(-1, d)).reshape(h, w, -1)
                    # Image-level max pooling asks whether any spatial cell activates a feature.
                    scores.append(z.amax(dim=(0, 1)).cpu().numpy())
                    if i % 20 == 0:
                        print(f"{name} {split}: {i}/{len(subset)}", flush=True)
                    del residual, z
                split_scores[split] = np.stack(scores)
        all_scores[name] = split_scores
        del sae
        torch.cuda.empty_cache()

    args.output.mkdir(parents=True)
    records = []
    summary = {
        "cohort": "East Asian woman prompts; C2C flower labels",
        "train_images": len(train),
        "train_positive": int(labels_train.sum()),
        "train_negative": int(len(labels_train) - labels_train.sum()),
        "test_images": len(test),
        "test_positive": int(labels_test.sum()),
        "test_negative": int(len(labels_test) - labels_test.sum()),
        "feature_score": "maximum SAE activation across 16x16 spatial cells per image",
        "selection": "rank individual features by train AUC; evaluate selected IDs on held-out test",
        "models": {},
    }
    for name, values in all_scores.items():
        train_auc = np.array([auc(labels_train, values["train"][:, j])
                              for j in range(values["train"].shape[1])])
        order = np.argsort(-train_auc, kind="stable")
        selected = [int(j) for j in order if train_auc[j] > 0.5][:args.top_k]
        if not selected:
            selected = [int(order[0])]
        model_result = {"selected_features": []}
        for rank, feature in enumerate(selected, 1):
            test_auc = auc(labels_test, values["test"][:, feature])
            row = dict(model=name, rank=rank, feature=feature,
                       train_auc=float(train_auc[feature]), test_auc=test_auc,
                       train_positive_active=int((values["train"][labels_train == 1, feature] > 0).sum()),
                       test_positive_active=int((values["test"][labels_test == 1, feature] > 0).sum()))
            records.append(row)
            model_result["selected_features"].append(row)
        model_result["best_test_auc_among_train_selected"] = max(
            (r["test_auc"] for r in model_result["selected_features"]), default=float("nan"))
        model_result["median_test_auc_among_train_selected"] = float(np.median(
            [r["test_auc"] for r in model_result["selected_features"]]))
        summary["models"][name] = model_result

    with (args.output / "feature_auc.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    # The already-trained box heads give a complementary comparison using the same test prompts.
    head_scores = {}
    for name, path in {
        "frozen": HERE / "runs/frozen_sae_box_baseline_002/test_predictions.csv",
        "finetuned": HERE / "runs/c2c_box_sae_003/test_predictions.csv",
    }.items():
        pred = [r for r in read_csv(path) if r["concept"] == "flower"
                and r["sample_id"] in {r["sample_id"] for r in test}]
        pred_map = {r["sample_id"]: float(r["score"]) for r in pred}
        labels = np.array([int(r["detector_present"]) for r in test])
        scores = np.array([pred_map[r["sample_id"]] for r in test])
        head_scores[name] = {"test_auc": auc(labels, scores),
                             "test_average_precision": float("nan")}

    # AP for the exact same matched cohort.
    for name, path in {
        "frozen": HERE / "runs/frozen_sae_box_baseline_002/test_predictions.csv",
        "finetuned": HERE / "runs/c2c_box_sae_003/test_predictions.csv",
    }.items():
        pred = [r for r in read_csv(path) if r["concept"] == "flower"
                and r["sample_id"] in {r["sample_id"] for r in test}]
        pred_map = {r["sample_id"]: float(r["score"]) for r in pred}
        order = np.argsort(-np.array([pred_map[r["sample_id"]] for r in test]), kind="stable")
        ranked = labels[order].astype(bool)
        precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
        head_scores[name]["test_average_precision"] = float((precision * ranked).sum() / ranked.sum())
    summary["box_head_test_metrics_same_cohort"] = head_scores
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
