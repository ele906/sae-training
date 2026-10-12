# C2C-supervised SAE box prediction

## Goal

Train an SAE with a small box-prediction head so it predicts *kimono* and *flower* boxes that match Concept2Concept (C2C). The model receives spatial SDXL residuals and produces spatial SAE codes; a detector head reads those codes and predicts boxes. For this exploratory run, treat C2C boxes as correct working targets. This tests imitation of C2C's annotations, not independent object detection accuracy.

## What exists now

- `backprop/sae_features.npz` contains pooled activations for 1,200 SDXL portraits from the existing 5,120-feature Top-K SAE (k=10), at `unet.down_blocks.2.attentions.1`.
- The current SAE remains frozen in all completed concept work. Its feature 5046 for kimono and 2230 for flower came from Florence-2 labels, not human review.
- `c2c_targets.csv` is the normalized C2C target set for kimono and flower, using the same phrase patterns as `concept/compare_sae.py`. It includes split metadata; reserve test rows for final evaluation only.
- `prepare_box_targets.py` joins C2C phrase-grounding and object boxes to the sample manifest. Run it to create `c2c_box_targets.csv`; boxes are normalized `x1,y1,x2,y2` in `[0,1]`, and every sample has a row per concept, including empty-box rows.
- C2C targets may contain misses or hallucinations. The existing kimono validation comparison has no matched within-prompt validation group; flower has only one. The ablation pilot used only four same-prompt images per concept (kimono: 2 positive/2 negative) and did not show reliable object-specific removal.
- `annotations_template.csv` remains an optional human-audit sheet, not a prerequisite for this assumed-label experiment.

## Experiment stages

1. **Use C2C targets.** `c2c_targets.csv` maps the existing detector phrases to binary labels. Keep the exact phrase rules and source CSV hash in run settings.
2. **Check feasibility before training.** Report class counts by split and race/gender prompt cell. Require train and validation positives and negatives; require both classes within at least two prompt cells for a first controlled result. If labels are too sparse or correlated with prompt group, mark that limitation and avoid a concept-specific claim.
3. **Run a frozen-SAE probe baseline first.** Train the existing linear probe (`backprop/train_probe.py`) on C2C targets and pooled SAE activations for each label. This measures whether current SAE codes already contain linearly accessible concept information and yields candidate features. It does not train an SAE and its candidate IDs are associations, not causal controls.
4. **Collect residual activations.** `capture_residuals.py` replays the original prompt/seed and requires exact RGB equality. It saves one float16 spatial residual array per sample. Submit `capture_residuals.slurm` on the lab GPU cluster.
5. **Train the SAE and box head jointly.** `train_box_sae.py` initializes the SAE from the pretrained checkpoint and adds per-concept objectness and box regression over spatial SAE codes. It combines normalized reconstruction loss, objectness loss, and box regression loss. The SAE is fine-tuned on train images; validation loss selects the checkpoint; test images remain held out. The target box is the union of C2C's matching boxes for each image/concept. Submit `train_box_sae.slurm` after residual capture completes.
6. **Evaluate imitation and representation separately.** The run writes held-out box AP, IoU, recall, per-image predictions, and validation history under its run directory. These metrics measure agreement with C2C; visual review or independent annotations are needed to claim the boxes are correct in the images. Run the frozen-SAE baseline with `train_box_baseline.slurm` to see whether SAE fine-tuning improves over the original codes.

## Running the first experiment

From the workspace root:

```bash
python sae-training/prepare_box_targets.py
mkdir -p sae-training/logs
sbatch sae-training/capture_residuals.slurm
# After residual capture completes successfully:
sbatch sae-training/train_box_sae.slurm
# Matched baseline (can run independently once residual capture completes):
sbatch sae-training/train_box_baseline.slurm
```

Residual capture job 4088384 completed for all 1,200 images with exact RGB replay. The five-epoch fine-tuned run and matched frozen-SAE baseline completed; see [first-run results](RESULTS.md). Kimono IoU did not improve, and flower box localization remains poor. Training checkpoints, residual arrays, and run outputs are excluded from Git.

## Visual results

These held-out examples show the original image, C2C target (orange), frozen-SAE box (blue), and fine-tuned-SAE box (green). The selected sheet includes low, middle, and high fine-tuned IoU positives plus the highest-score C2C-negative image. Full positive test pages are alongside them.

- [Kimono examples](visualizations/kimono_examples.jpg) · [all kimono positives](visualizations/)
- [Flower examples](visualizations/flower_examples.jpg) · [all flower positives](visualizations/)

Recreate the sheets with `python sae-training/render_box_comparisons.py` after the ignored run outputs have been generated locally.

## Guardrails

- Split by the existing sample split; never use test labels for model choice, feature ranking, or auxiliary-loss tuning.
- This is explicitly a detector-imitation experiment: preserve C2C's raw output and normalized target separately; do not describe C2C matches as independent ground truth.
- Report pooled and within-prompt-group performance. A global score can learn the race/gender prompt cell instead of the object.
- Compare the new SAE to the frozen checkpoint and a probe on frozen SAE activations. A better probe score alone does not establish better feature disentanglement or causal image editing.
- Keep checkpoint, settings, label version/hash, and outputs in a unique run directory under this folder. Do not overwrite prior runs or copy model weights/datasets into Git.

## Current readiness

**Box targets are preparable; residual inputs and trainer are not.** Run `python sae-training/prepare_box_targets.py` from the workspace root to build the normalized box table from `concept/full_run/records`. Then prepare a replay job that saves per-image spatial residuals for train/validation/test splits, followed by the joint SAE and box-head trainer. Existing pooled activation files cannot be used for box prediction because they discard spatial positions.

The existing non-training intervention pilot is documented in [tuning/README.md](../tuning/README.md), and the broader detector/SAE evidence is in [concept/README.md](../concept/README.md).
