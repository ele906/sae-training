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
4. **Collect residual activations.** Current files save pooled SAE codes, not the spatial residuals needed to train an SAE or box head. Replay the existing images through SDXL and save the residual grid at `unet.down_blocks.2.attentions.1`, with sample IDs and split metadata. Do not use test samples to train.
5. **Train the SAE and box head jointly.** Initialize the SAE from the pretrained checkpoint. Add an objectness score and box-regression head over the spatial SAE codes. Optimize reconstruction loss plus the existing Top-K sparsity behavior, plus objectness and box losses against the C2C boxes. Only train the head on train images; tune loss weights on validation. Use all valid matching C2C boxes per image/concept, including multiple boxes.
6. **Evaluate imitation and representation separately.** On validation and held-out test images, report box AP/IoU against C2C, reconstruction MSE / explained variance, sparsity and dead-feature rate, and metrics by prompt group. Compare with the frozen SAE plus the same detection head. Inspect predicted boxes and spatial maps. This measures agreement with C2C; visual review or independent annotations are needed to claim the boxes are correct in the images.

## Guardrails

- Split by the existing sample split; never use test labels for model choice, feature ranking, or auxiliary-loss tuning.
- This is explicitly a detector-imitation experiment: preserve C2C's raw output and normalized target separately; do not describe C2C matches as independent ground truth.
- Report pooled and within-prompt-group performance. A global score can learn the race/gender prompt cell instead of the object.
- Compare the new SAE to the frozen checkpoint and a probe on frozen SAE activations. A better probe score alone does not establish better feature disentanglement or causal image editing.
- Keep checkpoint, settings, label version/hash, and outputs in a unique run directory under this folder. Do not overwrite prior runs or copy model weights/datasets into Git.

## Current readiness

**Box targets are preparable; residual inputs and trainer are not.** Run `python sae-training/prepare_box_targets.py` from the workspace root to build the normalized box table from `concept/full_run/records`. Then prepare a replay job that saves per-image spatial residuals for train/validation/test splits, followed by the joint SAE and box-head trainer. Existing pooled activation files cannot be used for box prediction because they discard spatial positions.

The existing non-training intervention pilot is documented in [tuning/README.md](../tuning/README.md), and the broader detector/SAE evidence is in [concept/README.md](../concept/README.md).
