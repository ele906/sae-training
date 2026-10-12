# Concept-supervised SAE experiment

## Goal

Test whether C2C-labeled *kimono* and *flower* concepts help an SAE represent those concepts more directly than the current reconstruction-trained SAE, while preserving useful reconstruction quality. For this exploratory run, treat C2C outputs as correct working labels. The resulting claim is conditional on that assumption and does not validate the detector.

## What exists now

- `backprop/sae_features.npz` contains pooled activations for 1,200 SDXL portraits from the existing 5,120-feature Top-K SAE (k=10), at `unet.down_blocks.2.attentions.1`.
- The current SAE remains frozen in all completed concept work. Its feature 5046 for kimono and 2230 for flower came from Florence-2 labels, not human review.
- `c2c_targets.csv` is the normalized C2C target set for kimono and flower, using the same phrase patterns as `concept/compare_sae.py`. It includes split metadata; reserve test rows for final evaluation only.
- C2C targets may contain misses or hallucinations. The existing kimono validation comparison has no matched within-prompt validation group; flower has only one. The ablation pilot used only four same-prompt images per concept (kimono: 2 positive/2 negative) and did not show reliable object-specific removal.
- `annotations_template.csv` remains an optional human-audit sheet, not a prerequisite for this assumed-label experiment.

## Experiment stages

1. **Use C2C targets.** `c2c_targets.csv` maps the existing detector phrases to binary labels. Keep the exact phrase rules and source CSV hash in run settings.
2. **Check feasibility before training.** Report class counts by split and race/gender prompt cell. Require train and validation positives and negatives; require both classes within at least two prompt cells for a first controlled result. If labels are too sparse or correlated with prompt group, mark that limitation and avoid a concept-specific claim.
3. **Run a frozen-SAE probe baseline first.** Train the existing linear probe (`backprop/train_probe.py`) on C2C targets and pooled SAE activations for each label. This measures whether current SAE codes already contain linearly accessible concept information and yields candidate features. It does not train an SAE and its candidate IDs are associations, not causal controls.
4. **Collect residual activations.** Current files save pooled SAE codes, not the block residuals needed to retrain an autoencoder. Replay the existing images through SDXL and save the spatial residual vectors at `unet.down_blocks.2.attentions.1`, with sample IDs and split metadata. Do not use test samples to train.
5. **Train the concept-guided SAE variant.** Initialize from the pretrained SAE and use those residuals. Optimize reconstruction loss plus Top-K sparsity and an auxiliary concept prediction loss on pooled SAE codes. Keep reconstruction active for every spatial vector; apply concept loss only to labeled train images. Compare against the frozen baseline at matched width, k, steps, and data. Tune auxiliary-loss weight on validation only.
6. **Evaluate without claiming object control.** Report reconstruction MSE / explained variance, sparsity and dead-feature rate, probe AUROC and average precision by concept, within-prompt-cell metrics, and performance on held-out test seeds. Inspect top activating images and spatial maps. Only then test selected features with same-prompt controls and matched random features.

## Guardrails

- Split by the existing sample split; never use test labels for model choice, feature ranking, or auxiliary-loss tuning.
- This is explicitly a detector-supervised exploratory experiment: preserve C2C's raw output and normalized target separately; do not describe detector predictions as human ground truth.
- Report pooled and within-prompt-group performance. A global score can learn the race/gender prompt cell instead of the object.
- Compare the new SAE to the frozen checkpoint and a probe on frozen SAE activations. A better probe score alone does not establish better feature disentanglement or causal image editing.
- Keep checkpoint, settings, label version/hash, and outputs in a unique run directory under this folder. Do not overwrite prior runs or copy model weights/datasets into Git.

## Current readiness

**Targets are ready; residual inputs are not.** The C2C labels are accepted as working targets for this run. Existing feature files contain pooled SAE activations only, so the concept-guided SAE cannot be trained from them as-is. Next prepare a replay job that saves per-image residuals for train/validation/test splits, then train the auxiliary-loss variant. The sparse kimono/flower validation groups still constrain what we can conclude.

The existing non-training intervention pilot is documented in [tuning/README.md](../tuning/README.md), and the broader detector/SAE evidence is in [concept/README.md](../concept/README.md).
