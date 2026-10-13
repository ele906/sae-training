# First C2C box imitation run

Run date: 2026-10-12. The SDXL replay produced spatial residuals for all 1,200 images and verified exact RGB equality against each saved source image. Residual shape was `16 x 16 x 1280`. The train/validation/test split remained 720/240/240 images.

We trained the same 1x1 spatial box head in two settings for five epochs: the pretrained Top-K SAE frozen, and the SAE fine-tuned jointly with the head. Validation loss selected the fine-tuned checkpoint. Evaluation below uses the held-out 240 test images. Each target is the union of all matching C2C grounding/object boxes for one concept in an image. `mIoU` is averaged only over C2C-positive images; recall requires both confidence >= 0.5 and IoU >= 0.5.

| Concept | SAE | C2C positives / 240 | Objectness AP | Mean IoU on positives | Recall at score/IoU 0.5 |
|---|---|---:|---:|---:|---:|
| Kimono | Frozen | 21 | 0.932 | 0.575 | 0.524 |
| Kimono | Fine-tuned | 21 | 0.921 | 0.550 | 0.476 |
| Flower | Frozen | 13 | 0.799 | 0.035 | 0.000 |
| Flower | Fine-tuned | 13 | 0.840 | 0.087 | 0.000 |

The fine-tuned SAE did not improve kimono box quality. Flower objectness ranking improved slightly, but its boxes remain poorly localized: mean IoU is 0.087 and none reaches IoU 0.5. The frozen baseline performs similarly poorly on flower. Both systems rank presence above chance, but this does not mean they predict useful boxes.

These numbers measure agreement with C2C's unreviewed boxes. The flower test set has only 13 positives, and the SAE feature grid is coarse at 16 x 16. The model predicts one union box per image/concept, so it cannot reproduce multiple separate C2C boxes. This is a first pipeline result, not evidence of independent object detection or an improvement over C2C.

Artifacts are kept locally under the ignored `runs/` and `residuals/` folders: fine-tuned checkpoint `runs/c2c_box_sae_003`, frozen baseline `runs/frozen_sae_box_baseline_002`, and per-image predictions in each run's `test_predictions.csv`.

## Flower feature intervention test

Run date: 2026-10-12. We ranked SAE features by their contribution to the trained flower objectness head on the East Asian woman training subset (33 C2C-positive, 27 negative), then attenuated the top three features (940, 2810, 1855) during one-step SDXL generation. Evaluation used 20 held-out matched prompts: 12 C2C-positive and 8 C2C-negative. Every source image replayed with exact RGB equality. Strength 1 was an exact no-op. We compared strength 0 (zero the selected features), strength 0.5, and a zeroed activity-matched random feature (3459).

The target-feature ablation changed pixels substantially (mean RGB MAE 11.01/255 at strength 0, 6.21/255 at 0.5); the random-feature control averaged 1.52/255. Changes were similar for C2C-positive and negative examples, which suggests these features are not specific to flower presence. In the inspected C2C-positive examples, the flower hair ornament remained visible after ablation. The intervention altered other appearance details, including clothing and facial appearance. So this test did not remove the flower, despite changing the image.

### Example images

Each strip shows source, exact replay, selected flower features zeroed, features attenuated by half, the no-op check, and the random-feature control.

**C2C-positive example (seed 007):**

![Positive flower example, seed 007](visualizations/flower_intervention_001/positive_seed007.jpg)

**C2C-positive example (seed 009):**

![Positive flower example, seed 009](visualizations/flower_intervention_001/positive_seed009.jpg)

**C2C-negative example (seed 086):**

![Negative flower example, seed 086](visualizations/flower_intervention_001/negative_seed086.jpg)

This is a small exploratory causal test using a learned detector head to select SAE features. C2C labels are assumed correct, and pixel MAE measures overall image change rather than flower removal. Per-image pixel metrics are in [`visualizations/flower_intervention_001/metrics.csv`](visualizations/flower_intervention_001/metrics.csv).

## Did box fine-tuning improve flower information?

All flower and earring feature results reported so far use a single SAE hook at `unet.down_blocks.2.attentions.1`; they predate the requested multi-layer extension. We compared the frozen SAE and the SAE fine-tuned with the C2C box objective on the same East Asian woman prompt group. Individual SAE features were scored by their maximum activation over the 16 x 16 grid. We ranked features on the 60 training images (33 C2C-positive, 27 negative), then measured the top 10 features on 20 held-out images (12 positive, 8 negative). The median held-out AUC among these train-selected features was 0.578 for the frozen SAE and 0.531 for the fine-tuned SAE. Fine-tuning therefore did not make individual features more reliable flower detectors in this small test.

The separately trained box head did rank the same held-out images somewhat better after fine-tuning: flower AUC was 0.792 vs. 0.729 for the frozen-SAE head; average precision was 0.892 vs. 0.844. This suggests the head learned a more useful combination of spatial codes, while individual SAE features did not become more concept-specific. These estimates are noisy because there are only 20 held-out images and labels are automatic. Per-feature scores are in [`visualizations/flower_auc_frozen_vs_finetuned_001/feature_auc.csv`](visualizations/flower_auc_frozen_vs_finetuned_001/feature_auc.csv), with cohort details in the adjacent JSON.

### What backpropagation updated

Training started from the pretrained Top-K SAE checkpoint. For each training image, the SAE encoded the SDXL residual into sparse spatial codes and decoded a reconstruction; a small 1 x 1 convolutional head read the codes and predicted flower/kimono presence and box coordinates at each grid cell. The loss combined normalized reconstruction MSE (weight 1), objectness binary cross-entropy (weight 1, with positive weight 32), and box Smooth L1 regression (weight 5). `loss.backward()` propagated gradients through the prediction head and SAE. Adam used learning rates 1e-3 for the head and 1e-6 for the SAE; decoder weights were renormalized after updates. Training ran for five epochs, and the checkpoint with the lowest validation loss was selected. The frozen baseline used the same objective and data but updated only the head. This taught the SAE to support box prediction; it did not directly train a dedicated flower unit or an object-removal operation.

## Four-step generation check

We repeated feature ablation on four held-out C2C-positive and four negative prompts using four inference steps, with same-prompt, same-seed four-step replays as the image baseline. Zeroing the three selected flower features changed images by mean RGB MAE 28.90/255; half attenuation changed them by 19.70/255; the random-feature control changed them by 6.84/255. However, the four-step replays already differed strongly from the original one-step images (mean MAE 66.64/255 for positives and 57.99/255 for negatives). The flower ornament remained visible in the inspected positive example, while the image style shifted substantially. In this SDXL Turbo setup, four steps amplified the intervention and also caused substantial baseline drift, so it did not solve flower removal.

![Four-step positive flower comparison: one-step source, four-step replay, feature ablations, no-op, and random control](visualizations/flower_intervention_4step_001/positive_seed007.jpg)

Per-image measurements are in [`visualizations/flower_intervention_4step_001/metrics.csv`](visualizations/flower_intervention_4step_001/metrics.csv).
