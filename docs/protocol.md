# Evaluation protocol

This document describes the fixed protocol under which every experiment in this repository is trained and evaluated. Holding the protocol constant is what lets the controlled experiment in the thesis attribute differences in Big Rock IoU to a single design lever at a time.

For the constraint thresholds applied to the final model, see [`constraints.md`](constraints.md).
For which experiments are in scope for the thesis, see [`../thesis_scope.md`](../thesis_scope.md).

---

## Dataset

| | |
|---|---|
| Source | AI4Mars (Swan et al., CVPRW 2021) |
| Subset used | Curiosity NAVCAM only |
| Training labels | Merged labels at ≥ 3 labellers and > 65% pixel agreement (the release default) |
| Ignore-pixel value | `255` - excluded from loss and from all metrics |
| Per-image masks (training only) | The rover mask (`mxy`) and range mask (`rng-30m`) are ORed into the label and overwrite those pixels with `255` before training. The gold test labels already have these masks baked in. |

The Spirit and Opportunity subsets of AI4Mars and the MER NCAM data are out of scope for this thesis and are not used in any run in this repository.

---

## Test set

The expert-labelled gold test set has 322 images and is released at three agreement-strictness variants.

| Variant | Big Rock images / 322 | Role |
|---|---|---|
| `min1-100agree` | 53 | reported for transparency |
| `min2-100agree` | - | reported for transparency |
| **`min3-100agree`** | **5** | **headline reporting variant** for the thesis (all three labellers agree) |

The headline variant matches the variant under which the closest comparable transformer baseline (Mohammad et al. 2025) reports its rare-class IoU on the same dataset. The other two variants are computed and saved alongside in each run's `evaluation_results.json`.

Fully-ignored test images (label is all `255`) are skipped during evaluation. The skipped count is logged.

---

## Train / validation / test split

| | |
|---|---|
| Test | The gold expert test set above. Never seen during training. Touched once per final model. |
| Train + validation | 80 / 20 random split of the released training set, `random_state = 42` on sorted file lists. |
| Validation role | Drives all selection decisions (best-validation-epoch checkpoint, architecture-stage winner, training-configuration-stage winner). |

After offline refinement, the training set used by this repository contains **16,063 image-label pairs** (those for which the EDR image, the merged label, and both mask files are all present).

---

## Input pipeline

| | |
|---|---|
| Native input resolution | 1024 × 1024 (the AI4Mars NCAM size) |
| Working resolution | Variable in this thesis (256, 512, or 1024) per the resolution lever. Default: 1024. |
| Resize on EDR image | Bilinear (`cv2.INTER_LINEAR`) |
| Resize on label | Nearest-neighbour (`cv2.INTER_NEAREST`) - preserves discrete class indices |
| Channel order | RGB (OpenCV BGR is converted at load time) |
| Normalization | Divide by `255.0`. No per-channel mean/std subtraction. |

### Strategy-A evaluation

When a model is trained at a non-1024 resolution and evaluated against the 1024 × 1024 gold labels, the evaluation pipeline:

1. resizes the input image to the model's training resolution
2. runs the forward pass at that resolution
3. **bilinear-upsamples the logits back to 1024 × 1024** before applying `argmax`

This keeps IoU computation against the unaltered gold labels, so cross-resolution numbers stay directly comparable. The chosen resolution is recorded in each run's `config.json` under `data.input_hw`.

---

## Augmentation pipelines

Three named pipelines are defined in [`training/augmentations.py`](../training/augmentations.py):

| Name | Used by | Transforms |
|---|---|---|
| `none` | Architecture sweep (Stage 1), and runs R2–R8 of the training-configuration sweep | Just `ToFloat` + `ToTensorV2` - bit-identical to the dataset's no-transform fallback |
| **`basic`** | **R11 (chosen final model), and runs R10/R12/R13** | Horizontal flip (p=0.5), brightness/contrast jitter (±0.15, p=0.5), hue/sat jitter (±5/±10, p=0.3), small affine warp (rotate ±8°, translate ±5%, p=0.3) |
| `strong` | R13 only | Higher-amplitude versions of all `basic` transforms plus optional Gaussian noise / blur |

All augmentation is applied only to the training set. Validation and test are never augmented.

Geometric transforms use `mask_fill = 255` so any pixels introduced by rotation or translation become ignore-pixels and do not contribute to the loss.

---

## Loss functions

A custom loss factory (`training/losses.py`) exposes five named losses through the trainer's `--loss` flag:

| Name | Definition |
|---|---|
| `ce` | Cross-entropy (with optional `--class-weights` for inverse-frequency reweighting) |
| `focal` | Focal loss (Lin et al. 2017) at `gamma = 2.0` |
| `focal_dice` | Convex combination α · Focal + (1−α) · Dice |
| **`focal_tversky`** | **Convex combination α · Focal + (1−α) · Tversky.** Tversky term uses asymmetric α, β weights on the false-positive and false-negative terms. |
| `ftl` | Abraham-Khan single-term focal Tversky formulation |

The chosen final model (R11) uses `focal_tversky` with Tversky weights `α = 0.3`, `β = 0.7`. The asymmetric weighting penalises missed Big Rock pixels more heavily than wrongly flagged ones, biasing the model toward higher recall on the rare class.

All losses honour the `255` ignore label.

---

## Optimizer and training duration

| | |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3`, no schedule |
| Weight decay | `0` (default) |
| Batch size | 16 |
| Epochs | 10 (architecture stage), 15 or 25 (training-configuration stage depending on the run) |
| Early stopping | Not used. The best-validation-epoch checkpoint is selected post-hoc from `training_history.csv`. |

---

## Metrics reported per run

All of these are derived analytically from a single confusion matrix per test variant, accumulated in 64-bit integer arithmetic:

- Pixel accuracy
- Mean IoU over labelled classes (255 excluded)
- Per-class IoU, precision, recall, F1, support - for soil, bedrock, sand, Big Rock
- Macro precision, recall, F1 - over classes with non-zero support

See [`../metrics.py`](../metrics.py) for the implementation.

---

## Reproducibility hooks

| | |
|---|---|
| Random seed | `random_state = 42` for the train/val split (sklearn `train_test_split` on sorted file lists) |
| Checkpoint loading | `torch.load(..., weights_only=True)` |
| Per-run artifacts | `config.json` (CLI flags + protocol parameters), `weights.pth` (best-validation-epoch checkpoint), `training_history.csv` (per-epoch metrics), `evaluation_results.json` (gold test-set IoU at all three variants) |
| Per-run logs | Each run's training log is dropped after the run completes; per-epoch metrics in `training_history.csv` are the canonical record |

---

## Hardware

Training runs were executed on a university NVIDIA RTX-class GPU (single card). Inference latency reported in the thesis is taken on a single CPU thread under the FP32 PyTorch runtime to approximate the rover-class deployment envelope ([`constraints.md`](constraints.md) §SGC #2).
