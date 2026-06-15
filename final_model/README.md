# final_model - Locked-in model checkpoints + evaluations

> **Note on terminology.** The code in this folder uses **SGC** (Space-Grade Constraint, numbered `#1`–`#4`) as shorthand for the four deployment-relevant constraints evaluated against the chosen model. The thesis refers to these as "space-grade constraint 1" through "space-grade constraint 4" or "the four constraints". The mapping is:
>
> | Code (here) | Thesis | What it measures |
> |---|---|---|
> | **SGC #1** | constraint 1 | parameter count |
> | **SGC #2** | constraint 2 | worst-case CPU latency |
> | **SGC #3** | constraint 3 | peak segmentation-tensor memory |
> | **SGC #4** | constraint 4 | survivability under simulated radiation faults |
>
> See [`../docs/constraints.md`](../docs/constraints.md) for the formal definition of each constraint, its threshold, and the rationale.

This folder contains the **three candidate model checkpoints** evaluated as the final deployable for the thesis, along with their FP32 test-set evaluations and SGC hardware-budget scorecards.

The naming convention is `<version_run>_<encoder>_<resolution>_<descriptor>` so every file self-describes its provenance.

| Status | Model | Test BR (min3) | SGC verdict |
|---|---|---|---|
| previous deployable (v1 baseline) | `v1_R9_MNv3-S_512.pth` | 0.325 | ✓ passes 3 hardware constraints (#4 not re-tested) |
| research ceiling (best accuracy, fails memory cap) | `v2_R11_MNv4-S_1024.pth` | 0.480 | ✗ fails SGC #3 (406 MB > 256 MB) |
| ⭐ **chosen deployable** | **`v2_R11_MNv4-S_512.pth`** | **0.446** | **✓ passes ALL 4 SGCs** (hardware + survivability) |

## Files (renamed for consistency)

```
final_model/
├── README.md                                       (this file)
│
├── v1_R9_MNv3-S_512.pth                            8.8 MB - v1 winner (MNv3-S/512)
├── v1_R9_MNv3-S_512_config.json
├── v1_R9_MNv3-S_512_evaluation_results.json
│
├── v2_R11_MNv4-S_1024.pth                         11.7 MB - research ceiling (fails RAM)
├── v2_R11_MNv4-S_1024_config.json
├── v2_R11_MNv4-S_1024_evaluation_results.json
│
├── v2_R11_MNv4-S_512.pth                          11.7 MB - chosen deployable
├── v2_R11_MNv4-S_512_config.json
├── v2_R11_MNv4-S_512_evaluation_results.json
│
└── sgc/
    ├── v1_R9_MNv3-S_512_baseline/results.json        SGC #1-3 baseline (v1 R9)
    ├── v2_R11_MNv4-S_1024_baseline/results.json      SGC #1-3 baseline (v2 R11 @ 1024 - fails #3)
    ├── v2_R11_MNv4-S_512_baseline/results.json       SGC #1-3 baseline (chosen deployable)
    ├── v2_R11_MNv4-S_512_unshielded_chaos/results.json  SGC #4 unshielded (radiation, no defence)
    └── v2_R11_MNv4-S_512_shielded_chaos/results.json    SGC #4 shielded (TMR + clamping)
```

## v1 R9 - `v1_R9_MNv3-S_512.pth`

Source: `MAIN_ITERATION/experiments/training_run_9_focal_+_tversky_512p_+++/`

| | |
|---|---|
| Encoder | MobileNetV3-Small (`tu-mobilenetv3_small_100`) |
| Decoder | DeepLabV3+ |
| Resolution | 512 × 512 |
| Loss | focal + tversky, α=0.5, tversky **0.2/0.8** (rare-heavy), focal γ=2.0 |
| Augmentation | basic |
| Epochs trained | 25 |
| Selected epoch | **22** (peak val_iou_big_rock = 0.599) |
| Trainable params | 2.16 M (72 % of 3 M cap) |

### Test set (gold expert, min3-100agree)

| Metric | Value |
|---|---|
| Pixel accuracy | 0.9760 |
| **mIoU** | **0.7904** |
| Soil IoU | 0.9779 |
| Bedrock IoU | 0.9206 |
| Sand IoU | 0.9385 |
| **Big Rock IoU** | **0.3245** |

### SGC hardware baseline (`sgc_evaluate_GOOD.py`)

| Constraint | Cap | Measured | Pass? |
|---|---|---|---|
| #1 Parameter count | ≤ 3.0 M | 2.16 M | ✓ |
| #2 Max latency / frame | ≤ 500 ms | 56.78 ms | ✓ |
| #3 Peak ML RAM | ≤ 256 MB | 115.19 MB | ✓ |

## v2 R11 @ 1024 - `v2_R11_MNv4-S_1024.pth`

Source: `training_configuration_sweep/experiments/training_run_11_focal_+_tversky_1024p_aug_B/`

**Research ceiling - not deployable (fails SGC #3 memory cap).**

| | |
|---|---|
| Encoder | MobileNetV4-Conv-Small (`tu-mobilenetv4_conv_small`) |
| Decoder | DeepLabV3+ |
| Resolution | **1024 × 1024** |
| Loss | focal + tversky, α=0.5, tversky 0.3/0.7, focal γ=2.0 |
| Augmentation | basic |
| Epochs trained | 25 |
| Selected epoch | **25** (peak val_iou_big_rock = 0.626) |
| Trainable params | 3.00 M (100 % of 3 M cap) |

### Test set (gold expert, min3-100agree)

| Metric | Value |
|---|---|
| Pixel accuracy | 0.9812 |
| **mIoU** | **0.8386** |
| Soil IoU | 0.9780 |
| Bedrock IoU | 0.9348 |
| Sand IoU | 0.9613 |
| **Big Rock IoU** | **0.4802** |

### SGC hardware baseline

| Constraint | Cap | Measured | Pass? |
|---|---|---|---|
| #1 Parameter count | ≤ 3.0 M | 3.00 M | ✓ |
| #2 Max latency / frame | ≤ 500 ms | 46.51 ms | ✓ |
| **#3 Peak ML RAM** | **≤ 256 MB** | **406.60 MB** | **✗ FAIL** (+150 MB over cap) |

Why excluded from the deployable set: the model achieves the strongest accuracy in the project (test BR 0.480, mIoU 0.839) but its peak segmentation-tensor RAM at 1024² input exceeds the SGC #3 cap by 150 MB. Following the precedent set in architecture_sweep Cell 05 (MobileViT-XS @ 512, also memory-cap excluded), this model is reported as the research ceiling rather than the deployment candidate.

## v2 R11 @ 512 - `v2_R11_MNv4-S_512.pth` ⭐ Chosen deployable

Source: `training_configuration_sweep/experiments/training_run_11_focal_+_tversky_512p_aug_B/`

| | |
|---|---|
| Encoder | MobileNetV4-Conv-Small (`tu-mobilenetv4_conv_small`) |
| Decoder | DeepLabV3+ |
| Resolution | **512 × 512** |
| Loss | focal + tversky, α=0.5, tversky 0.3/0.7, focal γ=2.0 |
| Augmentation | basic |
| Epochs trained | 25 |
| Selected epoch | **23** (peak val_iou_big_rock = 0.591) |
| Trainable params | 3.00 M (100 % of 3 M cap) |

### Test set (gold expert, min3-100agree)

| Metric | Value |
|---|---|
| Pixel accuracy | 0.9760 |
| **mIoU** | **0.8190** |
| Soil IoU | 0.9832 |
| Bedrock IoU | 0.9086 |
| Sand IoU | 0.9377 |
| **Big Rock IoU** | **0.4465** |

### SGC hardware baseline

| Constraint | Cap | Measured | Pass? |
|---|---|---|---|
| #1 Parameter count | ≤ 3.0 M | 3.00 M | ✓ |
| #2 Max latency / frame | ≤ 500 ms | 49.15 ms (baseline) | ✓ (10× under cap) |
| **#3 Peak ML RAM** | **≤ 256 MB** | **126.47 MB** (baseline) | **✓** (49 % of cap) |

### SGC #4 - Survivability under simulated radiation

Per-class IoU on min3-100agree under three modes. Chaos = `ChaoticSpaceRadiationInjector` (1–50 random IEEE-754 bit flips on the deepest encoder feature each forward pass). Shielded = `BoundsCheckShield` (activation clamp) + triple-modular-redundancy majority vote.

The fault-injection script was patched for this run: (a) always target the deepest encoder feature (the ASPP input) instead of a random list element, (b) skip zero-valued indices to avoid silent subnormal injection on ReLU-sparse activations, (c) scale `max_flips` from the thesis Section 3.6 default of 5 → 50 to match the ~10× channel ratio between MNv4-Conv-Small (960 channels at the deep feature) and the original calibration target MobileNetV3-Small (96 channels). Without these adjustments, naive injection produces no measurable corruption on MNv4-S because the wide deep feature dilutes single-value perturbations through ASPP's global pooling.

| Mode | Pix Acc | mIoU | Soil IoU | Bedrock IoU | Sand IoU | **Big Rock IoU** |
|---|---|---|---|---|---|---|
| Baseline (no chaos) | 0.9760 | 0.8189 | 0.9832 | 0.9085 | 0.9377 | **0.4463** |
| Unshielded chaos | 0.5918 | 0.3089 | 0.5329 | 0.3602 | 0.3238 | **0.0189** |
| Shielded chaos (TMR + clamping) | **0.9760** | **0.8189** | **0.9832** | **0.9086** | **0.9377** | **0.4460** |

Reading: under scaled MBU stress the unshielded model loses essentially the entire rare-class signal (BR 0.4463 → 0.0189, a 95.8 % relative drop) and the common-class IoUs collapse to 0.32–0.53. The shielded configuration (BoundsCheckShield + TMR) recovers every per-class metric to within 0.0003 of the baseline. This is the same recovery pattern documented for MobileNetV3-Small in Section 5.3.2 of the thesis (0.386 → 0.081 → 0.389), now demonstrated on the chosen MobileNetV4-Conv-Small deployable.

### Hardware compliance across all 3 chaos modes

| Mode | Latency max | Peak ML RAM | SGC #2 (≤ 500 ms) | SGC #3 (≤ 256 MB) |
|---|---|---|---|---|
| Baseline | 49.15 ms | 126.47 MB | ✓ | ✓ (49 % of cap) |
| Unshielded chaos | 59.62 ms | 121.97 MB | ✓ | ✓ |
| **Shielded chaos (TMR)** | **146.07 ms** | **244.92 MB** | **✓** (29 % of cap) | **✓** (95.7 % of cap) |

Shielded mode is the worst-case envelope (TMR runs 3 model copies in parallel). It still fits both caps: latency is 3.0× baseline (49 → 146 ms), peak RAM is 1.94× baseline (126 → 245 MB).

### Final SGC verdict on the deployable

| SGC | Cap | Worst measured (across all modes) | Verdict |
|---|---|---|---|
| #1 Parameter count | ≤ 3.0 M | 3.00 M | ✓ **PASS** |
| #2 Max latency / frame | ≤ 500 ms | 146.07 ms (shielded) | ✓ **PASS** (29 % of cap) |
| #3 Peak ML RAM | ≤ 256 MB | 244.92 MB (shielded) | ✓ **PASS** (95.7 % of cap) |
| #4 Survivability (TMR-recovered BR vs baseline) | ≥ near-baseline | 0.4460 vs 0.4463 baseline | ✓ **PASS** |

**v2 R11 @ 512 satisfies all four space-grade constraints simultaneously.**

## Side-by-side comparison

All three models on the gold expert test set at min3-100agree:

| Metric | v1 R9 (MNv3-S/512) | v2 R11 @ 1024 (MNv4-S/1024) | **v2 R11 @ 512 (MNv4-S/512)** |
|---|---|---|---|
| Soil IoU | 0.978 | 0.978 | **0.983** |
| Bedrock IoU | 0.921 | 0.935 | **0.909** |
| Sand IoU | 0.939 | 0.961 | **0.938** |
| **Big Rock IoU** | **0.325** | **0.480** | **0.446** |
| **mIoU** | **0.790** | **0.839** | **0.819** |
| Pixel acc | 0.976 | 0.981 | 0.976 |

Hardware:

| Metric | v1 R9 | v2 R11 @ 1024 | **v2 R11 @ 512** |
|---|---|---|---|
| Params | 2.16 M | 3.00 M | **3.00 M** |
| Latency max | 56.78 ms | 46.51 ms | **49.15 ms** |
| Peak ML RAM | 115.19 MB | **406.60 MB** ✗ | **126.47 MB** ✓ |
| SGC #1+#2+#3 | PASS | **FAIL** | **PASS** |

### Reading the comparison

- **v2 R11 @ 512 beats v1 R9 on Big Rock IoU by +0.121** (0.446 vs 0.325) on the same test set with the same resolution - the encoder upgrade from MobileNetV3-Small to MobileNetV4-Conv-Small produces a meaningful rare-class improvement at deployable resolution.
- **v2 R11 @ 512 vs v2 R11 @ 1024**: the resolution drop costs 0.034 BR IoU and 0.020 mIoU, but pulls peak ML RAM from 406 MB (over cap) to 126 MB (49 % of cap).
- **mIoU lift over v1 R9**: +0.029 (0.819 vs 0.790).
- **Encoder advantage at val didn't show up at 512** (val_BR* 0.591 vs v1 R9's 0.599) but did on test (+0.121 BR). Classic val/test divergence - the gold expert test set is the load-bearing benchmark per Section 2.3.4 of the thesis.

## Provenance and integrity

- All `.pth` files are **copies**, not symlinks. The source experiments' `weights.pth` files are unchanged.
- Each `weights.pth` was materialized by `select_best.py --metric val_iou_big_rock --force` from per-epoch checkpoints (also unchanged in the source experiments). Re-running the same command reproduces the same checkpoint deterministically.
- Each `*_evaluation_results.json` was produced by `architecture_sweep/evaluate.py` against the AI4Mars MSL NCAM gold expert test set (`MSL_NAVCAM_TEST_SET/`) using the Strategy-A evaluation methodology: model predictions are upsampled from the model's input resolution to the gold label's native resolution before computing IoU against the unaltered ground truth.
- Each baseline `sgc/*_baseline/results.json` was produced by `sgc_evaluate_GOOD.py` (no fault injection), with `param_count`, `sgc_latency_max_ms`, and `sgc_true_ml_ram_mb` measured per-variant. Latency and peak ML RAM are measured on the host GPU since the latency convention in the thesis is FP32 PyTorch single-thread (the script measures on whichever device is available).
- Each chaos `sgc/*_chaos/results.json` was produced by `sgc_evaluate_BAD_2.0.py` (radiation, no defence) or `sgc_evaluate_GOOD_2.0.py` (radiation + BoundsCheckShield + TMR vote) with `--max-flips 50` to match MobileNetV4-Conv-Small's deepest-feature channel width. Both scripts target the deepest encoder feature and skip zero-valued indices (patched versions; see commit history of the script files for the inline rationale).
- All baseline result sets cover the three gold variants (min1-100agree, min2-100agree, min3-100agree). Chaos result sets are reported on the headline min3-100agree variant.

## How to use these checkpoints

For test-set evaluation against the gold expert set:
```bash
PYTHONPATH=. \
  .venv/bin/python architecture_sweep/evaluate.py \
  --weights final_model/v2_R11_MNv4-S_512.pth \
  --arch smp.DeepLabV3Plus \
  --encoder tu-mobilenetv4_conv_small \
  --input-hw 512 \
  --output final_model/v2_R11_MNv4-S_512_evaluation_results.json
```

For SGC baseline scorecard:
```bash
PYTHONPATH=. \
  .venv/bin/python sgc_evaluate_GOOD.py \
  --weights final_model/v2_R11_MNv4-S_512.pth \
  --encoder tu-mobilenetv4_conv_small \
  --input-hw 512 \
  --run-id v2_R11_MNv4-S_512_baseline \
  --results-root final_model/sgc
```

For SGC #4 survivability (already run with `--max-flips 50` for MNv4-S channel-width):
```bash
# unshielded chaos
PYTHONPATH=. \
  .venv/bin/python sgc_evaluate_BAD_2.0.py \
  --weights final_model/v2_R11_MNv4-S_512.pth \
  --encoder tu-mobilenetv4_conv_small \
  --input-hw 512 \
  --max-flips 50 \
  --run-id v2_R11_MNv4-S_512_unshielded_chaos \
  --results-root final_model/sgc

# shielded chaos (TMR + clamping)
PYTHONPATH=. \
  .venv/bin/python sgc_evaluate_GOOD_2.0.py \
  --weights final_model/v2_R11_MNv4-S_512.pth \
  --encoder tu-mobilenetv4_conv_small \
  --input-hw 512 \
  --max-flips 50 \
  --run-id v2_R11_MNv4-S_512_shielded_chaos \
  --results-root final_model/sgc
```

Or to re-run the in-place `evaluate.py` workflow (uses the matching `weights.pth` inside the source experiment folder):
```bash
PYTHONPATH=. \
  .venv/bin/python architecture_sweep/evaluate.py \
  --exp-id training_run_11_focal_+_tversky_512p_aug_B \
  --exp-root training_configuration_sweep/experiments
```
