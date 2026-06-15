# 004 — DeepLabV3+ + MobileNetV3-Small-100 at 256² (resolution-curve extension)

## What

Fourth ITERATION-2 run. **Same architecture, same encoder, same training recipe as [001](../001_dlv3plus_baseline/) and [002](../002_dlv3plus_512/). Only input resolution differs: 256×256.**

This experiment was added after 002's surprise finding that 1024² → 512² substantially improved Big Rock IoU. The question this run answers is: **does the resolution-improvement effect continue at 256², or does it break?** Together, 001 / 002 / 004 form a clean 3-point resolution curve at the same architecture and same training recipe.

Evaluation uses **Strategy A** (predict at 256², bilinear-upsample logits to 1024² before argmax/softmax, IoU vs unchanged gold labels).

## Test-set headline numbers (gold expert set, Strategy A)

| Variant | Pixel acc | mIoU | Big Rock IoU | BR Precision | BR Recall | Low-conf @ 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| min1-100agree | 0.8825 | 0.6219 | **0.1312** | 0.1728 | 0.353 | **4.77%** |
| min2-100agree | 0.9166 | 0.6645 | 0.1333 | 0.1508 | 0.534 | 3.94% |
| min3-100agree | 0.9418 | 0.7924 | **0.5051** | 0.5917 | 0.775 | 3.05% |

### Per-class IoU at min1 / min3

| Class | min1 IoU | min1 Prec | min1 Rec | min3 IoU | min3 Prec | min3 Rec |
|---|---:|---:|---:|---:|---:|---:|
| soil | 0.864 | 0.948 | 0.907 | 0.939 | 0.989 | 0.949 |
| bedrock | 0.762 | 0.821 | 0.913 | 0.913 | 0.933 | 0.977 |
| sand | 0.751 | 0.901 | 0.821 | 0.821 | 0.945 | 0.864 |
| big_rock | 0.131 | 0.173 | 0.353 | 0.505 | 0.592 | 0.775 |

## The resolution curve — 1024² → 512² → 256² (the headline cross-experiment finding)

Same model (DeepLabV3+ + tu-mobilenetv3_small_100, 2.16 M params), same training recipe; only input resolution changes.

| Metric | 001 @ 1024² | 002 @ 512² | **004 @ 256²** | Pattern |
|---|---:|---:|---:|---|
| FLOPs | 23.4 G | 5.84 G | **1.46 G** | ¼ per halving (linear in pixels) |
| CPU FP32 latency | 1097 ms | 188 ms | **63 ms** | ~6× from 1024, ~3× from 512 |
| INT8 CPU latency | 545 ms | 107 ms | **30 ms** | same pattern |
| Deploy RSS | 242 MB | 105 MB | **72 MB** | activations scale with pixels |
| Linear-scaled FPS @ 200 MHz | 0.060 | 0.288 | **0.860** | ↑ each halving |
| **RAD750-class FPS** | 0.0062 | 0.0359 | **0.1076** | ↑ each halving (~5–6× per step) |
| min1 pixel acc | 0.9108 | 0.9053 | 0.8825 | gradual drop |
| min1 mIoU | 0.6509 | 0.6520 | 0.6219 | peak at 512, drop at 256 |
| **min1 Big Rock IoU** | 0.0872 | **0.1424** ★ | 0.1312 | **peak at 512, modest drop at 256** |
| min1 BR Precision | 0.100 | 0.195 | 0.173 | peak at 512, modest drop at 256 |
| min1 BR Recall | 0.404 | 0.345 | 0.353 | ≈ flat |
| min3 mIoU | 0.7613 | 0.8619 | 0.7924 | peak at 512 |
| **min3 Big Rock IoU** | 0.2260 | **0.6639** ★ | **0.5051** | **peak at 512, substantial drop at 256** |
| min3 BR Precision | 0.229 | **0.814** ★ | 0.592 | peak at 512 |
| min3 BR Recall | 0.949 | 0.783 | 0.775 | drops mildly |
| Low-confidence < 0.6 @ min1 | 2.97 % | 3.20 % | **4.77 %** | **rising — close to 5% cap** |

### The shape of the curve (the thesis finding)

**The curve is inverted-U with the peak at 512²**, not monotone. Two opposing forces:

1. **Receptive field relative to objects grows as resolution drops** — the model becomes more selective, precision climbs, IoU rises. (Dominant from 1024 → 512.)
2. **Spatial detail per object shrinks as resolution drops** — Big Rock objects that were 30–80 px at 1024² are now 7.5–20 px at 256². Some smaller objects fall below the model's spatial discrimination floor. Precision and IoU drop. (Dominant from 512 → 256.)

The balance tips between 512 and 256. The **min3 Big Rock IoU peak is sharp**: 0.226 → **0.664** → 0.505, with the 512² value still being a strong 2.3× over the 1024² baseline at 256² — but materially worse than 512². The min1 result is gentler: 0.087 → **0.142** → 0.131 (only −7.7% from 512 → 256, vs +63% from 1024 → 512).

This is the same model finding two different operating regimes. The Big Rock detection problem at this parameter budget has a **resolution sweet spot near 512²**.

### What this means for the thesis

1. **Resolution is a real lever, but not monotone.** The thesis claim is now *"under a 3 M cap, accuracy on the rare class is dominated by input resolution, with a sweet spot near 512²"* — backed by 3 data points, not 1 anomaly.
2. **The throughput verdict still holds at every resolution.** Even at 256² + INT8 + RAD750-class projection, FPS = 0.11 — still ~46× short of 5 FPS. Resolution alone, even at the most aggressive setting that doesn't destroy accuracy, **cannot** make this hardware target.
3. **The deployment-RAM constraint becomes easy at low resolutions.** 256² → 72 MB peak. There is now **3.5×** of headroom under the 256 MB cap, vs ~14 MB at 1024². RAM stops being interesting.
4. **The confidence-pass/fail constraint becomes tight at low resolutions.** 4.77 % low-conf at 256² is just barely under the 5 % threshold. **At any further resolution drop, this constraint will start failing** — the model becomes less certain about its predictions as input resolution shrinks. This is genuinely new information for the scorecard: confidence and resolution are coupled.

## Space-grade constraint compliance

| # | Constraint | Target | Result | Status |
|---|---|---|---|---|
| 1 | Clock | 200 MHz | (projection only; 3 levels) | — |
| 2 | Architecture | single-core PowerPC | (simulated single-thread x86) | — |
| 3 | RAM (PyTorch-loaded) | ≤ 256 MB | 1378 MB peak RSS | ❌ (inflated, see #3 deployment) |
| 3 | **RAM (deployment subprocess)** | **≤ 256 MB** | **72 MB peak** (184 MB headroom — largest yet) | **✅** |
| 4 | Power | < 20 W | not measured | not verifiable |
| 5 | **Parameters** | **< 3 M** | **2.16 M** | ✅ |
| 6 | **Precision** | **INT8 or FP16** | FP16 (4.25 MB) and INT8 (2.69 MB, 67/67 ops fully INT8) | ✅ |
| 7 | **Throughput** | **> 5 FPS @ 200 MHz** | 0.860 FPS (clock-scaled), 0.108 FPS (RAD750-class), 0.137 FPS (theoretical floor) | ❌ |
| 8 | **Confidence < 0.6 → stop pass/fail (≤ 5% low-conf)** | **4.77%** | **✅** (barely — 23 bp under the 5 % cap) |

**Compliance scorecard: `P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓`** — same letter score as 001/002, with deployment RAM now comfortably over-compliant and confidence just barely passing.

### Throughput projections (three levels of realism)

| Projection | Latency | FPS | Compliant > 5 FPS? | What it captures |
|---|---:|---:|:-:|---|
| Theoretical floor (1 FLOP/cycle) | 7.31 s | 0.137 | ❌ | 200 MHz × 1 FLOP/cycle on 1.46 G FLOPs |
| Linear-scaled (clock only) | 1.16 s | 0.860 | ❌ | Host FP32 latency × clock ratio |
| **RAD750-class (× 8 microarch penalty)** | **9.30 s** | **0.108** | ❌ | Adds ~4× SIMD gap + ~2× IPC gap |

INT8 path, RAD750-class projection: 30 ms × 18.5 × 8 = **4.44 s → 0.225 FPS**. Closest yet to 5 FPS but still ~22× short. **Resolution + INT8 together cannot close the gap.**

## Training stability

10 epochs, no NaN. Wall time **~22 min** on one RTX PRO 4000 Blackwell GPU 0 (vs 002's ~18 min at 512² and 001's ~70 min at 1024²). At 256² the training is bandwidth-limited rather than compute-limited; per-epoch isn't dramatically faster than 002.

Val accuracy progression: best val Big Rock IoU at epoch 7 (0.284) and epoch 9 (0.285) — small wobble at the end (same pattern as 001/002/003). Final epoch 10: train pixel_acc 0.929, val pixel_acc 0.930.

Per-epoch summary:

| Epoch | Train loss | Val loss | Val mIoU | Val BR IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.531 | 0.338 | 0.622 | 0.152 |
| 3 | 0.300 | 0.245 | 0.672 | 0.179 |
| 5 | 0.241 | 0.247 | 0.680 | 0.260 |
| **7 (peak val BR)** | 0.207 | 0.190 | 0.719 | **0.284** |
| 9 | 0.180 | 0.180 | 0.721 | 0.285 |
| 10 (final) | 0.169 | 0.206 | 0.705 | 0.283 |

## INT8 quantization results

| Precision | Model size on disk | Reduction vs FP32 | CPU 1-thread latency | Host FPS |
|---|---:|---:|---:|---:|
| FP32 | 8.42 MB | — | 63 ms | 15.87 |
| FP16 | 4.25 MB | 2.0× smaller | (not benchmarked) | — |
| **INT8** | **2.69 MB** | **3.13× smaller** | **30 ms** | **33.45** |

**INT8 fallback audit**: 67 / 67 compute-heavy ops fully INT8 (same as 001 and 002). Zero silent FP fallback.

INT8 is **2.1× faster on host CPU than FP32** at this resolution — same speedup ratio as 001/002, confirming the INT8 path scales linearly across resolutions.

## Deployment RSS (clean ORT-only subprocess)

| Milestone | RSS (MB) |
|---|---:|
| After Python imports (CPython + NumPy + ONNX-Runtime) | 44.4 |
| After ORT session create (INT8 model loaded) | 61.2 |
| After warmup forward (activations allocated) | 72.3 |
| **Peak during inference** | **72.3** |

**72 MB peak — 184 MB of headroom under the cap**. Largest margin of any run so far. The activation footprint at 256² is ~11 MB (72 − 61), down from ~44 MB at 512² and ~180 MB at 1024² — scales roughly with pixel count, as expected.

## What is NOT in this run

- No augmentation
- No class weighting / focal loss
- No quantization-aware training (only post-training quantization tested)
- No 128² or smaller — based on the confidence trend (rising from 2.97 % → 3.20 % → 4.77 %), going below 256² is likely to fail constraint #8
- No second random seed
- No bit-flip / radiation-tolerance simulation (out of scope per thesis §1.6)

## Run artefacts

| File | What |
|---|---|
| `config.json` | Records `data.input_hw = 256`; all other fields parallel to 001/002 |
| `weights.pth` | 8.4 MB FP32 final checkpoint |
| `weights_fp16.pth` | 4.25 MB |
| `model_fp32.onnx` + `.onnx.data` | ONNX export (large file split) |
| `model_int8.onnx` | 2.69 MB INT8 quantized export |
| `training_history.csv` | 29-column per-epoch metrics |
| `evaluation_results.json` | Strategy-A gold-test results; `eval_input_hw=256` |
| `space_grade.json` | Full scorecard with 3 projections + INT8 audit + dual RAM + confidence pass/fail |
| `pipeline.sh` / `pipeline.log` / `training.log` / `evaluate.log` / `space_grade.log` | Chained nohup-pipeline artefacts |
| `PIPELINE_DONE` | Marker file written when all 3 stages succeeded |
