# 003 — FPN + MobileNetV3-Small-100 at 512² (ITERATION-2 cross-architecture experiment)

## What

Third ITERATION-2 run. **Same encoder (tu-mobilenetv3_small_100), same training recipe, same input resolution (512×512) as [002_dlv3plus_512](../002_dlv3plus_512/). Only the decoder family differs: FPN (top-down feature pyramid) replaces DeepLabV3+ (ASPP).**

The one variable: decoder family. Everything else (encoder, CE loss, Adam lr 1e-3, batch 8, 10 epochs, no augmentation, no class weights, train/val 80/20 with seed 42, 512² Strategy A evaluation) is held flat per [docs/protocol.md](../../../docs/protocol.md) — so any difference in test-set numbers is attributable to the decoder family alone. This is the first **decoder-isolated** cross-architecture data point in the project.

## Test-set headline numbers (gold expert set, Strategy A)

| Variant | Pixel acc | mIoU | Big Rock IoU | BR Precision | BR Recall | Low-conf @ 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| min1-100agree | 0.9188 | 0.6625 | **0.1076** | 0.1230 | 0.463 | 2.49% |
| min2-100agree | 0.9546 | 0.7075 | 0.1047 | 0.1092 | 0.719 | 1.83% |
| min3-100agree | 0.9763 | 0.8131 | **0.4127** | 0.4244 | 0.938 | 1.02% |

### Per-class IoU at min1 / min3

| Class | min1 IoU | min1 Prec | min1 Rec | min3 IoU | min3 Prec | min3 Rec |
|---|---:|---:|---:|---:|---:|---:|
| soil | 0.909 | 0.963 | 0.942 | 0.976 | 0.992 | 0.985 |
| bedrock | 0.800 | 0.856 | 0.924 | 0.926 | 0.945 | 0.979 |
| sand | 0.833 | 0.953 | 0.870 | 0.937 | 0.977 | 0.958 |
| big_rock | 0.108 | 0.123 | 0.463 | 0.413 | 0.424 | 0.938 |

## Apples-to-apples vs 002_dlv3plus_512 (DLV3+ → FPN, decoder only)

Same encoder (tu-mobilenetv3_small_100), same training recipe, same input resolution (512²); only the decoder family differs.

| Metric | 002 @ 512² (DLV3+) | 003 @ 512² (FPN) | Δ | Direction |
|---|---:|---:|---:|---|
| **Params** | 2.16 M | 2.72 M | +26% | FPN heavier |
| **FP32 checkpoint** | 8.42 MB | 10.39 MB | +23% | larger |
| **FLOPs (forward, 512²)** | 5.84 G | **16.44 G** | **+2.81×** | FPN does ~3× more compute |
| **GPU latency (batch 1)** | 7.6 ms | 8.0 ms | +5% | ≈ same |
| **CPU FP32 latency (1 thread)** | 188 ms | 265 ms | +41% | slower |
| **INT8 CPU latency** | 107 ms | 224 ms | +109% | **2.1× slower** |
| **INT8 model size on disk** | 2.69 MB | 3.33 MB | +24% | larger |
| **Deployment RSS (subprocess)** | 105 MB | **235 MB** | **+124%** | much higher; only 21 MB headroom |
| **Linear-scaled FPS @ 200 MHz** | 0.288 | 0.204 | −29% | worse |
| **RAD750-class FPS** | 0.036 | 0.026 | −29% | worse |
| | | | | |
| min1 pixel acc | 0.9053 | 0.9188 | +0.014 | ↑ FPN slightly better |
| min1 mIoU | 0.6520 | 0.6625 | +0.011 | ↑ FPN slightly better |
| **min1 Big Rock IoU** | **0.1424** | **0.1076** | **−0.035 (−24%)** | **↓ FPN worse on rare class** |
| min1 BR Precision | 0.1951 | 0.1230 | −0.072 | ↓ FPN less selective |
| min1 BR Recall | 0.3450 | 0.4628 | +0.118 | ↑ FPN catches more BR |
| min3 mIoU | 0.8619 | 0.8131 | −0.049 | ↓ FPN worse |
| **min3 Big Rock IoU** | **0.6639** | **0.4127** | **−0.251 (−38%)** | **↓↓ FPN substantially worse on rare class** |
| min3 BR Precision | 0.8138 | 0.4244 | −0.389 | ↓↓ FPN much less selective |
| min3 BR Recall | 0.7828 | 0.9375 | +0.155 | ↑ FPN higher recall |
| Low-confidence < 0.6 @ min1 | 3.20% | 2.49% | −0.71pp | ↑ FPN more confident |

### The cross-architecture finding

**FPN is worse than DeepLabV3+ for Big Rock IoU at the same encoder and the same input resolution**, in two different directions:
- It is heavier (+26% params, +2.81× FLOPs, +124% deployment RSS).
- It scores lower on Big Rock IoU at both gold variants (−24% on min1, −38% on min3).

The pattern within Big Rock metrics is clear and consistent: **FPN trades precision for recall**. At min3, FPN recall is 0.94 (vs DLV3+'s 0.78) but precision drops to 0.42 (vs DLV3+'s 0.81). FPN finds more Big Rocks but is wrong much more often about each — net IoU goes down.

On global metrics (pixel accuracy, min1 mIoU), FPN is slightly better — but the gain is small (+0.014 pixel acc, +0.011 mIoU) and the loss on the headline rare-class metric is large.

**Hypothesis**: DeepLabV3+'s ASPP module concentrates decoder capacity on multi-rate atrous convolutions at a single scale (the output of the encoder). FPN spreads its decoder capacity across the feature pyramid (multiple encoder stages, top-down + lateral). For a *small* model at *low* resolution, the ASPP's concentrated multi-rate dilation is apparently doing more useful work for rare-class detection than FPN's pyramid does — even though FPN has more total decoder parameters.

**Reading for the thesis**: under a 3 M parameter cap and 512² input, **decoder family matters more than parameter count**. FPN has more parameters and more FLOPs than DLV3+, but is materially worse at the rare-class task. This is empirical evidence that bigger ≠ better when the budget is binding.

## Space-grade constraint compliance

| # | Constraint | Target | Result | Status |
|---|---|---|---|---|
| 1 | Clock | 200 MHz | (projection only; 3 levels) | — |
| 2 | Architecture | single-core PowerPC | (simulated single-thread x86) | — |
| 3 | RAM (PyTorch-loaded) | ≤ 256 MB | 1369 MB peak RSS | ❌ (inflated, see #3 deployment) |
| 3 | **RAM (deployment subprocess)** | **≤ 256 MB** | **235 MB peak** (only **21 MB** under cap) | **✅** (barely) |
| 4 | Power | < 20 W | not measured | not verifiable |
| 5 | **Parameters** | **< 3 M** | **2.72 M** (90.7% of cap) | ✅ |
| 6 | **Precision** | **INT8 or FP16** | FP16 (5.30 MB) and INT8 (3.33 MB, 64/64 ops fully INT8) | ✅ |
| 7 | **Throughput** | **> 5 FPS @ 200 MHz** | 0.204 FPS (clock-scaled), 0.026 FPS (RAD750-class), 0.012 FPS (theoretical floor) | ❌ |
| 8 | **Confidence < 0.6 → stop pass/fail (≤ 5% low-conf)** | **2.49%** | **✅** |

**Compliance scorecard: `P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓`** — same letter score as 002, but two warning signs:
- Deployment RSS dropped from 105 MB (002) to 235 MB (003) — close to the cap, would not survive a heavier model in the same family.
- Throughput regressed from 0.036 FPS (002) to 0.026 FPS (003), because FPN's heavier decoder runs ~2× slower per inference at the same resolution.

### Throughput projections (three levels of realism)

| Projection | Latency | FPS | Compliant > 5 FPS? | What it captures |
|---|---:|---:|:-:|---|
| Theoretical floor (1 FLOP/cycle) | 82.18 s | 0.012 | ❌ | 200 MHz × 1 FLOP/cycle on 16.44 G FLOPs |
| Linear-scaled (clock only) | 4.91 s | 0.204 | ❌ | Host FP32 latency × clock ratio |
| **RAD750-class (× 8 microarch penalty)** | **39.29 s** | **0.026** | ❌ | Adds ~4× SIMD gap + ~2× IPC gap |

INT8 path, RAD750-class projection: 224 ms × 18.5 × 8 = **33.2 s → 0.030 FPS**. Same ballpark — still ~165× short of 5 FPS.

## Training stability

10 epochs, no NaN. Wall time **~28 min** on one RTX PRO 4000 Blackwell (vs 002's ~18 min). FPN's heavier decoder is meaningfully slower to train; ~2.8 min per epoch vs ~1.8 min for DLV3+ at the same resolution.

Val accuracy progression: epoch 7 was the best val BR IoU (0.343), with epochs 8–10 hovering or slightly regressing. Final epoch 10: train pixel_acc 0.961, val pixel_acc 0.949.

Per-epoch summary:

| Epoch | Train loss | Val pixel acc | Val mIoU | Val BR IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.477 | 0.916 | 0.640 | 0.142 |
| 3 | 0.213 | 0.941 | 0.705 | 0.238 |
| 5 | 0.165 | 0.945 | 0.716 | 0.242 |
| **7 (peak val BR)** | 0.139 | 0.953 | 0.763 | **0.343** |
| 8 | 0.130 | 0.950 | 0.746 | 0.335 |
| 9 | 0.118 | 0.950 | 0.754 | 0.334 |
| 10 (final) | 0.110 | 0.949 | 0.747 | 0.323 |

Same end-of-training small wobble as 001 and 002. Final-epoch numbers reported above for protocol consistency.

## INT8 quantization results

| Precision | Model size on disk | Reduction vs FP32 | CPU 1-thread latency | Host FPS |
|---|---:|---:|---:|---:|
| FP32 | 10.39 MB | — | 265 ms | 3.78 |
| FP16 | 5.30 MB | 2.0× smaller | (not benchmarked) | — |
| **INT8** | **3.33 MB** | **3.12× smaller** | **224 ms** | **4.47** |

**INT8 fallback audit**: 64 / 64 compute-heavy ops (Conv, ConvTranspose, Gemm, MatMul) wrapped in DequantizeLinear on both data and weight inputs → **100% fully INT8**, zero silent FP fallback. (FPN has 64 such ops vs DLV3+'s 67 — the decoders use different op counts.)

INT8 is only **1.18× faster on host CPU than FP32** for this model — much less than DLV3+'s 2.0× speedup. ONNX-Runtime's INT8 kernels apparently get less benefit from FPN's top-down + lateral structure than from DLV3+'s ASPP. This is a real finding for INT8 deployment with FPN.

## Deployment RSS (clean ORT-only subprocess)

| Milestone | RSS (MB) |
|---|---:|
| After Python imports (CPython + NumPy + ONNX-Runtime) | 44.3 |
| After ORT session create (INT8 model loaded) | 65.6 |
| After warmup forward (activations allocated) | 234.9 |
| **Peak during inference** | **234.9** |

**235 MB peak — passes the 256 MB cap with only 21 MB of headroom**. Compare to 002 at 512² with the same encoder: 105 MB peak (151 MB headroom).

The increase comes from FPN's pyramid: it keeps activations from multiple encoder stages alive in memory simultaneously (for the lateral connections), where DLV3+ only needs the final encoder stage feeding into ASPP. This is the real-world version of "FPN trades memory for capacity at multiple scales" — except in this case the capacity isn't paying off for rare-class IoU.

## What this tells us — for the thesis

1. **Decoder family matters more than parameter count under a 3 M cap**. FPN (2.72 M, 16.4 G FLOPs) lost to DLV3+ (2.16 M, 5.84 G FLOPs) on the headline rare-class metric, despite using 26% more parameters and 2.8× more FLOPs. ASPP outperforms feature-pyramid for Big Rock detection at this scale.
2. **FPN's higher recall + lower precision is a different operating point**, not a strict regression. If a downstream system prefers recall (catch as many rocks as possible, even with false positives), FPN's pattern is fine. For IoU-based reporting (the thesis's headline metric), DLV3+ is clearly better.
3. **FPN narrows the deployment RAM margin substantially** — from 151 MB to 21 MB at the same input resolution. A bigger FPN would not fit.
4. **The throughput verdict holds** at this new architecture too: 0.026 FPS RAD750-class is still ~190× short of 5 FPS. No segmentation architecture in the smp family at the 3 M budget is going to close this gap by decoder choice alone.
5. **The 4-architecture story for the thesis** is now: U-Net (ITERATION-1, 6.63 M, 1024²) vs DLV3+ (1024² and 512²) vs FPN (512²). DLV3+ + small encoder at 512² remains the strongest per-rare-class-IoU result so far.

## Space-grade methodology — same as 001/002 (no changes)

This run uses the same [space_grade.py](../../../space_grade.py) and [space_grade_rss_subprocess.py](../../../space_grade_rss_subprocess.py) as 001/002, plus the constraint #8 binary pass/fail added in the 2026-05-14 re-run. See 001's [notes.md](../001_dlv3plus_baseline/notes.md) "Space-grade methodology" section for the full description of what is and isn't modelled.

## Run artefacts

| File | What |
|---|---|
| `config.json` | Records `model.arch = smp.FPN`, `data.input_hw = 512`, all other fields parallel to 001/002 |
| `weights.pth` | 10.39 MB FP32 final checkpoint (larger than 002's 8.42 MB) |
| `weights_fp16.pth` | 5.30 MB FP16 state-dict |
| `model_fp32.onnx` + `.onnx.data` | ONNX export (split into model + external data because larger than 2 GB protobuf limit margin) |
| `model_int8.onnx` | 3.33 MB INT8 quantized export |
| `training_history.csv` | 29-column per-epoch metrics |
| `evaluation_results.json` | Gold-test results across all 3 variants; includes `eval_input_hw=512` and `eval_strategy="A"` |
| `space_grade.json` | Full constraint scorecard (3 throughput projections, INT8 audit, dual RAM measurements, confidence pass/fail) |
| `pipeline.sh` | Chained train → eval → space_grade script (used for this nohup run) |
| `pipeline.log` | High-level pipeline stage log |
| `training.log` / `evaluate.log` / `space_grade.log` | Raw stdout per stage |
| `PIPELINE_DONE` | Marker file written when all three stages succeeded; contains finish timestamp |

## What is NOT in this run

- No augmentation
- No class weighting / focal loss
- No quantization-aware training (only post-training quantization tested)
- No actual embedded-hardware benchmark
- No bit-flip / radiation-tolerance simulation (out of scope per thesis §1.6)
- No second random seed
- No FPN at 1024² for full apples-to-apples vs 001 (deferred — the 512² point already shows FPN doesn't help, so the 1024² point isn't needed for the cross-architecture story)
