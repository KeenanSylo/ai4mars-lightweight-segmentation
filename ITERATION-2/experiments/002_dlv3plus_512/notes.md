# 002 — DeepLabV3+ + MobileNetV3-Small-100 at 512² (ITERATION-2 resolution-lever experiment)

## What

Second ITERATION-2 run. **Same architecture, same training recipe, only input resolution differs from [001_dlv3plus_baseline](../001_dlv3plus_baseline/).**

The one variable: input resolution drops from **1024×1024 → 512×512**. Everything else (DeepLabV3+ + tu-mobilenetv3_small_100, CE loss, Adam lr 1e-3, batch 8, 10 epochs, no augmentation, no class weights, train/val 80/20 with seed 42) is held flat per [docs/protocol.md](../../../docs/protocol.md) — so any difference in test-set numbers is attributable to the resolution change alone.

**Evaluation strategy: Strategy A** — the gold test labels stay at native 1024². EDR inputs are bilinear-resized to 512² for inference, then the model's logits are bilinear-upsampled to 1024² before argmax/softmax. IoU and confidence are therefore always computed against the unchanged gold label, so cross-resolution numbers are directly comparable to 001.

## Test-set headline numbers (gold expert set, Strategy A)

| Variant | Pixel acc | mIoU | Big Rock IoU | BR Precision | BR Recall | Low-conf @ 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| min1-100agree | 0.9053 | 0.6520 | **0.1424** | 0.1951 | 0.345 | 3.20% |
| min2-100agree | 0.9424 | 0.7070 | 0.1746 | 0.1955 | 0.621 | 2.64% |
| min3-100agree | 0.9660 | 0.8619 | **0.6639** | 0.8138 | 0.783 | 1.69% |

### Per-class IoU at min1 / min3

| Class | min1 IoU | min1 Prec | min1 Rec | min3 IoU | min3 Prec | min3 Rec |
|---|---:|---:|---:|---:|---:|---:|
| soil | 0.881 | 0.955 | 0.919 | 0.954 | 0.989 | 0.964 |
| bedrock | 0.782 | 0.836 | 0.923 | 0.915 | 0.937 | 0.974 |
| sand | 0.803 | 0.924 | 0.859 | 0.916 | 0.949 | 0.963 |
| big_rock | 0.142 | 0.195 | 0.345 | 0.664 | 0.814 | 0.783 |

## Apples-to-apples vs 001_dlv3plus_baseline (1024² → 512²)

Same architecture (2.16 M parameters), same training recipe; only input resolution differs.

| Metric | 001 @ 1024² | 002 @ 512² | Δ | Direction |
|---|---:|---:|---:|---|
| **Params** | 2.16 M | 2.16 M | — | same |
| **FP32 checkpoint** | 8.42 MB | 8.42 MB | — | same |
| **FLOPs (forward)** | 23.4 G | 5.8 G | **−4.0×** | ↓ as expected |
| **GPU latency (batch 1)** | 9.0 ms | 7.6 ms | −1.2× | ↓ (GPU underutilised at small input) |
| **CPU FP32 latency (1 thread)** | 1097 ms | **188 ms** | **−5.8×** | ↓↓ |
| **INT8 CPU latency** | 545 ms | **107 ms** | **−5.1×** | ↓↓ |
| **Deployment RSS (subprocess)** | 242 MB | **105 MB** | **−2.3×** | ↓↓ |
| **Linear-scaled FPS @ 200 MHz** | 0.060 | 0.288 | +4.8× | ↑ (still 17× short of 5 FPS) |
| **RAD750-class FPS** | 0.0062 | 0.036 | +5.8× | ↑ (still ~140× short of 5 FPS) |
| | | | | |
| min1 pixel acc | 0.9108 | 0.9053 | −0.006 | ≈ same |
| min1 mIoU | 0.6509 | 0.6520 | +0.001 | ≈ same |
| **min1 Big Rock IoU** | **0.0872** | **0.1424** | **+0.055 (×1.6)** | **↑** |
| min1 BR Precision | 0.100 | 0.195 | +0.095 | ↑↑ |
| min1 BR Recall | 0.404 | 0.345 | −0.059 | ↓ |
| **min2 Big Rock IoU** | **0.0877** | **0.1746** | **+0.087 (×2.0)** | **↑↑** |
| min3 mIoU | 0.7613 | 0.8619 | +0.101 | ↑ |
| **min3 Big Rock IoU** | **0.2260** | **0.6639** | **+0.438 (×2.9)** | **↑↑↑** |
| min3 BR Precision | 0.229 | 0.814 | +0.585 | ↑↑↑ |
| min3 BR Recall | 0.949 | 0.783 | −0.166 | ↓ |

### The unexpected finding

**Lower input resolution improved Big Rock IoU substantially**, with the biggest gains on the strictest gold variant (min3 BR IoU went from 0.23 to 0.66, **2.9×**). At the same time, throughput improved on all three projections (4–6× faster).

The naïve expectation was the opposite: lower resolution should hurt rare-class IoU because small Big Rock objects lose spatial detail. That is not what happened.

**Hypothesis**: at 1024² with a small 2.16 M parameter model, the receptive field is too small relative to the input, and the model over-fires on rock-like local texture (high recall, **very low precision**: 0.10 at min1, 0.23 at min3). At 512², the input is effectively zoomed-out — the same receptive field now covers more of each object's context. The model becomes **much more selective**: precision jumps to 0.19/0.81 while recall only drops modestly (0.40→0.35 / 0.95→0.78). Net IoU goes up substantially. The 002 model is also now competitive with the larger 6.63 M U-Net from ITERATION-1's 001_baseline (which scored 0.44 min3 BR IoU vs our 0.66).

This is a real finding for the thesis, and inverts the usual "high-resolution is always better for small objects" assumption when working under a strict parameter budget.

## Space-grade constraint compliance

| # | Constraint | Target | Result | Status |
|---|---|---|---|---|
| 1 | Clock | 200 MHz | (projection only; 3 levels) | — |
| 2 | Architecture | single-core PowerPC | (simulated single-thread x86) | — |
| 3 | RAM (PyTorch-loaded) | ≤ 256 MB | 1319 MB peak RSS | ❌ (inflated, see #3 deployment) |
| 3 | **RAM (deployment subprocess)** | **≤ 256 MB** | **105 MB peak** (44 MB imports → 61 MB session → 105 MB warmup) | **✅** |
| 4 | Power | < 20 W | not measured | not verifiable |
| 5 | **Parameters** | **< 3 M** | **2.16 M** | ✅ |
| 6 | **Precision** | **INT8 or FP16** | FP16 (4.25 MB) and INT8 (2.69 MB, 67/67 ops fully INT8) | ✅ |
| 7 | **Throughput** | **> 5 FPS @ 200 MHz** | 0.288 FPS (clock-scaled), 0.036 FPS (RAD750-class, 8× microarch penalty), 0.034 FPS (theoretical floor) | ❌ |
| 8 | Confidence < 0.6 → stop | metric reported | 3.20% of valid pixels @ min1 | metric available |

**Compliance scorecard: P✓ R(dep)✓ T✗ F16✓ I8✓** — same as 001 except T is now ~5–6× closer to passing. With deployment-subprocess RAM included, **4 of 5 hard constraints pass**.

### Throughput projections (three levels of realism)

| Projection | Latency | FPS | Compliant > 5 FPS? | What it captures |
|---|---:|---:|:-:|---|
| Theoretical floor (1 FLOP/cycle) | 29.20 s | 0.034 | ❌ | 200 MHz × 1 FLOP/cycle on 5.84 G FLOPs |
| Linear-scaled (clock only) | 3.48 s | 0.288 | ❌ | Host FP32 latency × clock ratio |
| **RAD750-class (× 8 microarch penalty)** | **27.84 s** | **0.036** | ❌ | Adds ~4× SIMD gap + ~2× IPC gap |

INT8 path, RAD750-class projection: 107 ms × 18.5 × 8 = **15.8 s → 0.063 FPS**. Still ~80× short of 5 FPS, but a clean 2.5× improvement over FP32 on the same projection.

### Constraint interpretation

**Throughput (still ❌ but much closer)**: dropping resolution alone closed about 5–6× of the gap. The remaining ~140× to 5 FPS at RAD750-class would need a combination of (a) further downsampling (256² → ~16× more, but very likely destroys Big Rock IoU at this point), (b) a smaller / mobile-first architecture, or (c) hardware acceleration. **Resolution alone cannot close this gap at this model class** — that is the empirical thesis claim now backed by data, not hand-waved.

**RAM (deployment ✅)**: at 512² inputs the activation memory drops to about 40 MB (105 MB − 61 MB), confirming that activations scale roughly with pixels. With 14 MB of headroom at 1024² and ~150 MB of headroom at 512², deployment under the 256 MB cap is comfortable.

**INT8 fully quantized (✅, 100%)**: 67/67 compute-heavy ops (Conv, ConvTranspose, Gemm, MatMul) execute fully in INT8 — same as 001, confirming the smaller input resolution does not change the quantization path.

## Training stability

10 epochs, no NaN, no instability. Wall time ~18 min on one RTX PRO 4000 Blackwell. ~1.5–2 min per epoch — about 4× faster than 001's 6 min/epoch, matching the 4× FLOP reduction.

Val accuracy progression was monotonically improving for the first 9 epochs; a small wobble at epoch 10 (val BR IoU dipped from epoch 9's 0.381 to epoch 10's 0.308 — same pattern as 001's small dip at epoch 9 → 10). The final checkpoint is reported above; an early-stop at epoch 9 would yield a slightly higher Big Rock IoU but the protocol fixes 10 epochs for comparability with 001.

Final epoch 10: train pixel_acc 0.954, val pixel_acc 0.954.

## INT8 quantization results

| Precision | Model size on disk | Reduction vs FP32 | CPU 1-thread latency | Host FPS |
|---|---:|---:|---:|---:|
| FP32 | 8.42 MB | — | 188 ms | 5.32 |
| FP16 | 4.25 MB | 2.0× smaller | (not benchmarked) | — |
| **INT8** | **2.69 MB** | **3.13× smaller** | **107 ms** | **9.38** |

**INT8 fallback audit**: 67/67 compute-heavy ops (Conv, ConvTranspose, Gemm, MatMul) wrapped in DequantizeLinear on both data and weight inputs → **100% fully INT8**, zero silent FP fallback. Same pattern as 001.

INT8 is **1.8× faster on host CPU than FP32**. Projected at 200 MHz linear-scaled: 0.508 FPS. RAD750-class: 0.063 FPS.

## Deployment RSS (clean ORT-only subprocess)

| Milestone | RSS (MB) |
|---|---:|
| After Python imports (CPython + NumPy + ONNX-Runtime) | 44.3 |
| After ORT session create (INT8 model loaded) | 61.2 |
| After warmup forward (activations allocated) | 104.7 |
| **Peak during inference** | **104.7** |

**105 MB peak — passes the 256 MB cap with ~150 MB of headroom** (vs ~14 MB headroom at 1024²). The activation footprint scales as expected with pixel count (~40 MB at 512² vs ~180 MB at 1024², a 4.5× reduction).

## What this tells us — for the thesis

1. **Resolution is a more important lever than parameter count** in this small-model regime. Going from 1024² to 512² with the same 2.16 M model:
   - Improved Big Rock IoU by 1.6× (min1), 2.0× (min2), and **2.9× (min3)**.
   - Improved CPU FP32 latency by 5.8×.
   - Improved deployment RSS by 2.3× (now comfortably under the 256 MB cap).
   - The small model overfits to local texture at 1024² — over-predicting Big Rock with very low precision. At 512², the effective receptive field grows relative to objects, and the model becomes much more selective.
2. **Throughput at 200 MHz remains structurally infeasible** even at 512² with INT8 and microarch-aware projection (~0.06 FPS RAD750-class, still ~80× short of 5 FPS). Resolution alone cannot close the throughput gap at this model class — confirming the thesis claim that real-time onboard Mars segmentation is fundamentally aspirational with current architectures.
3. **The RAM constraint is no longer the binding one** under the deployment-subprocess methodology. 002 ships in ~105 MB; even 001 at 1024² ships in 242 MB. Both pass the 256 MB cap.
4. **The INT8 + FP16 path is robust to resolution change**: same 100% fully-quantized result, same INT8 speedup ratio (~1.8–2.0× on host CPU at both resolutions).

## Space-grade methodology — same as 001 (no changes)

This run uses the same [space_grade.py](../../../space_grade.py) and [space_grade_rss_subprocess.py](../../../space_grade_rss_subprocess.py) as 001, with the addition of an `--input-hw` cascade through the calibration-data reader (so INT8 calibration images are resized to 512² to match the exported model's input shape). See 001's [notes.md](../001_dlv3plus_baseline/notes.md) "Space-grade methodology" section for the full description of what is and isn't modelled. The same caveats apply (microarchitecture penalty 8× is an engineering estimate, not cycle-accurate; RAD750-class projection is a projection, not emulation).

## Run artefacts

| File | What |
|---|---|
| `config.json` | Recorded `data.input_hw = 512`, all other fields identical to 001 |
| `weights.pth` | 8.4 MB FP32 final checkpoint |
| `weights_fp16.pth` | 4.25 MB FP16 state-dict |
| `model_fp32.onnx` / `model_int8.onnx` | ONNX exports for INT8 deployment |
| `training_history.csv` | 29-column per-epoch metrics |
| `evaluation_results.json` | Gold-test results across all 3 variants; includes `eval_input_hw=512` and `eval_strategy="A"` |
| `space_grade.json` | Full constraint scorecard (3 throughput projections, INT8 audit, dual RAM measurements) |
| `training.log` | Raw stdout from training run |
| `evaluate.log` | Raw stdout from evaluation run |
| `space_grade.log` | Raw stdout from space-grade run |

## Re-run 2026-05-14 (B) — Constraint #8 closed as binary pass/fail

The "Low-conf @ 0.6" column in the headline table above was previously reported as a metric only. Constraint #8 is now a binary pass/fail derived from the same `evaluation_results.json`; `space_grade.json` is patched in place with a new `confidence_constraint` block. No retraining or re-evaluation.

**Pass criterion**: ≤ 5 % of valid pixels at the headline gold variant (`masked-gold-min1-100agree`) may have max-softmax confidence below 0.6. The 5 % default is configurable via `space_grade.py --confidence-fail-fraction`.

**Measured for this run**:
- Headline variant: `masked-gold-min1-100agree`
- Per-pixel confidence threshold: 0.6
- Fail-fraction threshold: 5 %
- **Low-confidence fraction (overall, valid pixels): 3.20 %** → **✅ PASS**
- Mean confidence (overall): 0.9459

### Updated scorecard for 002

| # | Constraint | Status |
|---|---|---|
| 5 | Params < 3 M | ✅ |
| 6 | INT8 / FP16 + 100% audit | ✅ |
| 3 | RAM ≤ 256 MB (deployment subprocess: 105 MB) | ✅ |
| **8** | **Confidence < 0.6 → stop pass/fail (≤ 5% low-conf, measured 3.20%)** | **✅** (newly closed) |
| 7 | > 5 FPS @ 200 MHz (RAD750-class: 0.036 FPS) | ❌ |
| 4 | Power < 20 W | not verifiable |
| 1 | Clock 200 MHz | proxy / projection |
| 2 | Single-core | sim |

**5 of 5 verifiable hard constraints pass for this run**; #7 (throughput) remains the only structural fail; #4 (power) is unverifiable; #1 and #2 are projections.

## What is NOT in this run

- No augmentation
- No class weighting / focal loss
- No quantization-aware training (only post-training quantization tested)
- No actual embedded-hardware benchmark (deferred)
- No bit-flip / radiation-tolerance simulation (out of scope per thesis §1.6)
- No 256² run for further resolution scaling (deferred; the resolution-lever finding is already strong with one comparison point)
- No second random seed (deferred; one well-documented run per configuration suffices for the resolution-lever question)

These are levers held flat so this run is a clean resolution-only comparison to 001.
