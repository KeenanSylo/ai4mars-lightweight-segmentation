# 006 — DeepLabV3+ + MobileNetV4-conv-small at 512² (newer-CNN modernity test)

## What

Sixth ITERATION-2 run. **Same architecture (DeepLabV3+), same training recipe, same input resolution (512²) as [002_dlv3plus_512](../002_dlv3plus_512/). Only the encoder differs: tu-mobilenetv4_conv_small (2024 mobile CNN) replaces tu-mobilenetv3_small_100 (2019 mobile CNN).**

The motivation: testing the "newer encoder = better" intuition. MobileNetV4 (Google, 2024) is the direct CNN successor to MobileNetV3 (2019), with the same lineage and design team but a 5-year update window — hardware-aware NAS over a newer search space (Universal Inverted Bottleneck blocks, deployment targets for newer NPUs). The question this run answers: **does the newer CNN actually beat the older CNN at the same task and same training recipe under the 3 M cap?**

Encoder pick rationale:
- **tu-mobilenetv4_conv_small** = MobileNetV4 conv_small variant. Pure-CNN (no attention).
- Total params with DLV3+ decoder: **2,999,220** = **100.0 % of the 3 M cap** — the most cap-saturating pure-CNN option in our `smp.DeepLabV3Plus` build.
- Newest mobile-CNN encoder never tested on AI4Mars.

## Test-set headline numbers (gold expert set, Strategy A)

| Variant | Pixel acc | mIoU | Big Rock IoU | BR Precision | BR Recall | Low-conf @ 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| min1-100agree | 0.9117 | 0.6564 | **0.1109** | 0.1348 | 0.385 | 4.22 % |
| min2-100agree | 0.9508 | 0.7092 | 0.1245 | 0.1318 | 0.694 | 3.23 % |
| min3-100agree | 0.9713 | 0.8121 | **0.4264** | 0.4371 | 0.946 | 2.23 % |

### Per-class IoU at min1 / min3

| Class | min1 IoU | min1 Prec | min1 Rec | min3 IoU | min3 Prec | min3 Rec |
|---|---:|---:|---:|---:|---:|---:|
| soil | 0.911 | 0.964 | 0.943 | 0.974 | 0.991 | 0.983 |
| bedrock | 0.794 | 0.846 | 0.926 | 0.929 | 0.946 | 0.980 |
| sand | 0.811 | 0.926 | 0.869 | 0.928 | 0.974 | 0.952 |
| big_rock | 0.111 | 0.135 | 0.385 | 0.426 | 0.437 | 0.946 |

## Apples-to-apples vs 002_dlv3plus_512 (MNv3-S → MNv4-cs, encoder only — pure CNN both)

Same decoder (DLV3+), same training recipe, same input resolution (512²); only the encoder differs (both are pure CNNs, same lineage, 5-year update window).

| Metric | 002 (MNv3-Small, 2019) | 006 (MNv4-conv-small, 2024) | Δ | Direction |
|---|---:|---:|---:|---|
| **Params** | 2.16 M | 3.00 M | +39 % | MNv4 heavier |
| **FP32 checkpoint** | 8.42 MB | 11.62 MB | +38 % | larger |
| **FLOPs (forward, 512²)** | 5.84 G | 8.86 G | +52 % | MNv4 does more compute |
| **GPU latency (batch 1)** | 7.6 ms | 7.6 ms | ≈ same | (GPU underutilised at this size) |
| **CPU FP32 latency (1 thread)** | 188 ms | 229 ms | +22 % | slightly slower |
| **INT8 CPU latency** | 107 ms | 113 ms | +6 % | comparable; INT8 path works |
| **INT8 audit (fully INT8 ops)** | 67/67 (100 %) | 60/60 (100 %) | (fewer ops — MNv4 has different structure) | both fully quantized |
| **Deployment RSS (subprocess)** | 105 MB | **89 MB** | **−15 %** | ↓ MNv4 lighter on RAM (best 512² result so far) |
| **Linear-scaled FPS @ 200 MHz** | 0.288 | 0.236 | −18 % | worse |
| **RAD750-class FPS** | 0.036 | 0.030 | −18 % | worse |
| | | | | |
| min1 pixel acc | 0.9053 | 0.9117 | +0.006 | ↑ MNv4 slightly better |
| min1 mIoU | 0.6520 | 0.6564 | +0.004 | ↑ MNv4 slightly better |
| **min1 Big Rock IoU** | **0.1424** | **0.1109** | **−22 %** | **↓ MNv4 worse on rare class** |
| min1 BR Precision | 0.1951 | 0.1348 | −0.060 | ↓ less selective |
| min1 BR Recall | 0.3450 | 0.3845 | +0.040 | ↑ catches slightly more |
| min3 mIoU | 0.8619 | 0.8121 | −0.050 | ↓ worse |
| **min3 Big Rock IoU** | **0.6639** | **0.4264** | **−36 %** | **↓↓ MNv4 substantially worse on rare class** |
| min3 BR Precision | 0.8138 | 0.4371 | −0.377 | ↓↓↓ much less selective |
| min3 BR Recall | 0.7828 | 0.9456 | +0.163 | ↑↑ much higher recall |
| Low-confidence < 0.6 @ min1 | 3.20 % | 4.22 % | +1.02 pp | ↓ less confident |

## The "newer ≠ better" finding (key thesis result, sister to 005)

**MobileNetV4-conv-small does not beat MobileNetV3-Small on AI4Mars under the 3 M parameter cap.** Same pattern as MobileViT (005) but milder:

- **min1 Big Rock IoU**: 0.142 (MNv3) vs 0.111 (MNv4) → MNv4 is **22 % worse**.
- **min3 Big Rock IoU**: 0.664 (MNv3) vs 0.426 (MNv4) → MNv4 is **36 % worse**.
- Same precision-vs-recall trade as 003 and 005: MNv4 has higher recall but much lower precision on rare class.
- Global metrics (pixel acc, mIoU) are marginally better with MNv4 (+0.006, +0.004) — but the gain is in the noise.

The MNv3 → MNv4 jump (2019 → 2024, same design team, same lineage) was *supposed* to be a strict upgrade — and on ImageNet classification it is (~1–3 pp top-1 improvement at matched param count). But on **AI4Mars semantic segmentation under a 3 M cap**, **the newer encoder is worse on the headline rare-class metric**.

**Hypothesis** (consistent with 005's interpretation): MobileNetV4's hardware-aware NAS targeted ImageNet-style classification and modern phone NPUs. Its inductive biases — Universal Inverted Bottleneck (UIB) blocks, designed for fast end-to-end classifier throughput — don't necessarily produce *better feature pyramids for semantic segmentation*. Specifically, MNv3's squeeze-excite + bottleneck stack apparently produces more *selective* features for rare-class detection than MNv4's UIB blocks at the same parameter class. Newer-on-ImageNet ≠ better-on-AI4Mars-segmentation.

**Reading for the thesis**: combined with 005 (hybrid encoder is worse) and 003 (different decoder is worse), the cross-architecture verdict is:

> **The 2.16 M MobileNetV3-Small + DeepLabV3+ configuration (002) is the strongest under 3 M cap. Newer encoders (MNv4), different-paradigm encoders (MobileViT), and different decoders (FPN) all underperform on rare-class IoU at the same training recipe.**

This is the kind of negative-but-defensible cross-architecture finding the thesis was made for: ruling out plausible upgrades empirically rather than hand-waving them away.

## Space-grade constraint compliance

| # | Constraint | Target | Result | Status |
|---|---|---|---|---|
| 1 | Clock | 200 MHz | (projection only; 3 levels) | — |
| 2 | Architecture | single-core PowerPC | (simulated single-thread x86) | — |
| 3 | RAM (PyTorch-loaded) | ≤ 256 MB | ~1.4 GB peak RSS | ❌ (inflated) |
| 3 | **RAM (deployment subprocess)** | **≤ 256 MB** | **89.3 MB peak** (167 MB headroom — best of any 512² run) | **✅** |
| 4 | Power | < 20 W | not measured | not verifiable |
| 5 | **Parameters** | **< 3 M** | **2,999,220 = 99.97 %** of cap | ✅ (closest to cap of any run) |
| 6 | **Precision** | **INT8 or FP16** | FP16 (5.86 MB) and INT8 (3.40 MB, 60/60 ops fully INT8) | ✅ |
| 7 | **Throughput** | **> 5 FPS @ 200 MHz** | 0.236 FPS (clock-scaled), 0.030 FPS (RAD750-class), 0.014 FPS (theoretical floor) | ❌ |
| 8 | **Confidence < 0.6 → stop pass/fail (≤ 5 % low-conf)** | **4.22 %** | **✅** (78 bp headroom under cap — same magnitude as 004's 23 bp) |

**Compliance scorecard: `P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓`** — same letter score as 001/002/004 and DLV3+/FPN/MNv3 cousins. **Five of five verifiable hard constraints pass. Only throughput fails.**

### Throughput projections (three levels of realism)

| Projection | Latency | FPS | Compliant > 5 FPS? | What it captures |
|---|---:|---:|:-:|---|
| Theoretical floor (1 FLOP/cycle) | 44.29 s | 0.023 | ❌ | 200 MHz × 1 FLOP/cycle on 8.86 G FLOPs |
| Linear-scaled (clock only) | 4.24 s | 0.236 | ❌ | Host FP32 latency × clock ratio |
| **RAD750-class (× 8 microarch penalty)** | **33.92 s** | **0.030** | ❌ | Adds ~4× SIMD gap + ~2× IPC gap |

INT8 path, RAD750-class projection: 113 ms × 18.5 × 8 = **16.7 s → 0.060 FPS**. Same ballpark as 002 — INT8 + RAD750-class is still ~80× short of 5 FPS.

## Training stability

10 epochs, no NaN. Wall time **~37 min** on one RTX PRO 4000 Blackwell GPU 0 (vs 002's ~18 min — 2× slower training because MNv4 has more FLOPs and parameters).

Val accuracy progression: notable **slow start** (val BR IoU 0.015 at epoch 1 — much lower than 002's 0.082) but **strong finish** (val BR IoU 0.409 at epoch 10, the best of any epoch). MNv4 converges more slowly but to a similar val-set peak as 002 — though as the test-set numbers show, that val-set peak doesn't transfer to better generalisation on rare class.

Per-epoch summary:

| Epoch | Train loss | Val loss | Val mIoU | Val BR IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.673 | 0.366 | 0.500 | 0.015 |
| 3 | 0.292 | 0.255 | 0.712 | 0.274 |
| 5 | 0.205 | 0.232 | 0.711 | 0.234 |
| 7 | 0.166 | 0.171 | 0.767 | 0.349 |
| 8 | 0.155 | 0.156 | 0.770 | 0.382 |
| 9 | 0.137 | 0.232 | 0.702 | 0.349 |
| **10 (peak)** | 0.131 | 0.169 | **0.777** | **0.409** |

## INT8 quantization results

| Precision | Model size on disk | Reduction vs FP32 | CPU 1-thread latency | Host FPS |
|---|---:|---:|---:|---:|
| FP32 | 11.62 MB | — | 229 ms | 4.37 |
| FP16 | 5.86 MB | 2.0× smaller | (not benchmarked) | — |
| **INT8** | **3.40 MB** | **3.42× smaller** | **113 ms** | **8.85** |

**INT8 fallback audit**: 60 / 60 compute-heavy ops fully INT8. Zero silent FP fallback. (Note: fewer ops than 002's 67 — MNv4's UIB blocks have different internal structure.)

INT8 is **2.0× faster on host CPU than FP32** — same ratio as 002. **MNv4 does not suffer the INT8 slowdown that MobileViT did**; pure-CNN architecture quantizes cleanly on ORT.

## Deployment RSS (clean ORT-only subprocess)

| Milestone | RSS (MB) |
|---|---:|
| After Python imports (CPython + NumPy + ONNX-Runtime) | 44.3 |
| After ORT session create (INT8 model loaded) | 64.9 |
| After warmup forward (activations allocated) | 89.3 |
| **Peak during inference** | **89.3** |

**89 MB peak — best deployment RSS of any 512² run** (vs 002's 105 MB and 005's 311 MB). MNv4's narrower activation tensors stream through more efficiently than MobileViT's attention layers. The 167 MB headroom under the cap is the largest of any 512² run.

## What this tells us — for the thesis

1. **Newer ≠ better at this budget on this task.** MobileNetV4 (2024) loses to MobileNetV3 (2019) on rare-class IoU by 22 % at min1 and 36 % at min3, despite using 39 % more parameters and 52 % more FLOPs. The MNv3→MNv4 jump is primarily a *hardware-aware redesign for newer mobile NPUs*, not a *strictly better representation for semantic segmentation*.
2. **MNv4 does pass all the deployment constraints that MobileViT failed.** INT8 quantization gives the expected ~2× speedup (not slow-down). Deploy RSS is 89 MB (well under cap). Same pure-CNN compliance pattern as 002.
3. **The cross-encoder verdict is now complete.** Three encoders tested at the same decoder, resolution, recipe:
   - **MobileNetV3-Small** (002, 2019 CNN): **best** (min1 BR IoU 0.142, min3 0.664).
   - MobileNetV4-conv-small (006, 2024 CNN): worse (-22 % / -36 %).
   - MobileViT-XS (005, 2022 hybrid): worst (-19 % / -47 %).
4. **At the same training recipe, the dominant predictor of rare-class IoU at 3 M cap is NOT the encoder's architectural modernity** — it's the resolution and decoder choice (per 002 vs 001 and 002 vs 003). Encoder choice is **third-order**.
5. **MNv3-Small at 2.16 M (72 % of cap) remains the strongest configuration** — being further under the cap doesn't hurt. *The cap itself isn't binding for this dataset; the binding constraints are throughput and (at low resolutions) confidence.*

## Run artefacts

| File | What |
|---|---|
| `config.json` | Records `model.encoder_name = tu-mobilenetv4_conv_small`, `data.input_hw = 512` |
| `weights.pth` | 11.62 MB FP32 |
| `weights_fp16.pth` | 5.86 MB |
| `model_fp32.onnx` + `.onnx.data` | ONNX export |
| `model_int8.onnx` | 3.40 MB INT8 quantized |
| `training_history.csv` | 29-column per-epoch metrics |
| `evaluation_results.json` | Strategy-A gold-test results; `eval_input_hw=512` |
| `space_grade.json` | Full scorecard including new constraint #8 binary pass/fail |
| `pipeline.sh` / `pipeline.log` / `*.log` | Chained nohup-pipeline artefacts |
| `PIPELINE_DONE` | Marker — written 2026-05-14T03:56:31Z |

## What is NOT in this run

- No augmentation, class weighting, focal loss (per protocol)
- No QAT (only post-training quantization)
- No second random seed
- No MNv4 *hybrid* variants (which include attention) — those would likely follow the MobileViT pattern (deployment failures)
- No 1024² or 256² runs of MNv4 — the cross-encoder claim is established at 512²; cross-resolution claim was already established with the MNv3 baseline (001/002/004)
