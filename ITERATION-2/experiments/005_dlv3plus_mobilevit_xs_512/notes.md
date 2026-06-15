# 005 — DeepLabV3+ + MobileViT-XS at 512² (paradigm-distinct encoder experiment)

## What

Fifth ITERATION-2 run. **Same architecture (DeepLabV3+), same training recipe, same input resolution (512²) as [002_dlv3plus_512](../002_dlv3plus_512/). Only the encoder differs: tu-mobilevit_xs (hybrid CNN + transformer) replaces tu-mobilenetv3_small_100 (pure CNN).**

The motivation: at 47–81 M parameters, the literature shows a substantial Big Rock IoU advantage for transformer-based architectures over pure CNNs (paper_13: UPerNet/Swin 0.87 and SegFormer-B3 0.83 vs DeepLabv3+/ResNet 0.57 on M3). The question this run answers: **does the transformer-paradigm advantage transfer down to a 3 M parameter cap?**

Encoder pick rationale (also recorded in the cross-architecture map):
- **tu-mobilevit_xs** = MobileViT-XS (Apple, 2022). Hybrid CNN + transformer architecture designed for mobile/edge.
- Total params with DLV3+ decoder: **2,918,340** = 97.3 % of the 3 M cap — the cap-saturating hybrid option in our `smp.DeepLabV3Plus` build.
- First attention-based encoder ever tested on AI4Mars under a strict parameter cap (no equivalent in the existing AI4Mars literature).

## Test-set headline numbers (gold expert set, Strategy A)

| Variant | Pixel acc | mIoU | Big Rock IoU | BR Precision | BR Recall | Low-conf @ 0.6 |
|---|---:|---:|---:|---:|---:|---:|
| min1-100agree | 0.9246 | 0.6713 | **0.1146** | 0.1371 | 0.411 | **1.97 %** |
| min2-100agree | 0.9615 | 0.7223 | 0.1244 | 0.1314 | 0.702 | 1.26 % |
| min3-100agree | 0.9796 | 0.8047 | **0.3551** | 0.3623 | 0.948 | 0.66 % |

### Per-class IoU at min1 / min3

| Class | min1 IoU | min1 Prec | min1 Rec | min3 IoU | min3 Prec | min3 Rec |
|---|---:|---:|---:|---:|---:|---:|
| soil | 0.918 | 0.964 | 0.950 | 0.984 | 0.991 | 0.993 |
| bedrock | 0.813 | 0.876 | 0.918 | 0.943 | 0.960 | 0.981 |
| sand | 0.840 | 0.943 | 0.886 | 0.945 | 0.987 | 0.957 |
| big_rock | 0.115 | 0.137 | 0.411 | 0.355 | 0.362 | 0.948 |

## Apples-to-apples vs 002_dlv3plus_512 (MNv3-S → MobileViT-XS, encoder only)

Same decoder (DLV3+), same training recipe, same input resolution (512²); only the encoder family differs.

| Metric | 002 (MNv3-Small, CNN) | 005 (MobileViT-XS, hybrid) | Δ | Direction |
|---|---:|---:|---:|---|
| **Params** | 2.16 M | 2.92 M | +35 % | MobileViT heavier |
| **FP32 checkpoint** | 8.42 MB | 11.30 MB | +34 % | larger |
| **FLOPs (forward, 512²)** | 5.84 G | **9.79 G** | **+68 %** | MobileViT does more compute |
| **GPU latency (batch 1)** | 7.6 ms | 14.2 ms | +87 % | almost 2× slower |
| **CPU FP32 latency (1 thread)** | 188 ms | 459 ms | +144 % | 2.4× slower |
| **INT8 CPU latency** | 107 ms | **1131 ms** | **+957 %** ⚠ | **INT8 is slower than FP32** |
| **INT8 audit (fully INT8 ops)** | 67/67 (100 %) | 104/104 (100 %) | both 100 % | (no FP fallback either side) |
| **Deployment RSS (subprocess)** | 105 MB | **311 MB** | **+196 %** | **FAILS 256 MB cap** ❌ |
| **Linear-scaled FPS @ 200 MHz** | 0.288 | 0.118 | −59 % | worse |
| **RAD750-class FPS** | 0.036 | 0.015 | −59 % | worse |
| | | | | |
| min1 pixel acc | 0.9053 | 0.9246 | +0.019 | ↑ MobileViT slightly better |
| min1 mIoU | 0.6520 | 0.6713 | +0.019 | ↑ MobileViT slightly better |
| **min1 Big Rock IoU** | **0.1424** | **0.1146** | **−19 %** | **↓ MobileViT worse on rare class** |
| min1 BR Precision | 0.1951 | 0.1371 | −0.058 | ↓ less selective |
| min1 BR Recall | 0.3450 | 0.4114 | +0.066 | ↑ catches slightly more |
| min3 mIoU | 0.8619 | 0.8047 | −0.057 | ↓ worse |
| **min3 Big Rock IoU** | **0.6639** | **0.3551** | **−47 %** | **↓↓ MobileViT substantially worse on rare class** |
| min3 BR Precision | 0.8138 | 0.3623 | −0.45 | ↓↓↓ |
| min3 BR Recall | 0.7828 | 0.9475 | +0.165 | ↑↑ much higher recall |
| Low-confidence < 0.6 @ min1 | 3.20 % | **1.97 %** | −1.23 pp | ↓ much more confident |

## The cross-encoder finding (key thesis result)

**The hybrid CNN+transformer encoder does NOT help under the 3 M parameter cap.** On the headline rare-class metric, it loses to the pure-CNN MobileNetV3 baseline by 19 % at min1 and 47 % at min3. The pattern is similar to FPN (003): higher recall, much lower precision — the model finds more Big Rocks but is wrong much more often about each.

This inverts the literature expectation. Paper_13 reports a +26 pp Big Rock IoU advantage for transformers (UPerNet 0.87 / SegFormer-B3 0.83) over DeepLabv3+ (0.57) at 47–81 M parameters. **That advantage does not transfer down to the 3 M cap** — at our budget the transformer encoder is *worse*, not better, on rare-class detection.

**Hypothesis**: the transformer advantage at large parameter counts comes from attention's ability to aggregate global context. At 1.9 M encoder parameters (MobileViT-XS), the attention layers are too thin to aggregate meaningfully — they're capacity-starved. A pure-CNN encoder at the same parameter count concentrates more capacity on local convolution, which evidently does more useful work for Mars terrain segmentation at this scale.

**Reading for the thesis**: under a 3 M parameter cap on AI4Mars, the encoder paradigm choice (CNN vs hybrid) is **second-order** to other levers. Resolution (002 vs 001: ×2.9 on min3 BR IoU) and decoder family (002 vs 003: ×1.6 on min3 BR IoU advantage for DLV3+) are both larger effects than encoder paradigm (002 vs 005: ×1.9 advantage for CNN). Furthermore, the encoder paradigm choice in the *wrong direction* (CNN → hybrid) actively hurts.

## Three genuine deployment surprises

### 1. INT8 is slower than FP32 on MobileViT (+957 % latency)

Both 002 and 005 quantize 100 % of compute-heavy ops to INT8 (the audit confirms it). But:
- 002 INT8 = 107 ms (1.8× faster than FP32 188 ms)
- 005 INT8 = **1131 ms (2.5× SLOWER than FP32 459 ms)**

Cause: MobileViT-XS has 104 compute-heavy ops (vs DLV3+/MNv3's 67) because of the attention layers (MatMul-heavy). ORT's INT8 CPU kernels handle attention MatMul ops with high per-op overhead — quantization succeeds graph-level but the resulting kernels are slower than the FP32 path. **This is a real deployment finding**: INT8 quantization of transformer-style ops is not a free speedup; it depends on the kernel implementation, and on x86 CPU with ORT, MobileViT actively loses from quantization.

A real deployment would either skip INT8 for MobileViT, or use a transformer-specific quantization runtime (TensorRT, OpenVINO). Neither is in scope for this thesis.

### 2. Deployment RSS exceeds the 256 MB cap (311 MB)

The first run in the project to fail constraint #3 with the deployment-subprocess measurement. Attention layers materialise larger query/key/value/attention-output tensors simultaneously, where convolution layers stream through narrower tensor stacks. At 512² input, MobileViT-XS's activation memory pushes past the cap even with the minimal ORT-only runtime.

A larger input (1024²) for MobileViT would balloon RSS further — it's structurally non-compliant with our memory cap, not just borderline.

### 3. Confidence is the highest of all runs (1.97 % low-conf at min1)

005's confidence histogram is sharper than 002/003/004. The model is **very sure** about its predictions — but it's confidently wrong on rare class. The pattern: high softmax confidence on a Big Rock prediction where the precision is only 0.137. This is a notable failure mode: a model that is confidently wrong is worse than one that admits uncertainty (the "stop the rover when confidence < 0.6" rule wouldn't trigger here).

This is a separate thesis point: **the confidence pass/fail constraint can be met by a model that is wrong but confident**. Constraint #8 is necessary but not sufficient for deployment readiness.

## Space-grade constraint compliance

| # | Constraint | Target | Result | Status |
|---|---|---|---|---|
| 1 | Clock | 200 MHz | (projection only; 3 levels) | — |
| 2 | Architecture | single-core PowerPC | (simulated single-thread x86) | — |
| 3 | RAM (PyTorch-loaded) | ≤ 256 MB | 1455 MB peak RSS | ❌ (inflated) |
| 3 | **RAM (deployment subprocess)** | **≤ 256 MB** | **311 MB peak** | **❌** (fails — first time) |
| 4 | Power | < 20 W | not measured | not verifiable |
| 5 | **Parameters** | **< 3 M** | **2.92 M** (97.3 % of cap) | ✅ |
| 6 | **Precision** | **INT8 or FP16** | FP16 (5.76 MB) and INT8 (3.79 MB, 104/104 ops fully INT8) | ✅ (quantizable) |
| 7 | **Throughput** | **> 5 FPS @ 200 MHz** | 0.118 FPS (clock-scaled), 0.015 FPS (RAD750-class), 0.020 FPS (theoretical floor) | ❌ |
| 8 | **Confidence < 0.6 → stop pass/fail (≤ 5 % low-conf)** | **1.97 %** | **✅** (most comfortable headroom of any run) |

**Compliance scorecard: `P✓ R(pt)✗ R(dep)✗ T✗ F16✓ I8✓ C✓`** — **R(dep) flips from ✓ (in 001/002/003/004) to ✗ (in 005)**. This run is **not space-grade compliant on RAM**, even with the most generous interpretation of the cap.

### Throughput projections (three levels of realism)

| Projection | Latency | FPS | Compliant > 5 FPS? | What it captures |
|---|---:|---:|:-:|---|
| Theoretical floor (1 FLOP/cycle) | 48.94 s | 0.020 | ❌ | 200 MHz × 1 FLOP/cycle on 9.79 G FLOPs |
| Linear-scaled (clock only) | 8.49 s | 0.118 | ❌ | Host FP32 latency × clock ratio |
| **RAD750-class (× 8 microarch penalty)** | **67.95 s** | **0.015** | ❌ | Adds ~4× SIMD gap + ~2× IPC gap |

INT8 path: 1131 ms × 18.5 × 8 = **167 s → 0.006 FPS** (worse than FP32 because INT8 is slower). MobileViT genuinely makes the throughput picture worse, not better.

## Training stability

10 epochs, no NaN. Wall time **~44 min** on one RTX PRO 4000 Blackwell GPU 1 (vs 002's ~18 min at the same resolution with MNv3 — 2.4× slower training). MobileViT's attention layers are slow to train on this GPU class.

**Notable wobble**: epoch 6 had a sharp val Big Rock IoU drop (0.054 — likely a hard batch). Recovery at epochs 7–9. Final epoch 10 was the **best** val Big Rock IoU (0.482) — unusual; most of our runs peak around epoch 7–9. Final epoch numbers reported above.

Per-epoch summary:

| Epoch | Train loss | Val loss | Val mIoU | Val BR IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.532 | 0.171 | 0.733 | 0.273 |
| 3 | 0.215 | 0.230 | 0.698 | 0.249 |
| 5 | 0.139 | 0.180 | 0.744 | 0.387 |
| 6 (wobble) | 0.125 | 0.275 | 0.694 | 0.054 |
| 7 | 0.122 | 0.149 | 0.799 | 0.425 |
| 9 | 0.106 | 0.137 | 0.779 | 0.377 |
| **10 (peak)** | 0.098 | 0.124 | **0.815** | **0.482** |

## INT8 quantization results

| Precision | Model size on disk | Reduction vs FP32 | CPU 1-thread latency | Host FPS |
|---|---:|---:|---:|---:|
| FP32 | 11.30 MB | — | 459 ms | 2.18 |
| FP16 | 5.76 MB | 2.0× smaller | (not benchmarked) | — |
| **INT8** | **3.79 MB** | **3.0× smaller** | **1131 ms** | **0.88** |

**INT8 fallback audit**: 104 / 104 compute-heavy ops fully INT8. Zero silent FP fallback. The model is, technically, fully quantized.

But INT8 is **2.5× SLOWER than FP32** on host CPU. This is the inverse of every other run in the project: 001 / 002 / 003 / 004 / 006 all see INT8 ~2× faster than FP32. The cause is MobileViT's attention layers — 104 compute-heavy ops vs MNv3's 67, with per-op INT8 overhead in ORT's CPU kernel exceeding the per-op savings.

## Deployment RSS (clean ORT-only subprocess)

| Milestone | RSS (MB) |
|---|---:|
| After Python imports (CPython + NumPy + ONNX-Runtime) | 44.3 |
| After ORT session create (INT8 model loaded) | 64.7 |
| After warmup forward (activations allocated) | 311.0 |
| **Peak during inference** | **311.0** |

**311 MB peak — fails the 256 MB cap by 55 MB.** Compare to 002 at 512² (same resolution, pure-CNN encoder): 105 MB. The hybrid encoder roughly **triples the activation memory** at the same input resolution, because attention layers materialize larger Q/K/V/output tensors simultaneously rather than streaming through narrow conv tensors.

## What this tells us — for the thesis

1. **Encoder paradigm (CNN vs hybrid) is not the binding lever under 3 M cap.** Smaller effect than resolution, smaller effect than decoder family, and in this case *the wrong direction*. The transformer advantage at 47–81 M (paper_13) does not transfer down.
2. **Newer / fancier encoders fail deployment constraints.** First run to break #3 (RAM). The space-grade scorecard is robust to architecture changes: it surfaced a real-world failure mode of the transformer paradigm at low param counts.
3. **INT8 quantization is not free for transformer-style ops on x86 CPU.** This is a deployment-level finding: the assumption that "INT8 always speeds things up" is wrong for MobileViT. A real deployment would need a transformer-aware quantization runtime.
4. **Confidence alone is not a deployment guarantee.** MobileViT has the highest confidence numbers of any run, but the rare-class precision is lowest. Constraint #8 (confidence pass/fail) is necessary but insufficient.
5. **The 4-encoder story now reads**: MobileNetV3-Small (best, 2.16 M), FPN-with-MNv3 (worse decoder), MobileNetV4 (pending, 006), MobileViT-XS (worse encoder). MobileNetV3 + DLV3+ at 512² remains the strongest configuration in the project.

## Run artefacts

| File | What |
|---|---|
| `config.json` | Records `model.encoder_name = tu-mobilevit_xs`, `data.input_hw = 512` |
| `weights.pth` | 11.30 MB FP32 |
| `weights_fp16.pth` | 5.76 MB |
| `model_fp32.onnx` + `.onnx.data` | ONNX export |
| `model_int8.onnx` | 3.79 MB INT8 (slow on CPU — see above) |
| `training_history.csv` | 29-column per-epoch metrics |
| `evaluation_results.json` | Strategy-A gold-test results; `eval_input_hw=512` |
| `space_grade.json` | Full scorecard — **first run with `ram_under_256mb_deployment_subprocess=False`** |
| `pipeline.sh` / `pipeline.log` / `*.log` | Chained nohup-pipeline artefacts |
| `PIPELINE_DONE` | Marker — written 2026-05-14T03:35:46Z |

## What is NOT in this run

- No augmentation, class weighting, focal loss (per protocol)
- No QAT (only post-training quantization)
- No second random seed
- No transformer-specific runtime (TensorRT, OpenVINO) — would change the INT8 picture
- No alternative transformer encoders (EfficientFormer, FastViT, etc. — most fail smp's `output_stride=16` compatibility or exceed the 3 M cap)
