# ITERATION-2 — Experiment log

Cross-architecture comparison under space-grade constraints. Same training recipe across runs (plain CE, no augmentation, no class weights, 10 epochs, batch 8, Adam lr 1e-3 — see [docs/protocol.md](../../docs/protocol.md)) — only the architecture changes, so any differences are attributable to the model.

**Constraint targets** (from [ITERATION-2/notes.md](../notes.md)): params < 3 M, RAM ≤ 256 MB, > 5 FPS @ 200 MHz, INT8 or FP16, power < 20 W (unverifiable), confidence < 0.6 ⇒ stop the rover (metric-only).

## Runs

| ID | Architecture | Encoder | Params | GPU lat. (ms) | INT8 size (MB) | INT8 lat. (ms) | INT8 audit² | RAD750-class FPS³ | Deploy RSS (MB)⁴ | min1 BR IoU | min3 BR IoU | min1 mIoU | min3 mIoU | Compliance¹ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| [001_dlv3plus_baseline](001_dlv3plus_baseline/) | DeepLabV3+ | tu-mobilenetv3_small_100 | 2.16 M | 9.0 | 2.69 | 545 | 67/67 (100%) | 0.0062 | 242 | 0.0872 | 0.2260 | 0.6509 | 0.7613 | P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓ |
| [002_dlv3plus_512](002_dlv3plus_512/) | DeepLabV3+ | tu-mobilenetv3_small_100 | 2.16 M | 7.6 | 2.69 | 107 | 67/67 (100%) | 0.0359 | 105 | 0.1424 | 0.6639 | 0.6520 | 0.8619 | P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓ |
| [003_fpn_512](003_fpn_512/) | FPN | tu-mobilenetv3_small_100 | 2.72 M | 8.0 | 3.33 | 224 | 64/64 (100%) | 0.0255 | 235 | 0.1076 | 0.4127 | 0.6625 | 0.8131 | P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓ |
| [004_dlv3plus_256](004_dlv3plus_256/) | DeepLabV3+ | tu-mobilenetv3_small_100 | 2.16 M | 7.6 | 2.69 | 30 | 67/67 (100%) | 0.1076 | 72 | 0.1312 | 0.5051 | 0.6219 | 0.7924 | P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓ |
| [005_dlv3plus_mobilevit_xs_512](005_dlv3plus_mobilevit_xs_512/) | DeepLabV3+ | tu-mobilevit_xs | 2.92 M | 14.2 | 3.79 | 1131 ⚠ | 104/104 (100%) | 0.0147 | **311 ❌** | 0.1146 | 0.3551 | 0.6713 | 0.8047 | P✓ R(pt)✗ **R(dep)✗** T✗ F16✓ I8✓ C✓ |
| [006_dlv3plus_mnv4_conv_small_512](006_dlv3plus_mnv4_conv_small_512/) | DeepLabV3+ | tu-mobilenetv4_conv_small | 3.00 M | 7.6 | 3.40 | 113 | 60/60 (100%) | 0.0295 | 89 | 0.1109 | 0.4264 | 0.6564 | 0.8121 | P✓ R(pt)✗ R(dep)✓ T✗ F16✓ I8✓ C✓ |

¹ **Compliance shorthand** (compact summary of `space_grade.json` constraint_summary):
- **P** = parameter count < 3 M
- **R(pt)** = peak inference RSS < 256 MB, measured in the PyTorch-loaded main process (inflated by ~10× from torch/CUDA overhead — preserved for documentation, see individual notes.md)
- **R(dep)** = peak inference RSS < 256 MB, measured in a clean ORT-only subprocess (deployment-footprint approximation — see footnote 4 and `space_grade_rss_subprocess.py`). This is the number to cite for space-grade compatibility.
- **T** = projected throughput > 5 FPS at 200 MHz (linear-scaled from host CPU; also tracked alongside the stricter RAD750-class projection — see footnote 3)
- **F16** = FP16 model save succeeded
- **I8** = INT8 ONNX quantization succeeded (via ONNX-Runtime static PTQ)
- **C** = confidence constraint #8 pass/fail (added 2026-05-14, see footnote 5)
- (`?` = not measured / not yet implemented)

Power is not in the compliance flag — it's unverifiable (no hardware power meter available).

> **Update 2026-05-14**: confidence (#8) is now a binary pass/fail and gets the `C` flag. See footnote 5.

² **INT8 audit** = fraction of Conv/ConvTranspose/Gemm/MatMul nodes whose data input AND weight input both come from `DequantizeLinear` in the QDQ-format ONNX graph. 100% means no silent FP fallback after quantization. See [space_grade.py](../../space_grade.py) `audit_int8_fallback()`.

³ **RAD750-class FPS** = `space_grade.json → projections_200mhz.rad750_class_fps`. Computed as (host FP32 CPU latency) × (host clock / 200 MHz) × (microarch penalty, default 8×). The 8× penalty decomposes as ~4× SIMD-gap (host AVX vs. RAD750 scalar FPU) × ~2× IPC-gap (OoO ~4-wide vs. in-order ~2-wide). This is the most honest projection toward actual flight-class hardware in this scorecard; cycle-accurate RAD750 simulation is out of scope. Configurable via `space_grade.py --microarch-penalty <N>`.

⁴ **Deploy RSS (MB)** = `space_grade.json → rss_clean_subprocess.rss_peak_mb`. Peak resident-set size measured in a clean Python subprocess that imports only ONNX-Runtime + numpy + psutil — no PyTorch, no CUDA, no segmentation-models-pytorch. Approximates a realistic embedded deployment footprint at the run's training resolution. See [space_grade_rss_subprocess.py](../../space_grade_rss_subprocess.py) and the "Re-run 2026-05-14 — Subprocess RSS measurement" section in the individual notes.md for the full breakdown.

⁵ **C — Confidence pass/fail** (constraint #8, added 2026-05-14): the runtime rule is *"if max-softmax confidence < 0.6, stop the rover"*. As a deployment-readiness binary, the scorecard reads `space_grade.json → confidence_constraint`, which is sourced from `evaluation_results.json`'s `confidence.overall_low_confidence_fraction` at the headline gold variant (`masked-gold-min1-100agree`). **Pass = ≤ 5%** of valid pixels are low-confidence (default; configurable via `space_grade.py --confidence-fail-fraction`). Larger fractions mean the rover would stop too often to be deployable. Measured: 001 = 2.97%, 002 = 3.20% — both pass with comfortable headroom. The per-pixel confidence threshold (0.6) and the pass/fail fraction (5%) are reported in `space_grade.json` for transparency.

## Cross-experiment observation (added 2026-05-14, post-002)

Same architecture (DeepLabV3+ + tu-mobilenetv3_small_100, 2.16 M params), same training recipe; only input resolution differs:

| | 001 @ 1024² | 002 @ 512² | Δ |
|---|---:|---:|---:|
| Big Rock IoU (min1 — headline) | 0.0872 | **0.1424** | **×1.6** |
| Big Rock IoU (min3) | 0.2260 | **0.6639** | **×2.9** |
| min1 BR precision | 0.100 | 0.195 | ×2.0 |
| min3 BR precision | 0.229 | 0.814 | ×3.6 |
| min1 mIoU | 0.6509 | 0.6520 | flat |
| min3 mIoU | 0.7613 | 0.8619 | +0.10 |
| CPU FP32 latency (ms, 1 thread) | 1097 | 188 | ÷5.8 |
| INT8 CPU latency (ms) | 545 | 107 | ÷5.1 |
| RAD750-class FPS | 0.0062 | 0.0359 | ×5.8 |
| Deploy RSS (MB) | 242 | 105 | ÷2.3 |

**Reading**: at a 3 M parameter budget, dropping the input from 1024² to 512² substantially improves rare-class IoU (most strongly on the strictest gold variant, min3) **and** improves throughput / RAM at the same time. The small model appears to over-predict Big Rock at 1024² (very low precision, modest recall); at 512² the effective receptive field grows relative to objects and the model becomes much more selective. **Resolution, not parameter count, was the binding constraint on accuracy in this configuration.** Throughput at 200 MHz remains structurally infeasible even at 512² + INT8 (~0.06 FPS RAD750-class, still ~80× short of 5 FPS) — empirical evidence that real-time onboard segmentation cannot be reached by resolution alone at this model class.

### Cross-decoder observation (added 2026-05-14, post-003)

Holding the encoder fixed (`tu-mobilenetv3_small_100`), training recipe, and input resolution (512²) constant, only the decoder family changes:

| | 002 @ 512² (DeepLabV3+) | 003 @ 512² (FPN) | Δ |
|---|---:|---:|---:|
| Params | 2.16 M | 2.72 M | +26 % (FPN) |
| FLOPs at 512² | 5.84 G | 16.44 G | **+2.81× (FPN)** |
| INT8 CPU latency (ms) | 107 | 224 | +109 % (FPN slower) |
| Deploy RSS (MB) | 105 | 235 | +124 % (FPN heavier; 21 MB headroom) |
| RAD750-class FPS | 0.0359 | 0.0255 | −29 % (FPN worse) |
| min1 Big Rock IoU | **0.1424** | 0.1076 | −24 % (FPN worse) |
| min3 Big Rock IoU | **0.6639** | 0.4127 | −38 % (FPN worse) |
| min1 mIoU | 0.6520 | 0.6625 | +0.011 (FPN slightly better) |
| min1 pixel acc | 0.9053 | 0.9188 | +0.014 (FPN slightly better) |

**Reading**: under a 3 M parameter cap and 512² input, **decoder family matters more than parameter count**. FPN has more parameters and ~3× more FLOPs than DLV3+, but is materially worse on the headline rare-class metric (−24 % min1 BR IoU, −38 % min3 BR IoU) — while being slightly better on global metrics (pixel accuracy, min1 mIoU). DLV3+'s ASPP-based decoder concentrates capacity on multi-rate dilation at a single scale; FPN spreads capacity across the feature pyramid. For Big Rock detection at this scale, the concentrated ASPP wins. **Bigger ≠ better when the budget is binding.**

The cross-resolution effect (1024² → 512², ×2.9 on min3 BR IoU) is also substantially larger than the cross-decoder effect (DLV3+ → FPN, ×0.62 on min3 BR IoU). Resolution remains the dominant lever in this regime.

### Resolution-curve completion (added 2026-05-14, post-004)

Same model (DeepLabV3+ + tu-mobilenetv3_small_100, 2.16 M params), same training recipe; only input resolution varies — 3 points form the curve.

| | 001 @ 1024² | 002 @ 512² | 004 @ 256² |
|---|---:|---:|---:|
| FLOPs | 23.4 G | 5.84 G | 1.46 G |
| INT8 CPU latency (ms) | 545 | 107 | 30 |
| Deploy RSS (MB) | 242 | 105 | 72 |
| RAD750-class FPS | 0.0062 | 0.0359 | 0.1076 |
| Low-conf < 0.6 @ min1 | 2.97 % | 3.20 % | **4.77 %** |
| **min1 Big Rock IoU** | 0.0872 | **0.1424** ★ | 0.1312 |
| **min3 Big Rock IoU** | 0.2260 | **0.6639** ★ | 0.5051 |
| min1 BR Precision | 0.100 | 0.195 | 0.173 |
| min3 BR Precision | 0.229 | **0.814** ★ | 0.592 |

**Reading**: the resolution curve is **inverted-U with the peak at 512²**, not monotone. Two opposing forces compete:

1. **Receptive field grows relative to objects as resolution drops** → model becomes more selective, precision climbs, IoU rises. Dominant from 1024² → 512².
2. **Spatial detail per object shrinks** → small Big Rock objects (originally 30–80 px at 1024²) become 7.5–20 px at 256²; some fall below the model's discrimination floor. Dominant from 512² → 256².

The balance tips between 512 and 256. **min3 Big Rock IoU peak is sharp (0.226 → 0.664 → 0.505)**; min1 is gentler (0.087 → 0.142 → 0.131).

Secondary findings from this curve:

- **Throughput verdict is robust**: even at 256² + INT8 + RAD750-class, FPS = ~0.23 — still ~22× short of 5 FPS. **Resolution + INT8 together cannot close the gap.**
- **RAM stops being interesting at low resolutions**: 72 MB at 256² has 184 MB headroom under the 256 MB cap.
- **Confidence becomes the binding constraint at low resolutions**: low-conf fraction climbs 2.97 % → 3.20 % → 4.77 % as resolution drops. At 256² it just barely passes the 5 % threshold. **Going below 256² would likely fail constraint #8** — confidence and resolution are coupled, and the model becomes less certain about its predictions as input shrinks.

Thesis claim now backed by 3 data points: *"under a 3 M parameter cap, the rare-class IoU is dominated by input resolution, with a sweet spot near 512². Throughput at 200 MHz remains structurally infeasible at every viable resolution."*

### Cross-encoder observation (added 2026-05-14, post-005)

Holding the decoder fixed (DeepLabV3+), training recipe, and input resolution (512²) constant, only the encoder paradigm changes:

| | 002 @ 512² (MNv3-S, CNN) | 005 @ 512² (MobileViT-XS, hybrid CNN+transformer) | Δ |
|---|---:|---:|---:|
| Params | 2.16 M | 2.92 M | +35 % (MobileViT) |
| FLOPs at 512² | 5.84 G | 9.79 G | +68 % |
| INT8 CPU latency (ms) | 107 | **1131** ⚠ | **+957 % — INT8 SLOWER than FP32** |
| Deploy RSS (MB) | 105 | **311** ❌ | **+196 % — fails 256 MB cap** |
| RAD750-class FPS | 0.036 | 0.015 | −59 % |
| min1 mIoU | 0.6520 | 0.6713 | +0.019 (slight win for MobileViT) |
| min1 pixel acc | 0.9053 | 0.9246 | +0.019 (slight win) |
| **min1 Big Rock IoU** | **0.1424** | 0.1146 | **−19 % (MobileViT worse on rare class)** |
| **min3 Big Rock IoU** | **0.6639** | 0.3551 | **−47 % (MobileViT substantially worse on rare class)** |
| Low-conf < 0.6 @ min1 | 3.20 % | **1.97 %** | MobileViT *more confident* — but wrong |

**Reading**: under a 3 M parameter cap on AI4Mars, **the transformer-paradigm advantage seen in the literature at 47–81 M (paper_13's UPerNet 0.87, SegFormer-B3 0.83 vs CNN baseline 0.57)** *does not transfer down to the 3 M cap*. At this budget, MobileViT-XS is worse on rare-class IoU, fails the deployment RAM constraint, sees INT8 *slow down* not speed up (transformer attention ops are slow under ORT's INT8 CPU kernels), and produces high-confidence but low-precision rare-class predictions.

Three genuine deployment surprises from 005:

1. **INT8 is slower than FP32** (1131 vs 459 ms) — first run where quantization is a net loss. MobileViT's 104 compute-heavy ops include attention MatMuls that ORT's INT8 kernel handles with high per-op overhead. The audit still confirms 100 % of ops are quantized; the cost just exceeds the saving.
2. **Deployment RSS fails the 256 MB cap** (311 MB) — first run to do so. Attention layers materialize larger simultaneous Q/K/V/output tensors than convolution layers, blowing the activation budget at the same input resolution.
3. **Confidence alone is not a deployment guarantee** — 005 has the lowest low-confidence fraction of any run (1.97 %, vs 002's 3.20 %) but the worst rare-class precision. *A confident model that is wrong is worse than an honest model that admits uncertainty.* This is an important caveat on constraint #8 — necessary but not sufficient.

Combined with the cross-resolution and cross-decoder findings, the cross-encoder finding completes the lever-ordering for this regime:

**Resolution lever (002 vs 001)** > **Decoder lever (002 vs 003)** > **Encoder-paradigm lever (002 vs 005)**

…with the encoder-paradigm lever pulled in the wrong direction by adding attention. The thesis claim is now: *"under a 3 M cap on AI4Mars, the binding accuracy lever is input resolution, with a sweet spot near 512². Decoder family is second-order; encoder paradigm (CNN vs hybrid) is third-order and can actively hurt rare-class detection if attention layers are introduced at this scale."*

### Cross-encoder observation extended (added 2026-05-14, post-006)

Same decoder (DLV3+), same training recipe, same input resolution (512²). **Three encoders now compared at the same baseline:**

| Encoder | Year | Type | Params | Cap % | min1 BR IoU | min3 BR IoU | Deploy RSS | INT8 vs FP32 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| **MobileNetV3-Small** (002) | 2019 | pure CNN | 2.16 M | 72 % | **0.142** ★ | **0.664** ★ | 105 MB ✓ | 1.8× faster |
| MobileNetV4-conv-small (006) | 2024 | pure CNN | 3.00 M | 100 % | 0.111 ↓ | 0.426 ↓ | 89 MB ✓ | 2.0× faster |
| MobileViT-XS (005) | 2022 | hybrid CNN+transformer | 2.92 M | 97 % | 0.115 ↓ | 0.355 ↓ | **311 MB ❌** | **2.5× slower** |

**Reading**: across three encoders spanning two paradigms (pure CNN, hybrid) and a five-year design window (2019, 2022, 2024):

1. **Older simpler encoder wins on rare-class IoU.** MNv3-Small (the smallest and oldest of the three) is the strongest configuration on both min1 and min3 Big Rock IoU. The newer/fancier encoders both lose: MNv4 by 22 % / 36 %, MobileViT-XS by 19 % / 47 %.
2. **"Newer encoder = better" is empirically false at this budget on this task.** MNv4 (2024) loses to MNv3 (2019) despite +39 % params and +52 % FLOPs, and despite being from the same lineage and design team.
3. **The transformer-paradigm advantage at 47–81 M (paper_13) does not transfer down to 3 M.** MobileViT-XS performs worst of the three on rare-class IoU.
4. **Deployment failures are encoder-architecture-specific**: only MobileViT (the hybrid) fails the RAM cap and sees INT8 slowdown. Pure CNNs are deployment-clean. Important separately from the accuracy result.
5. **The 3 M cap itself isn't binding for accuracy** — the winner (MNv3-Small, 72 % of cap) is comfortably the lightest. Saturating the cap (MNv4 at 100 %, MobileViT at 97 %) provides no rare-class IoU benefit. **Smaller-and-simpler beat newer-and-saturating.**

**Final lever-ordering at 3 M cap on AI4Mars** (binding levers for rare-class IoU, in descending magnitude):

1. **Resolution** (×2.9 effect from 1024 → 512)
2. **Decoder family** (×1.6 effect from FPN → DLV3+)
3. **Encoder choice within same paradigm** (×1.3 effect from MNv4 → MNv3 on min3 BR IoU)
4. **Encoder paradigm** (×1.9 effect from hybrid → CNN; but largely contained in the encoder choice if both options are pure-CNN)

This is the strongest cross-architecture story the data supports: **at this budget, the rare-class metric is dominated by resolution, with decoder and encoder choices contributing diminishing returns**. The thesis can defensibly recommend `DeepLabV3+ + tu-mobilenetv3_small_100 @ 512²` as the strongest space-grade-compliant configuration for AI4Mars 4-class semantic segmentation, with quantitative evidence for *why* the more obvious upgrades (newer encoder, attention, FPN decoder, higher resolution) all underperform.

## Reference baseline from ITERATION-1

For accuracy comparison only — not a space-grade-compliant run.

| ID (ITERATION-1) | Architecture | Encoder | Params | min1 BR IoU | min3 BR IoU | min1 mIoU | min3 mIoU |
|---|---|---|---:|---:|---:|---:|---:|
| [001_baseline](../../ITERATION-1/experiments/001_baseline/) | U-Net | mobilenet_v2 | 6.63 M | 0.0953 | 0.4384 | 0.6488 | 0.8140 |

## Conventions (mirrors ITERATION-1/log.md)

- Headline reporting variant for Big Rock IoU is **min1-100agree** (most pixels labelled — most reliable rare-class measurement).
- min3 numbers kept as a secondary view for comparison with literature (most papers report on min3).
- mIoU is computed over labelled classes only (255 excluded).
- All experiments use the same fixed protocol (see [docs/protocol.md](../../docs/protocol.md)). The variable being studied in ITERATION-2 is architecture (vs ITERATION-1's variable, training strategy).
- Each run produces `config.json`, `weights.pth`, `training_history.csv`, `evaluation_results.json`, `space_grade.json`, `notes.md` under `experiments/<exp-id>/`.

## To register a new ITERATION-2 run

1. Train: `.venv/bin/python <ArchTrainer>_training_v2.py --exp-id <new-id> ...`
2. Evaluate: `.venv/bin/python evaluate.py --exp-id <new-id>`
3. Measure: `.venv/bin/python space_grade.py --exp-id <new-id>`
4. Write notes.md
5. Add a row above with headline numbers from `evaluation_results.json` + `space_grade.json`
