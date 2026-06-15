# Deployment-relevant constraints (the four SGCs)

This document defines the four deployment-relevant constraints under which the final model is evaluated. They are referred to in the codebase as **SGCs** (space-grade constraints, numbered #1–#4) and in the thesis as "space-grade constraint 1" through "space-grade constraint 4" or "the four deployment-relevant constraints".

Each constraint has a fixed threshold drawn from the lightweight rover-segmentation literature and the RAD750 specification. The thresholds are conservative for a CPU-class rover processor and would shift on hardware with a different envelope.

---

## SGC #1 - Parameter count

| | |
|---|---|
| **What is measured** | Total trainable scalar parameters across all layers of the model |
| **Threshold** | < 3,000,000 parameters |
| **How it is measured** | `sum(p.numel() for p in model.parameters())` |
| **Where in the code** | `sgc_evaluate_GOOD_2.0.py` - `count_params()` |

### Rationale

The 3 M cap is the upper end of the parameter budget used in the recent rover-segmentation literature. Mobile-DeepRFB (Feng et al. 2025) sits at ~3.5 M parameters; the multi-mission classifier by Atha et al. (2022) is in the same range. Anything significantly above this size starts to look like a desktop-class model that would not fit on a flight-grade processor.

The parameter count is invariant to input resolution, so this constraint applies equally to the 512 and 1024 variants of any model with the same architecture.

---

## SGC #2 - Worst-case latency

| | |
|---|---|
| **What is measured** | Maximum per-frame inference time over the gold expert test set |
| **Threshold** | < 500 milliseconds per frame |
| **How it is measured** | Single CPU thread, FP32 PyTorch runtime, `torch.set_num_threads(1)`, warm-up pass discarded, `time.perf_counter` around each forward pass |
| **Where in the code** | `cpu_latency_eval/latency_cpu.py` |

### Rationale

500 ms is the worst-case frame budget for a rover autonomy loop where each frame's segmentation must complete before the next frame arrives. The CPU-class measurement protocol approximates the RAD750 deployment envelope - the rover processor runs at ~200 MHz and does not have a GPU. Measuring on a single thread of a commodity CPU gives a conservative upper bound that does not over-promise.

A warm-up pass is discarded because the first-inference cold-start cost (PyTorch JIT warm-up, kernel caching) does not reflect the steady-state inference cost that determines whether the autonomy loop holds 2 Hz.

---

## SGC #3 - Peak segmentation-tensor memory

| | |
|---|---|
| **What is measured** | Peak resident memory occupied by the segmentation tensors during inference, measured separately from interpreter and library overhead |
| **Threshold** | < 256 megabytes |
| **How it is measured** | Process RSS via `psutil.Process(...).memory_info().rss` on CPU, or `torch.cuda.max_memory_allocated()` on GPU. RSS is taken before and after a single forward pass; the difference is the segmentation tensor footprint. |
| **Where in the code** | `sgc_evaluate_GOOD_2.0.py` - `read_peak_ram_mb()` |

### Rationale

The RAD750 ships with 256 MB of on-board RAM (BAE Systems specification). The whole flight software stack must fit inside this envelope, but for the purpose of this evaluation only the segmentation model's tensors are charged against the budget, since the rest of the flight stack is out of scope for the thesis.

This is the constraint that excludes the 1024 variant of the chosen architecture (407 MB peak vs the 256 MB limit). The 512 variant uses only 126 MB and fits.

---

## SGC #4 - Survivability under simulated radiation faults

| | |
|---|---|
| **What is measured** | Per-class IoU on the gold expert test set under simulated bit-flip faults, compared against the same IoU at baseline (no faults) |
| **Threshold** | The shielded mode must keep per-class IoU within ~0.001 of the baseline |
| **How it is measured** | A PyTorch forward hook on the encoder's deepest feature map flips between 1 and 5 random bits per forward pass across the IEEE-754 exponent (bits 23–30) and fraction (bits 0–22) ranges. The unshielded mode lets the corrupted activations propagate; the shielded mode applies an activation clamp to a fixed range of [-20, +20] and combines three independent model copies through a per-pixel majority vote (triple-modular redundancy). |
| **Where in the code** | `sgc_evaluate_GOOD_2.0.py` - `ChaoticSpaceRadiationInjector`, `BoundsCheckShield`, `evaluate_on_loader_tmr` |

### Rationale

Commercial silicon is vulnerable to soft errors caused by ionising radiation. A single energetic particle striking a memory cell can flip a bit (Single Event Upset). In space, the rate of these events is several orders of magnitude higher than at ground level, which is why flight computers are normally built on radiation-hardened processes like the RAD750.

This constraint approximates radiation survivability through fault injection on commodity hardware rather than direct measurement on a flight-grade processor. The injected fault distribution covers both catastrophic magnitude changes (exponent-bit flips) and silent value perturbations (fraction-bit flips). The shielding stack follows the bit-flip propagation analysis of Li et al. (2017), the fault-injection framework of Reagen et al. (2018), and the robust-ML survey of Shafique et al. (2020).

The survivability claim is bounded to the injected fault distribution and the defence stack that wraps it. It is not a prediction of flight performance on the RAD750 or any other rad-hardened processor.

---

## Constraint compliance summary for the chosen model

| Constraint | Threshold | `v2_R11_MNv4-S_512.pth` (chosen) | Verdict |
|---|---|---|---|
| SGC #1 - parameters | < 3 M | 3.00 M | PASS |
| SGC #2 - latency | < 500 ms | 248 ms (max, baseline) | PASS |
| SGC #3 - memory | < 256 MB | 126 MB (baseline) / 245 MB (shielded) | PASS |
| SGC #4 - survivability | within measurement noise of baseline | Big Rock IoU recovered to within 0.0003 of baseline | PASS |

The same protocol applied to the 1024 reference model fails SGC #2 (1275 ms) and SGC #3 (407 MB).
