# ITERATION-2 — RQ1 with space-grade scoring

## What changed from ITERATION-1

| | ITERATION-1 | ITERATION-2 |
|---|---|---|
| Primary research question | RQ2 — simple training strategies for Big Rock | RQ1 — cross-architecture comparison |
| Strategies tested | unweighted CE, class-weighted CE, augmentation, combined, focal | (TBD — see "Planned architectures" below) |
| Architecture | fixed (U-Net + MobileNetV2) | varied (the variable being studied) |
| Outcome | negative — no simple strategy improved Big Rock IoU on a lightweight CNN | TBD |
| Hardware metric scope | none | **space-grade scoring** added |

## Scope decisions for this iteration

- **RQ2's negative result is taken as closed.** No further "simple training strategy" experiments planned. The ITERATION-1 grid (001–005) is the empirical evidence on which RQ2 conclusions rest.
- **RQ1 becomes the active research question.** Each architecture is compared on the fixed protocol (same data, loss, optimiser, epochs) so the architecture is the isolated variable.
- **Space-grade scoring is added as an additional reporting dimension.** For each architecture, the per-class IoU metrics from ITERATION-1's evaluation pipeline are still reported, plus deployment-relevant numbers.

## Open questions to lock down before first run

These need a decision from Keenan + Alisa before launching ITERATION-2's first experiment:

1. **Which architectures are in scope?** Candidates:
   - U-Net + MobileNetV2 (lightweight CNN — already trained in ITERATION-1; can be reused)
   - DeepLabv3+ + ResNet-50 (medium CNN baseline)
   - DeepLabv3+ + ResNet-101 (heavy CNN baseline)
   - SegFormer-B0 (small transformer)
   - SegFormer-B3 (large transformer)
   - Mobile-DeepRFB (literature-cited lightweight)
   - DDRNet, BiSeNet, FastSCNN (other lightweight CNN options)
2. **Which space-grade metrics to report.** Some options (rough priority):
   - Parameter count (free, deterministic) — already collected for ITERATION-1.
   - FLOPs / MACs (free, deterministic).
   - Checkpoint size on disk (free).
   - GPU inference latency (already collected indirectly via training speed).
   - CPU inference latency (representative of compute-constrained deployment).
   - ONNX exportability and ONNX-Runtime CPU latency.
   - INT8-quantised model size + latency (requires post-training quantisation).
3. **Is MER NCAM data still required for RQ1?** The thesis PDF says RQ1 reports per-rover-source. If yes, MER preprocessing remains a prerequisite.
4. **Does the thesis PDF need a revision to acknowledge space-grade scoring?** Currently §1.6 explicitly excludes hardware considerations.

## Plan once these decisions land

(To be filled in.)

## Folder convention (this iteration)

```
ITERATION-2/
├── notes.md                          (this file)
├── experiments/
│   ├── 001_<arch>/
│   │   ├── config.json
│   │   ├── weights.pth
│   │   ├── training_history.csv
│   │   ├── evaluation_results.json
│   │   ├── space_grade.json          (NEW: params, FLOPs, sizes, latency)
│   │   └── notes.md
│   ├── 002_<arch>/
│   └── ...
└── log.md                            (index of all ITERATION-2 runs)
```

Trainer and evaluator now default `--exp-root` to `ITERATION-2/experiments`. To target a different iteration's folder explicitly, override with `--exp-root <path>`.
