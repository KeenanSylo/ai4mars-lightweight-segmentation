# ITERATION-2 — Cells included in the thesis

This document records which of the ITERATION-2 cells are used as primary evidence in the thesis architecture-selection chapter, and which are excluded.

The exclusions are deliberate methodological choices, not failures. They are documented here so the thesis cell count (9) matches the experiments folder's apparent cell count (15) with a transparent paper trail.

## ✅ Included — 9 cells

These cells form the architecture × resolution sweep used to defend the choice of **DeepLabV3+ + MobileNetV4-Conv-Small @ 1024² (cell 13)** as the v2 backbone.

| Cell | Decoder | Encoder | Resolution | Role in thesis |
|---|---|---|---|---|
| 01 | DLV3+ | MNv3-S | 1024 | resolution lever / MNv3-S baseline at native |
| 02 | DLV3+ | MNv3-S | 512 | resolution lever / sweet-spot-claim for MNv3-S |
| 03 | FPN | MNv3-S | 512 | decoder lever (FPN vs DLV3+) at 512 |
| 04 | DLV3+ | MNv3-S | 256 | resolution lever / under-resolved baseline |
| 06 | DLV3+ | MNv4-S | 512 | encoder lever (MNv4-S vs MNv3-S) at 512 |
| 07 | FPN | MNv3-S | 1024 | decoder × resolution corner |
| 08 | FPN | MNv3-S | 256 | decoder × resolution corner |
| 12 | DLV3+ | MNv4-S | 256 | MNv4-S resolution sweep — low |
| 13 | DLV3+ | MNv4-S | 1024 | **chosen architecture** for MAIN_ITERATION_V2 |

Together these 9 cells form a clean 3 × 3 grid for `DeepLabV3+ × {MNv3-S, MNv4-S} × {256, 512, 1024}` plus 3 FPN cells at MNv3-S for the decoder ablation.

## ❌ Excluded — 6 cells (reasons below)

| Cell | Decoder | Encoder | Resolution | Reason for exclusion |
|---|---|---|---|---|
| 05 | DLV3+ | MViT-XS | 512 | MobileViT-XS family rejected |
| 09 | DLV3+ | MNv3-S | 512 | seed-replicate of cell 02 |
| 10 | DLV3+ | MNv3-S | 256 | seed-replicate of cell 04 |
| 11 | DLV3+ | MNv3-S | 256 | seed-replicate of cell 04 (second seed) |
| 14 | DLV3+ | MViT-XS | 256 | MobileViT-XS family rejected |
| 15 | DLV3+ | MViT-XS | 1024 | never completed (OOM at batch=8) + MViT-XS family rejected |

### Why the reseed cells (09, 10, 11) are excluded

Cells 09, 10, 11 were re-runs of cells 02 and 04 with different random seeds, used during the sweep to estimate seed-variance on rare-class IoU. The variance was characterised (val_BR std ≈ 0.04 on 256, ≈ 0.03 on 512) and the finding was published into [docs/protocol.md / log.md]; the reseed cells themselves are not used as primary evidence rows because each (architecture, resolution) combination is reported as a single representative cell to keep the thesis tables clean.

### Why the MobileViT-XS family (05, 14, 15) is excluded

The MobileViT-XS encoder was tested as a hybrid CNN + transformer candidate against pure-CNN MNv3-S and MNv4-Conv-Small. It is excluded from primary evidence for **three reasons**, in descending importance:

1. **Worst val→test divergence in the architecture sweep.** Cell 05 had the highest val Big Rock IoU in the table (0.482) but dropped to 0.355 on test (Δ −0.127), the largest val→test drop of any iter-2 cell. The val ceiling was not reliable as a test predictor for the transformer family.

2. **Deployment risk against the space-grade constraints.** Self-attention quantization to INT8 is significantly less mature than for pure CNNs, and bit-flip SDC behaviour on attention activations is not characterised in the AI4Mars literature. Choosing MViT-XS as the downstream architecture would have inherited unvalidated SGC #2 (INT8) and SGC #5 (SDC) risks.

3. **Quadratic attention activation cost at higher resolution.** Cell 15 (MViT-XS @ 1024) could not be trained at batch=8 on a 25 GB GPU due to O(N²) attention activation memory at 16,384 tokens — a fundamental architectural limit, not a tuning issue. This blocked any direct apples-to-apples comparison with cell 13 (MNv4-Conv-Small @ 1024) at the chosen resolution.

The MViT-XS data points exist in the experiments folder for reproducibility but are not part of the load-bearing architecture argument.

## Cell-numbering caveat

Future readers: cell numbers (01–15) are kept stable for historical reference. The "in use" cells listed above are **the 9 referenced by figures, tables, and claims in the thesis architecture-selection chapter**. The remaining 6 cells live in the same folder for transparency, not as supporting evidence.
