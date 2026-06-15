# Thesis Scope — Inclusion/Exclusion Map

This document records which experiments across all iterations are **primary evidence** in the thesis, which are **excluded** from the deployable analysis, and the rationale for each exclusion. The goal is to keep the thesis narrative tight and avoid presenting experiments that don't load-bear a research question.

See also: [`ITERATION-2/cells_in_use.md`](ITERATION-2/cells_in_use.md) for the iter-2-specific cell list (this document repeats and extends it).

## High-level mapping to thesis chapters

| Chapter | Research question | Iteration(s) used |
|---|---|---|
| 5.1 Architecture | RQ1 (cross-architecture comparison) | ITERATION-2 — 9 of 15 cells |
| 5.2 Training Configuration | RQ1 (loss / aug / epochs) | MAIN_ITERATION v1 @ 512 (9 cells) + MAIN_ITERATION_V2 @ 1024 (8 cells) + MAIN_ITERATION_V2 @ 512 (3 cells) |
| 5.3 Deployment-Relevant Constraints | RQ2 (SGCs) | ML_CHOICE/ — chosen deployable + reference rows |

---

## ITERATION-1 — Legacy U-Net baseline

Pre-SGC-era work; not load-bearing for any thesis claim. Cited only as a literature anchor for relative comparison.

### Use (1 cell — reference only)

| Cell | Recipe | Role |
|---|---|---|
| `001_baseline` | U-Net + MobileNetV2, 6.63 M params (non-compliant with SGC #1) | Cite as the prior-work baseline: min1 BR 0.095, min3 BR 0.438 |

### Don't include

Everything else from `ITERATION-1/`. The folder is preserved for traceability, not for thesis evidence.

---

## ITERATION-2 — Architecture sweep (RQ1)

Already documented in [`ITERATION-2/cells_in_use.md`](ITERATION-2/cells_in_use.md). Summarised here.

### ✅ Include (9 cells — the deployable architecture sweep)

| Cell | Recipe | Role |
|---|---|---|
| 01 | DLV3+ / MNv3-S / 1024 | MNv3-S resolution lever at native resolution |
| 02 | DLV3+ / MNv3-S / 512 | MNv3-S sweet-spot |
| 03 | FPN / MNv3-S / 512 | Decoder lever at sweet-spot resolution |
| 04 | DLV3+ / MNv3-S / 256 | Resolution lever (low) |
| 06 | DLV3+ / **MNv4-S** / 512 | Encoder lever at sweet-spot resolution |
| 07 | FPN / MNv3-S / 1024 | Decoder × resolution corner |
| 08 | FPN / MNv3-S / 256 | Decoder × resolution corner |
| 12 | DLV3+ / **MNv4-S** / 256 | MNv4-S resolution sweep (low) |
| **13** | **DLV3+ / MNv4-S / 1024** | **Chosen architecture** for downstream RQ2 |

### ❌ Exclude (6 cells)

| Cell | Reason |
|---|---|
| 05, 14, 15 | MobileViT-XS family — rejected for (a) worst val→test divergence in the sweep, (b) INT8 + SDC deployment risk on attention-based encoders, (c) cell 015 could not be trained at 1024² with batch=8 due to O(N²) attention activation memory |
| 09, 10, 11 | Re-seed cells of 02 and 04 — used only for seed-variance characterisation, not for primary thesis evidence rows. Each (architecture, resolution) combination is reported as a single representative cell to keep tables clean |

---

## MAIN_ITERATION — v1 loss/aug ablation (MNv3-Small, RQ1)

The thesis uses the **512 grid only**. The 1024 runs were performed before the protocol switch to 512² and are not apples-to-apples with the rest of v1.

### ✅ Include (9 cells at MNv3-S / 512)

| Cell | Recipe | Role |
|---|---|---|
| `training_run_2_weighted_ce_512p` | wCE | Loss-family baseline (new fill-in at 512) |
| `training_run_3_focal_loss_512p` | focal γ=2 | Loss-family baseline (new fill-in at 512) |
| `training_run_4_focal_+_dice_512p` | focal + dice α=0.5 | Loss-family baseline (new fill-in at 512) |
| `training_run_8_focal_+_tversky_512p` | focal + tversky 0.3/0.7 / 15 ep | Tversky family 15-ep entry |
| **`training_run_9_focal_+_tversky_512p_+++`** | **focal + tversky 0.2/0.8 + basic / 25 ep** | **v1 chosen deployable** (val_BR* 0.599) |
| `training_run_10_focal_+_tversky_512p_b08` | focal + tversky 0.2/0.8 / 25 ep | Tversky tilt 0.2/0.8 control |
| `training_run_11_focal_+_tversky_512p_aug_B` | focal + tversky 0.3/0.7 + basic / 25 ep | Aug ablation: basic |
| `training_run_12_focal_+_tversky_512p_25epo` | focal + tversky 0.3/0.7 / 25 ep | Aug ablation: none (control) |
| `training_run_13_focal_+_tversky_512p_aug_S` | focal + tversky 0.3/0.7 + strong / 25 ep | Aug ablation: strong |

### ❌ Exclude (1024 + redundant)

| Cell | Reason |
|---|---|
| `training_run_1` | 10 ep / batch=8 — protocol-mismatched outlier (the only run with batch=8). Excluded to keep protocol parity within v1 |
| `training_run_1_20epochs` | Abandoned at epoch 13, no `weights.pth` artifact |
| `training_run_2_weighted_ce` (1024) | Superseded by `_512p` fill-in |
| `training_run_3_focal_loss` (1024) | Superseded by `_512p` fill-in |
| `training_run_4_focal_+_dice` (1024) | Superseded by `_512p` fill-in |
| `training_run_5_focal_+_dice_a03` | Redundant α-sweep variant of run 4; α=0.3 produced near-identical results to α=0.5. Keep run 4 as the focal+dice representative |
| `training_run_6_ftl` | Abraham–Khan single-term FTL formulation — different loss family from the focal+tversky compound, produced the worst test BR (0.149) in v1, treated as an abandoned exploration |
| `training_run_7_focal_+_tversky` (1024) | Superseded by run 8 at 512 (same recipe at the sweet-spot resolution) |

---

## MAIN_ITERATION_V2 — v2 loss/aug ablation (MNv4-Conv-Small, RQ1 + deployment evidence for RQ2)

v2 contains **both 1024 and 512 cells by design**. Each set has a distinct thesis purpose.

### ✅ Include at MNv4-S / 1024 (8 cells — "research ceiling" evidence)

Purpose: prove that the iter-2 architecture choice (MNv4-S at native resolution) wins on accuracy. Run 11 at 1024 is the **best accuracy in the project** but fails SGC #3, so it is the **research ceiling**, not the deployable.

| Cell | Recipe | Role |
|---|---|---|
| `training_run_2_weighted_ce_1024p` | wCE | Loss-family baseline at 1024 |
| `training_run_3_focal_loss_1024p` | focal γ=2 | Loss-family baseline at 1024 |
| `training_run_4_focal_+_dice_1024p` | focal + dice α=0.5 | Loss-family baseline at 1024 |
| `training_run_8_focal_+_tversky_1024p` | focal + tversky 0.3/0.7 / 15 ep | Tversky 15 ep |
| `training_run_10_focal_+_tversky_1024p_b08` | focal + tversky 0.2/0.8 / 25 ep | Tversky tilt 0.2/0.8 |
| **`training_run_11_focal_+_tversky_1024p_aug_B`** | **focal + tversky 0.3/0.7 + basic / 25 ep** | **Research ceiling** — test BR 0.480, but fails SGC #3 (406 MB > 256 MB cap) |
| `training_run_12_focal_+_tversky_1024p_25epo` | focal + tversky 0.3/0.7 / 25 ep | Aug ablation at 1024: none |
| `training_run_13_focal_+_tversky_1024p_aug_S` | focal + tversky 0.3/0.7 + strong / 25 ep | Aug ablation at 1024: strong |

### ✅ Include at MNv4-S / 512 (3 cells — "deployable" evidence)

Purpose: prove that the chosen architecture fits the SGC #3 memory cap at deployable resolution and that the augmentation ranking transfers from 1024 to 512.

| Cell | Recipe | Role |
|---|---|---|
| ⭐ **`training_run_11_focal_+_tversky_512p_aug_B`** | **focal + tversky 0.3/0.7 + basic / 25 ep** | **Chosen deployable** — test BR 0.446, passes all 4 SGCs |
| `training_run_12_focal_+_tversky_512p_25epo` | focal + tversky 0.3/0.7 / 25 ep | Aug ablation at 512: none (control) |
| `training_run_13_focal_+_tversky_512p_aug_S` | focal + tversky 0.3/0.7 + strong / 25 ep | Aug ablation at 512: strong |

### ❌ NOT in scope (never ran — scope decision, not missing data)

| Missing | Why it's a scope decision |
|---|---|
| Loss family at MNv4-S / 512 (wCE, focal, focal+dice equivalents) | Loss-family ranking was established at MNv4-S/1024 (v2 stage 1) and is consistent with v1 MNv3-S/512 findings. Re-running the four loss-family cells at 512 was unnecessary for the deployment claim and was deprioritised in favour of GPU time on the deployment-critical augmentation ablation |
| Tversky tilt 0.2/0.8 at MNv4-S / 512 | Tilt comparison was established at MNv4-S/1024 (run 10 vs run 12); the chosen tilt (0.3/0.7) was applied directly to the 512 deployment |

**Honest framing for the thesis**: these are deliberate scope choices, not gaps. The augmentation finding (basic > none > strong) is the only one re-validated at 512 because it has the biggest accuracy impact and the largest val→test divergence risk. Loss family and tversky tilt findings transfer across resolutions cleanly based on v1 and v2-1024 evidence.

---

## ML_CHOICE/ — The locked-in deployment artifacts

Only one cell is the chosen deployable. The other two are **reference rows** for comparison.

### Models on file

| File | Role |
|---|---|
| `v1_R9_MNv3-S_512.pth` | v1 winner — reference for "previous deployable" comparison row |
| `v2_R11_MNv4-S_1024.pth` | Research ceiling — best test BR in the project, fails SGC #3, NOT deployable |
| ⭐ `v2_R11_MNv4-S_512.pth` | **Chosen deployable** — passes ALL 4 SGCs |

### SGC results on file

| Folder | Role |
|---|---|
| `sgc/v1_R9_MNv3-S_512_baseline/` | SGC baseline for v1 R9 (passes 1-3 by GPU latency measurement convention) |
| `sgc/v2_R11_MNv4-S_1024_baseline/` | SGC baseline for v2 R11 @ 1024 — fails SGC #3 (406 MB > 256 MB cap). The "why we pivoted to 512" evidence |
| `sgc/v2_R11_MNv4-S_512_baseline/` | SGC #1–3 baseline for chosen deployable (passes all three with margin) |
| `sgc/v2_R11_MNv4-S_512_unshielded_chaos/` | SGC #4 unshielded: scaled radiation damage (BR 0.4463 → 0.0189) |
| `sgc/v2_R11_MNv4-S_512_shielded_chaos/` | SGC #4 shielded: TMR + clamping recovers to BR 0.4460 |

See [`ML_CHOICE/README.md`](ML_CHOICE/README.md) for the per-cell numerical tables.

---

## Recommended chapter-by-chapter cell mapping

### Chapter 5.1 Architecture (RQ1)

**Cells used**: ITERATION-2 cells 01, 02, 03, 04, 06, 07, 08, 12, 13 (9 cells)

**Headline figures**:
- Table 5.2: 9-row architecture comparison (one row per cell)
- Figure 5.3: resolution lever curves for DLV3+ MNv3-S and FPN MNv3-S families
- Figure 5.4 (new): resolution lever curve for DLV3+ MNv4-S (cells 12, 06, 13) — shows the monotone-up pattern

### Chapter 5.2 Training Configuration (RQ1)

**Cells used**: v1 @ 512 (9 cells) + v2 @ 1024 (8 cells) + v2 @ 512 (3 cells)

**Headline figures**:
- Table 5.3: v1 loss/aug ablation at 512 (9-row table — one row per v1 cell)
- Table 5.4 (new): v2 loss/aug ablation at 1024 (8-row table)
- Figure / Table comparing v1 R9 vs v2 R11 @ 1024 vs v2 R11 @ 512 (encoder + resolution lifts on test set)

### Chapter 5.3 Deployment-Relevant Constraints (RQ2)

**Cells used**: ML_CHOICE/ — v2 R11 @ 512 as chosen deployable, with v1 R9 and v2 R11 @ 1024 as reference rows

**Headline figures**:
- Table 5.5 (updated): hardware-budget compliance for v2 R11 @ 512 across baseline / unshielded / shielded modes
- Table 5.6 (updated): per-class IoU under three modes (baseline / unshielded chaos / shielded chaos) for v2 R11 @ 512

---

## Defensible answers to anticipated reviewer questions

### "Why exclude iter-2 cells 05, 09, 10, 11, 14, 15?"

> Reseed cells (09, 10, 11) were run specifically for seed-variance characterisation on cells 02 and 04. To keep cross-architecture tables clean, each (architecture, resolution) combination is reported as a single representative cell. The MobileViT-XS family (05, 14, 15) was excluded from the deployable set on three grounds: (a) cell 05 had the worst val→test BR divergence in the sweep, (b) cell 015 could not be trained at 1024² batch=8 due to quadratic attention activation memory, and (c) INT8 + SDC deployment behaviour on attention-based encoders is not characterised in the AI4Mars literature, raising un-validated deployment risk relative to pure-CNN encoders.

### "Why exclude v1 1024 runs?"

> v1 used native 1024² input for the loss-family stage and 512² for the recipe-ablation stage, a protocol shift that confounds cross-stage comparisons within v1 with resolution. To restore protocol parity within v1, the loss-family runs were re-trained at 512² (cells 2_512p, 3_512p, 4_512p), producing a clean 9-cell v1 grid at one resolution.

### "Why didn't you run loss family at MNv4-S / 512?"

> The loss-family ranking was established at MNv4-S / 1024 (v2 stage 1, Section X.X) and is consistent with the v1 MNv3-S / 512 finding (Section Y.Y). Re-running the four loss-family cells at MNv4-S / 512 would have provided incremental confirmation but no new claim; GPU time was prioritised on the augmentation ablation at 512, which is the lever with the largest accuracy impact and the largest val→test divergence risk on the rare class. The chosen recipe (focal+tversky 0.3/0.7 + basic aug) was applied directly to the 512 deployment based on convergent evidence from v1, v2-1024, and v2-512 aug ablations.

### "Why didn't you do INT8 quantization?"

> INT8 quantization was investigated as a candidate fix for the v2 R11 @ 1024 memory failure but abandoned for three reasons: (a) PyTorch FX static quantization is CPU-only, so the latency benchmark would shift from GPU FP32 to CPU INT8 — inconsistent with the SGC #2 convention defined in Section 4.4.1, (b) full-graph post-training INT8 on DeepLabV3+ collapses accuracy substantially (the ASPP module's parallel-branch concatenations with mismatched activation scales is a documented weakness in the segmentation quantization literature), (c) the simpler pivot of training at 512² produces a deployable model in FP32 that already passes all four SGCs. INT8 deployment is noted as future work in the SGC chapter conclusion.

---

## File status legend (for repo readers)

| Folder/File | Status |
|---|---|
| `ITERATION-1/` | Historical baseline — preserved for traceability, not load-bearing |
| `ITERATION-2/` | Architecture sweep — primary evidence for RQ1 architecture chapter |
| `MAIN_ITERATION/` | v1 loss/aug ablation — primary evidence for RQ1 training-config chapter (512 grid only) |
| `MAIN_ITERATION_V2/` | v2 loss/aug ablation — primary evidence for RQ1 training-config chapter (1024 + 512 grids) and the source of the chosen deployable |
| `ML_CHOICE/` | Locked-in deployment artifacts — primary evidence for RQ2 deployment chapter |
| `legacy-code/` | Old scripts, old weights — preserved for traceability, not referenced in the thesis |
| `docs/` | Thesis source + lit-review extractions |
| `training/` | Active loss/augmentation/class-weight modules (used by trainers) |
| `MSL_NAVCAM_TRAINING_SET/`, `MSL_NAVCAM_TEST_SET/` | Datasets — not synced to local machine, not modified |
