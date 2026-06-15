# Thesis Scope — Inclusion/Exclusion Map

This document records which of the experiments in this repository are **primary evidence** in the thesis and which were run but **excluded** from the headline analysis, with the rationale for each exclusion. The goal is to keep the empirical record transparent without inflating the thesis narrative with experiments that do not load-bear a research question.

See also: [`architecture_sweep/cells_in_use.md`](architecture_sweep/cells_in_use.md) for the per-cell inclusion list of the architecture sweep.

---

## High-level mapping to thesis chapters

| Thesis chapter | Research question | Repository folder | In-scope runs |
|---|---|---|---|
| §5.1 Architecture | RQ1 (cross-architecture comparison) | [`architecture_sweep/`](architecture_sweep/) | 9 of 15 cells |
| §5.2 Training Configuration | RQ1 (loss, augmentation, duration) | [`training_configuration_sweep/`](training_configuration_sweep/) | 16 of 16 runs (8 configurations × 2 resolutions) |
| §5.3 Deployment-Relevant Constraints | RQ2 (parameter, latency, memory, survivability) | [`final_model/`](final_model/) | the chosen 512 model and the 1024 reference |

---

## Stage 1 — Architecture sweep ([`architecture_sweep/`](architecture_sweep/))

The architecture sweep evaluates DeepLabv3+ and FPN decoders paired with MobileNetV3-Small and MobileNetV4-Conv-Small encoders across three input resolutions (256, 512, 1024). The folder contains 15 trained cells. The thesis reports on 9 of them as the 3×3 grid that the architecture-lever analysis runs on.

### Included (9 cells — thesis Table 5.2)

| Cell | Decoder | Encoder | Resolution | Role |
|---|---|---|---|---|
| 01 | DLV3+ | MobileNetV3-Small | 1024 | resolution lever (high) |
| 02 | DLV3+ | MobileNetV3-Small | 512 | resolution lever (mid) |
| 03 | FPN | MobileNetV3-Small | 512 | decoder lever |
| 04 | DLV3+ | MobileNetV3-Small | 256 | resolution lever (low) |
| 06 | DLV3+ | **MobileNetV4-Conv-Small** | 512 | encoder lever |
| 07 | FPN | MobileNetV3-Small | 1024 | decoder × resolution corner |
| 08 | FPN | MobileNetV3-Small | 256 | decoder × resolution corner |
| 12 | DLV3+ | **MobileNetV4-Conv-Small** | 256 | MNv4-S resolution sweep (low) |
| 13 | DLV3+ | **MobileNetV4-Conv-Small** | 1024 | **chosen architecture** for the training-configuration stage |

### Excluded (6 cells, kept in the folder for transparency)

| Cell | Configuration | Reason for exclusion |
|---|---|---|
| 05 | DLV3+ / MobileViT-XS / 512 | MobileViT family not in scope for the deployable comparison (transformer-class encoders are evaluated separately in the related-work survey, not on the deployable path) |
| 09 | DLV3+ / MobileNetV3-Small / 512 (seed replicate of cell 02) | seed-variation run, not part of the headline lever-isolation table |
| 10 | DLV3+ / MobileNetV3-Small / 256 (seed replicate of cell 04) | seed-variation run |
| 11 | DLV3+ / MobileNetV3-Small / 256 (second seed replicate of cell 04) | seed-variation run |
| 14 | DLV3+ / MobileViT-XS / 256 | MobileViT family not in scope |
| 15 | DLV3+ / MobileViT-XS / 1024 | MobileViT family not in scope; run also did not complete |

---

## Stage 2 — Training-configuration sweep ([`training_configuration_sweep/`](training_configuration_sweep/))

The training-configuration sweep locks the architecture to the strongest cell from Stage 1 (DeepLabv3+ with MobileNetV4-Conv-Small) and varies the loss function, the augmentation pipeline, the training resolution, and the training duration.

### Included (16 runs — thesis Tables 5.3 and 5.4)

Eight training configurations evaluated at two resolutions (1024 and 512):

| Run | Loss | Augmentation | Epochs | Role |
|---|---|---|---|---|
| R2 | Class-weighted cross-entropy | None | 15 | reweighting baseline |
| R3 | Focal | None | 15 | focal baseline |
| R4 | Focal + Dice (equal weights) | None | 15 | hybrid loss |
| R8 | Focal + Tversky (α=0.3, β=0.7) | None | 15 | asymmetric Tversky |
| R10 | Focal + Tversky (α=0.2, β=0.8) | None | 25 | tilt-comparison run |
| **R11** | **Focal + Tversky (α=0.3, β=0.7)** | **Basic** | **25** | **chosen final configuration** |
| R12 | Focal + Tversky (α=0.3, β=0.7) | None | 25 | no-augmentation control for R11 |
| R13 | Focal + Tversky (α=0.3, β=0.7) | Strong | 25 | augmentation-strength control |

The strongest run by validation Big Rock IoU is R11. Both its 1024 and 512 variants are taken forward into Stage 3 and evaluated on the gold expert test set in Stage 3.

---

## Stage 3 — Final model and deployment-relevant constraints ([`final_model/`](final_model/))

Stage 3 takes the two strongest runs from Stage 2 and evaluates them under the four deployment-relevant constraints (parameter count, latency, memory, survivability under simulated radiation faults). Final test-set numbers and the constraint scorecards are in this folder.

### Included

| Artifact | Role | Test Big Rock IoU |
|---|---|---|
| `v2_R11_MNv4-S_512.pth` | **Chosen deployable model** | 0.446 |
| `v2_R11_MNv4-S_1024.pth` | Higher-accuracy reference (excluded by memory and latency limits) | 0.480 |
| `v1_R9_MNv3-S_512.pth` | Earlier baseline (cited as comparison only) | 0.325 |

The `final_model/sgc/` subfolder contains the per-constraint scorecards in JSON form for the chosen 512 model in three modes (baseline, unshielded fault injection, shielded fault injection with activation clamping and three-copy majority vote).

---

## What is intentionally NOT in this repository

- **The AI4Mars dataset itself.** This is a NASA JPL public release; see the main `README.md` for the download link.
- **Per-epoch checkpoints.** Each training run produced one `weights.pth` per epoch; only the best-validation-epoch checkpoint is retained per run.
- **Older iterations and exploratory work.** Earlier ITERATION-1 baselines, older `MAIN_ITERATION` runs, and quantization-related artifacts (ONNX exports, INT8 calibration variants) were dropped before the public release to keep the repository focused on the experiments that load-bear the thesis chapters.
