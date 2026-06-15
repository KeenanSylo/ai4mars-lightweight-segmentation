# Fixed evaluation protocol — first cut

This pins down how every experiment in this thesis is trained and evaluated, so that comparisons across models (RQ1) and across rare-class strategies (RQ2) are like-for-like.

Anything marked **[TBD]** is still under discussion.

---

## Dataset

- **Source:** AI4Mars (Swan et al., CVPRW 2021), official release.
- **Subsets used:** MSL NCAM (Curiosity). MER NCAM (Spirit + Opportunity) **[TBD — pending preprocessing]**.
- **Training labels:** the released merged labels — agreement among ≥3 labellers and >65% pixel agreement (per thesis §1.6).
- **Background / ignore pixels:** label value `255`. Excluded from loss and from all metrics.
- **Per-image masks (training only):** for each EDR frame, OR the rover mask (`mxy`) and range mask (`rng-30m`) into the label and overwrite those pixels with `255`. Test labels (gold) already have this baked in, so no extra mask step at evaluation.

## Test set (for reporting)

- The expert-labelled gold test set under `data/msl/ncam/labels/test/`.
- Three variants exist: `masked-gold-min1-100agree`, `…min2…`, `…min3…`. Each refers to expert-agreement threshold (`min1` = ≥1 expert at 100% pixel agreement, `min3` = all 3 experts agree).
- **Headline reporting variant:** `min1-100agree`.
  - **Why:** RQ2 is about Big Rock specifically. min1 has Big Rock pixels in 53/322 images; min3 has only 5/322. min3 makes the rare-class IoU statistically unreliable.
  - All three are still computed and reported in each run's `evaluation_results.json` for transparency and comparison to literature (which usually reports min3).
- **Fully-ignored test images** (label is all `255`) are skipped during evaluation — they don't affect metrics either way and just waste compute. Skipped count is logged.

## Train / val / test split

- **Test:** the gold set above. Never seen during training.
- **Train / val:** 80 / 20 random split of the released training set, `random_state=42`. **[TBD — should val also come from the gold set? Currently it's a random slice of train, which the thesis §1.3 criticises.]**

## Input pipeline

- **Resolution:** 1024 × 1024 (native AI4Mars NCAM size). All models are trained and evaluated at this resolution.
  - **ITERATION-2 extension (added 2026-05-14, for the resolution-lever experiment):** input resolution is now a documented experimental variable. The default remains 1024² and that is the protocol-compliant value for cross-experiment comparison. Runs that explicitly study throughput vs. resolution may opt in to a different square resolution via `--input-hw` on both the trainer and `evaluate.py`. When `--input-hw` is set:
    - **Training**: `MSL_train_preprocessor` resizes EDR (bilinear) and label (nearest) to the requested size *after* the rover/range ignore-masking has been applied at native resolution. Loss is computed at the downsampled resolution.
    - **Evaluation (Strategy A)**: input EDR is resized to the requested size, the model predicts at that size, and the *logits* are bilinear-upsampled back to the gold label's native resolution (1024²) before argmax / softmax. IoU and confidence are therefore always computed against the unchanged gold label, so cross-resolution numbers stay comparable.
    - The chosen resolution is recorded under `config.json` → `data.input_hw`. `evaluate.py` reads this automatically when `--exp-id` is given, so the resolution stays consistent between training and reporting.
  - Per-run notes.md must call out the resolution explicitly. The headline IoU comparison row in `experiments/log.md` may include rows at different resolutions; readers should consult the run's notes.md for the resolution used.
- **Channel order:** RGB (OpenCV BGR is converted at load time).
- **Normalization:** divide by `255.0`. No per-channel mean / std subtraction. **[TBD — switch to ImageNet mean/std for fairer pretrained-encoder comparison?]**
- **Augmentation (RQ1 baseline runs):** none.
- **Augmentation (RQ2 experiments):** Albumentations `Compose` named `"basic"` in [`augmentations.py`](../augmentations.py), applied **only to the training set** (validation and test never augmented):
  - `HorizontalFlip(p=0.5)`
  - `RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)`
  - `HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=0, p=0.3)`
  - `Affine(rotate=(-8°, +8°), translate_percent=(0.0, 0.05), p=0.3)` — image fill `0`, **mask fill `255`** so rotation-introduced corner pixels are excluded from the loss; mask uses `INTER_NEAREST` to preserve discrete labels.
  - `ToFloat(max_value=255.0)` → `ToTensorV2()` — produces CHW float32 in [0, 1] image and HW long mask.
- Random crop + resize was considered but kept out of the locked pipeline so all experiments share the same 1024 × 1024 field of view; can be reintroduced as a separate `"strong"` pipeline if the augmentation level is itself studied later.

## Loss

- **Default:** `CrossEntropyLoss(ignore_index=255)`.
- **RQ2 variants under test (committed 2 × 2 design):**
  - Loss: unweighted CE  vs  class-weighted CE.
  - Augmentation: none  vs  `"basic"`.

  Four runs in total. Focal loss, Dice + CE, and clipped class weights remain available as fallbacks if the 2 × 2 grid does not yield a strategy that improves Big Rock IoU on the headline test variant; not committed to in advance.
- **Class weights (when used):** computed by [`class_weights.py`](../class_weights.py) using sklearn-balanced inverse frequency: `w_c = N / (K · n_c)`, where `N` = total non-`255` pixels across all training labels, `K` = number of classes, `n_c` = pixel count for class `c`. Rover and range masks are applied before counting so weights reflect the loss-time pixel distribution exactly. Result is cached as JSON under the data directory. **No clipping** is applied to the resulting weights. (The actual computed weights and the empirical effect on Big Rock IoU are reported per-run in `experiments/<id>/notes.md` and `experiments/log.md`, not here.)

## Optimizer & schedule

- Optimizer: Adam.
- Learning rate: `1e-3`.
- LR schedule: **[TBD — currently none; consider cosine or step decay once experiments are longer than 10 epochs.]**
- Weight decay: 0 (default). **[TBD]**

## Batching

- Train batch size: 8. **[TBD per arch — transformers may need smaller.]**
- Eval batch size: 4.
- Workers: 4.

## Training duration

- Initial choice: 10 epochs. **[TBD — likely too few; revisit before RQ1 model comparison.]**
- Early stopping: not used. **[TBD]**

## Metrics reported per run

Always all of these, computed from a single confusion matrix per test variant:

- Pixel accuracy.
- Mean IoU over labelled classes (255 excluded).
- Per-class IoU, precision, recall, support — for soil, bedrock, sand, big_rock.

Per-rover-source breakdown (RQ1) is added once MER data is available.

## Reproducibility

- `random_state=42` for the train/val split.
- `weights_only=True` when loading checkpoints.
- Each experiment lives under `experiments/<id>/` with `config.json` (this protocol's parameters), `weights.pth`, `training_history.csv`, `evaluation_results.json`, `notes.md`.
- `experiments/log.md` is the index of all runs with their headline numbers.

## Hardware

- University server, NVIDIA RTX-class GPU (single card). Inference times reported in this thesis are only meaningful in that context.
