"""Trainer for the AI4Mars DeepLabV3+ pipeline (space-grade-compliant).

ITERATION-2 trainer using DeepLabV3+ as the decoder family and a MobileNetV3-Small
encoder by default. Total parameter count stays below the 3 M space-grade cap.

Feature-complete (mirrors U-Net_training_v2.py except the architecture line):
- Argparse with `--exp-id` (required) to scope outputs to <exp-root>/<exp-id>/
- Choice of loss (`--loss ce` or `--loss focal`) via losses.get_loss
- Optional class-weighted CE via `--class-weights <path-to-json>`
- Optional Albumentations augmentation pipeline via `--augmentation`
- Train + validation loops with per-epoch confusion-matrix-derived metrics
- Saves a checkpoint after every epoch into <exp-id>/checkpoints/epoch_NN.pth so
  the best-epoch model can be picked post-hoc via `select_best.py`
- Optional early stopping on val_loss via `--early-stop-patience`

Sister trainer: `U-Net_training_v2.py`. The only structural differences are
the architecture instantiation in PART-2 (`smp.DeepLabV3Plus` vs `smp.Unet`)
and the default encoder. If you change any other part of this file, propagate
it to the U-Net trainer too — RQ1 comparisons assume the two pipelines are
otherwise identical.

Outputs:
    <exp-root>/<exp-id>/config.json                 - exact CLI args + derived settings
    <exp-root>/<exp-id>/training_history.csv        - per-epoch metrics (29 columns)
    <exp-root>/<exp-id>/checkpoints/epoch_NN.pth    - one state_dict per completed epoch

Run after training:
    python select_best.py --exp-id <exp-id> --metric val_miou
                                              - copy best epoch to weights.pth
    python evaluate.py --exp-id <exp-id>      - per-class IoU on all 3 gold variants
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from MSL_train_preprocessor import MSL_train_preprocessor
from training.augmentations import get_transform, PIPELINES
from training.losses import get_loss, describe_loss, LOSSES
from metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    IGNORE_INDEX,
    update_confusion_matrix_torch,
    compute_metrics,
)


def build_csv_header(class_names):
    """Return the 29-column training_history.csv header for the current pipeline.

    Order: epoch, then a 14-column train block, then a 14-column val block.
    Each phase block: loss, pixel_acc, macro_precision, macro_recall, macro_f1,
    miou, per-class IoU (4 cols), per-class F1 (4 cols).
    """
    cols = ["epoch"]
    for phase in ("train", "val"):
        cols += [
            f"{phase}_loss",
            f"{phase}_pixel_acc",
            f"{phase}_macro_precision",
            f"{phase}_macro_recall",
            f"{phase}_macro_f1",
            f"{phase}_miou",
        ]
        cols += [f"{phase}_iou_{c}" for c in class_names]
        cols += [f"{phase}_f1_{c}" for c in class_names]
    return cols


def epoch_row(epoch, train_loss, train_m, val_loss, val_m, class_names):
    """Assemble a single CSV row from epoch losses + compute_metrics dicts."""
    row = [epoch]
    for loss, m in ((train_loss, train_m), (val_loss, val_m)):
        row += [
            loss,
            m["pixel_acc"],
            m["macro_precision"],
            m["macro_recall"],
            m["macro_f1"],
            m["miou"],
        ]
        row += [float(m["iou"][i]) for i in range(len(class_names))]
        row += [float(m["f1"][i]) for i in range(len(class_names))]
    return row


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train DeepLabV3+ + MobileNetV3-Small on AI4Mars MSL NCAM "
                    "under the ITERATION-2 space-grade constraints. "
                    "Outputs go to <exp-root>/<exp-id>/."
    )
    ap.add_argument("--exp-id", required=True,
                    help="Experiment ID. Outputs saved under <exp-root>/<exp-id>/.")
    ap.add_argument("--exp-root", default="MAIN_ITERATION/experiments",
                    help="Parent folder for experiment outputs. Defaults to the "
                         "current iteration's experiments/ folder.")
    ap.add_argument("--description", default="",
                    help="One-line description recorded in config.json.")
    ap.add_argument("--data-dir", default="MSL_NAVCAM_TRAINING_SET",
                    help="Refined-data root (containing images_op1/ and labels_op1/).")
    ap.add_argument("--encoder", default="tu-mobilenetv3_small_100",
                    help="smp encoder name. Default tu-mobilenetv3_small_100 keeps "
                         "total params at ~2.16 M (under the 3 M space-grade cap).")
    ap.add_argument("--encoder-weights", default="imagenet",
                    help="Encoder pretraining. Use 'imagenet' or None.")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-weights",
                    help="Path to a JSON file with a 'weights_list' field "
                         "(see training/class_weights.py). Only valid with --loss ce.")
    ap.add_argument("--augmentation", default="none", choices=sorted(PIPELINES),
                    help="Albumentations pipeline applied to the training set "
                         "(see training/augmentations.py). Validation is never augmented.")
    ap.add_argument("--loss", default="ce", choices=sorted(LOSSES),
                    help="Loss function (see training/losses.py). 'ce' = CrossEntropyLoss "
                         "(optional per-class weights via --class-weights). 'focal' = "
                         "smp.losses.FocalLoss (--focal-gamma controls focusing; class "
                         "weights not supported with focal).")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal-loss focusing parameter γ. Used by --loss focal "
                         "and --loss focal_dice.")
    ap.add_argument("--focal-dice-alpha", type=float, default=0.5,
                    help="Mixing weight α for --loss focal_dice: "
                         "α*Focal + (1-α)*Dice. Range [0, 1]. "
                         "α=1 is pure focal, α=0 is pure Dice.")
    ap.add_argument("--focal-tversky-alpha", type=float, default=0.5,
                    help="Mixing weight α for --loss focal_tversky: "
                         "α*Focal + (1-α)*Tversky. Range [0, 1]. "
                         "α=1 is pure focal, α=0 is pure Tversky.")
    ap.add_argument("--tversky-alpha", type=float, default=0.3,
                    help="Tversky FP weight for --loss ftl. "
                         "Paper default 0.3 for rare-class segmentation.")
    ap.add_argument("--tversky-beta", type=float, default=0.7,
                    help="Tversky FN weight for --loss ftl. Larger β → more "
                         "penalty for missed pixels → better recall on rare "
                         "classes. Paper default 0.7.")
    ap.add_argument("--ftl-gamma", type=float, default=4.0 / 3.0,
                    help="Focal exponent for --loss ftl: FTL = (1 − TI) ** γ. "
                         "Paper default 4/3 ≈ 1.333 (Abraham & Khan, 2018).")
    ap.add_argument("--input-hw", type=int, default=None,
                    help="Square input resolution for training. None = native "
                         "resolution (1024×1024). Set to 512 (or smaller) to study "
                         "the resolution lever on space-grade throughput. Image is "
                         "bilinear-resized, label is nearest-resized so class indices "
                         "and the 255 ignore region are preserved.")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="Stop training if val_loss does not improve for this many "
                         "consecutive epochs. 0 = disabled (default). Per-epoch "
                         "checkpoints already saved are kept; select_best.py picks "
                         "the winning epoch afterwards regardless of where we stopped.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing experiments/<exp-id>/ run "
                         "(training_history.csv + checkpoints/).")
    return ap.parse_args()


def load_class_weights(path, num_classes):
    """Load a class-weights JSON; expects a top-level 'weights_list' of length num_classes."""
    data = json.loads(Path(path).read_text())
    weights = data.get("weights_list")
    if weights is None or len(weights) != num_classes:
        raise SystemExit(
            f"{path} must contain a 'weights_list' of length {num_classes}; got {weights!r}"
        )
    return list(map(float, weights)), data


def main():
    args = parse_args()

    # Validate mutually-exclusive flags up front, before any expensive setup.
    if args.loss in {"focal", "focal_dice", "focal_tversky", "ftl"} and args.class_weights:
        raise SystemExit(
            f"--class-weights is not supported with --loss {args.loss}. "
            "These losses control class balance via their own knobs "
            "(focal-gamma / tversky alpha/beta), not per-class weights. "
            "Pick one mechanism or the other."
        )

    exp_dir = Path(args.exp_root) / args.exp_id
    checkpoints_dir = exp_dir / "checkpoints"
    history_file = exp_dir / "training_history.csv"
    if history_file.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite an existing run at {exp_dir}. "
            f"Use --force or pick a different --exp-id."
        )
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # PART-1: DATA
    # ==========================================
    data_dir = Path(args.data_dir)
    edr_dir = data_dir / "images_op1" / "edr_op1"
    label_dir = data_dir / "labels_op1" / "train_op1"

    image_paths = sorted(str(p) for p in edr_dir.iterdir() if p.is_file())
    label_paths = sorted(str(p) for p in label_dir.iterdir() if p.is_file())
    assert len(image_paths) == len(label_paths), (
        "Mismatch in file counts — has data_refiner.py been run on this --data-dir?"
    )

    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths,
        test_size=0.2, random_state=args.seed,
    )
    print(f"Training on {len(train_images)} images, validating on {len(val_images)} images.")

    train_transform = get_transform(args.augmentation)
    train_dataset = MSL_train_preprocessor(
        image_paths=train_images, label_paths=train_labels,
        transform=train_transform,
        input_hw=args.input_hw,
    )
    val_dataset = MSL_train_preprocessor(
        image_paths=val_images, label_paths=val_labels,
        # validation is never augmented
        input_hw=args.input_hw,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print("DataLoaders are ready.")

    # ==========================================
    # PART-2: MODEL
    # ==========================================
    print(f"Initializing DeepLabV3+ with {args.encoder} backbone (weights={args.encoder_weights})...")
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=args.num_classes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ==========================================
    # PART-3: LOSS (delegated to losses.get_loss)
    # ==========================================
    weights_list = None
    weights_meta = None
    if args.class_weights:
        weights_list, weights_meta = load_class_weights(args.class_weights, args.num_classes)

    criterion = get_loss(
        args.loss,
        num_classes=args.num_classes,
        class_weights=weights_list,
        focal_gamma=args.focal_gamma,
        focal_dice_alpha=args.focal_dice_alpha,
        focal_tversky_alpha=args.focal_tversky_alpha,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        ftl_gamma=args.ftl_gamma,
        device=device,
    )
    print("Using " + describe_loss(
        args.loss,
        focal_gamma=args.focal_gamma,
        focal_dice_alpha=args.focal_dice_alpha,
        focal_tversky_alpha=args.focal_tversky_alpha,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        ftl_gamma=args.ftl_gamma,
        has_class_weights=weights_list is not None,
    ))
    if weights_list is not None:
        print(f"  class weights: {weights_list}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ==========================================
    # PART-4: SAVE CONFIG
    # ==========================================
    config = {
        "exp_id": args.exp_id,
        "description": args.description,
        "data": {
            "data_dir": args.data_dir,
            "num_pairs": len(image_paths),
            "split": {"train_frac": 0.8, "val_frac": 0.2, "random_state": args.seed},
            "input_hw": args.input_hw,
        },
        "model": {
            "arch": "smp.DeepLabV3Plus",
            "encoder_name": args.encoder,
            "encoder_weights": args.encoder_weights,
            "in_channels": 3,
            "classes": args.num_classes,
        },
        "training": {
            "loss_type": args.loss,
            "loss_summary": describe_loss(
                args.loss,
                focal_gamma=args.focal_gamma,
                focal_dice_alpha=args.focal_dice_alpha,
                focal_tversky_alpha=args.focal_tversky_alpha,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
                ftl_gamma=args.ftl_gamma,
                has_class_weights=weights_list is not None,
            ),
            "focal_gamma": args.focal_gamma if args.loss in {"focal", "focal_dice", "focal_tversky"} else None,
            "focal_dice_alpha": args.focal_dice_alpha if args.loss == "focal_dice" else None,
            "focal_tversky_alpha": args.focal_tversky_alpha if args.loss == "focal_tversky" else None,
            "tversky_alpha": args.tversky_alpha if args.loss in {"focal_tversky", "ftl"} else None,
            "tversky_beta": args.tversky_beta if args.loss in {"focal_tversky", "ftl"} else None,
            "ftl_gamma": args.ftl_gamma if args.loss == "ftl" else None,
            "optimizer": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "augmentation": args.augmentation,
            "class_weights": {
                "source": args.class_weights,
                "values": weights_list,
                "meta": weights_meta,
            } if weights_list else "none",
            "resampling": "none",
            "early_stop_patience": args.early_stop_patience,
        },
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ==========================================
    # PART-5: TRAINING LOOP
    # ==========================================
    header = build_csv_header(CLASS_NAMES)
    with open(history_file, "w", newline="") as f:
        csv.writer(f).writerow(header)

    print(f"Starting training on {device}, outputs -> {exp_dir}")
    print(f"  per-epoch metrics: pixel_acc, macro precision/recall/F1, mIoU, "
          f"per-class IoU+F1 for {CLASS_NAMES} -> {history_file.name}")
    print(f"  per-epoch checkpoints -> {checkpoints_dir.name}/epoch_NN.pth")
    if args.early_stop_patience > 0:
        print(f"  early stopping: patience={args.early_stop_patience} epochs on val_loss")

    # Early-stop bookkeeping. Tracks lowest val_loss seen and how many epochs
    # have passed since the last improvement; on patience exhaustion we break
    # out of the loop. Final-epoch selection still happens post-hoc via
    # select_best.py — early stopping just bounds wasted compute on a clearly
    # overfit run.
    best_val_loss = float("inf")
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        # ----- train -----
        model.train()
        running_loss = 0.0
        train_cm = torch.zeros(args.num_classes, args.num_classes,
                               dtype=torch.int64, device=device)
        loop = tqdm(train_loader, leave=False, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            update_confusion_matrix_torch(train_cm, preds, labels,
                                          num_classes=args.num_classes)
            loop.set_postfix(loss=loss.item())
        train_loss_avg = running_loss / len(train_loader)
        train_metrics = compute_metrics(train_cm.cpu().numpy())

        # ----- validate -----
        model.eval()
        val_running = 0.0
        val_cm = torch.zeros(args.num_classes, args.num_classes,
                             dtype=torch.int64, device=device)
        with torch.no_grad():
            val_loop = tqdm(val_loader, leave=False, desc=f"Epoch [{epoch+1}/{args.epochs}] Val")
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_running += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                update_confusion_matrix_torch(val_cm, preds, labels,
                                              num_classes=args.num_classes)
        val_loss_avg = val_running / len(val_loader)
        val_metrics = compute_metrics(val_cm.cpu().numpy())

        # Console summary uses the headline numbers (mirrors thesis-style reporting).
        big_rock_idx = CLASS_NAMES.index("big_rock") if "big_rock" in CLASS_NAMES else -1
        train_br_iou = train_metrics["iou"][big_rock_idx] if big_rock_idx >= 0 else float("nan")
        val_br_iou = val_metrics["iou"][big_rock_idx] if big_rock_idx >= 0 else float("nan")
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: loss={train_loss_avg:.4f} acc={train_metrics['pixel_acc']:.4f} "
            f"mIoU={train_metrics['miou']:.4f} macro_F1={train_metrics['macro_f1']:.4f} "
            f"BR_IoU={train_br_iou:.4f} | "
            f"Val: loss={val_loss_avg:.4f} acc={val_metrics['pixel_acc']:.4f} "
            f"mIoU={val_metrics['miou']:.4f} macro_F1={val_metrics['macro_f1']:.4f} "
            f"BR_IoU={val_br_iou:.4f}"
        )

        row = epoch_row(epoch + 1, train_loss_avg, train_metrics,
                        val_loss_avg, val_metrics, CLASS_NAMES)
        with open(history_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

        ckpt_path = checkpoints_dir / f"epoch_{epoch + 1:02d}.pth"
        torch.save(model.state_dict(), ckpt_path)

        if args.early_stop_patience > 0:
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= args.early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}: val_loss has not "
                        f"improved for {args.early_stop_patience} epochs "
                        f"(best={best_val_loss:.4f})."
                    )
                    break

    print(
        f"Training complete. Per-epoch checkpoints in {checkpoints_dir}. "
        f"Run `select_best.py --exp-id {args.exp_id} --metric <metric>` to "
        f"pick the best-epoch weights."
    )


if __name__ == "__main__":
    main()
