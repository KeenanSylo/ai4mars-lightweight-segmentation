"""Loss functions for AI4Mars training.

Mirrors the structure of `augmentations.py`: a `LOSSES` set listing the
available choices and a `get_loss(name, ...)` factory that returns a torch
module callable as `criterion(logits, targets)`. The trainer's `--loss` flag
uses `sorted(LOSSES)` for its argparse `choices`, so adding a new loss here
exposes it on the CLI automatically.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


IGNORE_INDEX = 255

LOSSES = {"ce", "focal", "focal_dice", "focal_tversky", "ftl"}


class FocalDiceLoss(nn.Module):
    """Convex combination α · focal + (1 − α) · multiclass-Dice.

    Standard combo for rare-class semantic segmentation: focal provides
    pixel-level hard-example focusing, Dice provides region-level overlap
    pressure that the macro average weights small classes equally to large
    ones. α tunes the balance: α=0.5 is the standard half-and-half mix
    (gradient-equivalent to focal + dice up to scale), α→1 collapses to
    pure focal, α→0 to pure Dice.
    """

    def __init__(self, *, gamma=2.0, alpha=0.5, ignore_index=IGNORE_INDEX):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.focal = smp.losses.FocalLoss(
            mode="multiclass", gamma=gamma, ignore_index=ignore_index,
        )
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", ignore_index=ignore_index,
        )

    def forward(self, logits, targets):
        return (
            self.alpha * self.focal(logits, targets)
            + (1.0 - self.alpha) * self.dice(logits, targets)
        )


class FocalTverskyLoss(nn.Module):
    """Convex combination α · focal + (1 − α) · Tversky.

    Sum-style combo analogous to FocalDiceLoss but with Dice replaced by
    Tversky, which adds asymmetric FP/FN weighting (tversky_alpha / beta).
    For rare classes, beta > alpha penalises FNs more, pushing recall up.

    NOT the Abraham & Khan (2018) Focal Tversky Loss — that's `"ftl"` in
    this module, where the focal exponent is applied to (1 − TI) directly
    (single loss, not a sum). This class instead keeps focal and Tversky
    as separate terms so the pixel-level focal signal is preserved.
    """

    def __init__(self, *, gamma=2.0, alpha=0.5, tversky_alpha=0.3,
                 tversky_beta=0.7, ignore_index=IGNORE_INDEX):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.focal = smp.losses.FocalLoss(
            mode="multiclass", gamma=gamma, ignore_index=ignore_index,
        )
        self.tversky = smp.losses.TverskyLoss(
            mode="multiclass",
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
        )

    def forward(self, logits, targets):
        return (
            self.alpha * self.focal(logits, targets)
            + (1.0 - self.alpha) * self.tversky(logits, targets)
        )


def get_loss(name,
             *,
             num_classes=4,
             ignore_index=IGNORE_INDEX,
             class_weights=None,
             focal_gamma=2.0,
             focal_dice_alpha=0.5,
             focal_tversky_alpha=0.5,
             tversky_alpha=0.3,
             tversky_beta=0.7,
             ftl_gamma=4.0 / 3.0,
             device=None):
    """Build a loss function.

    Args:
        name: one of `LOSSES`.
        num_classes: number of classes (kept for symmetry / future losses).
        ignore_index: pixel value excluded from loss.
        class_weights: optional iterable of per-class weights. Only valid for
            `name="ce"`. Will be moved to `device` if provided.
        focal_gamma: focusing parameter γ. Used when `name` is `"focal"` or
            `"focal_dice"`.
        focal_dice_alpha: mixing weight α for `"focal_dice"` only.
        tversky_alpha: Tversky FP weight (smaller → less FP penalty). Used
            when `name="ftl"`. Paper default 0.3 for rare-class segmentation.
        tversky_beta: Tversky FN weight (larger → more FN penalty, helps
            recall on rare classes). Used when `name="ftl"`. Paper default 0.7.
        ftl_gamma: focal exponent applied to (1 − TverskyIndex). Used when
            `name="ftl"`. Paper default 4/3.
        device: torch device for moving the class-weights tensor onto.

    Returns:
        A `torch.nn.Module` callable as `criterion(logits, targets)`.

    Raises:
        ValueError if `name` is unknown, or if `class_weights` is given with
        a loss that does not support per-class weighting.
    """
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name!r}. Available: {sorted(LOSSES)}")

    if name == "ce":
        if class_weights is not None:
            weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
            if device is not None:
                weight_tensor = weight_tensor.to(device)
            return nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight_tensor)
        return nn.CrossEntropyLoss(ignore_index=ignore_index)

    if name == "focal":
        if class_weights is not None:
            raise ValueError(
                "class_weights is not supported with focal loss. "
                "smp.losses.FocalLoss takes a single scalar alpha, not per-class weights. "
                "Pick one or the other."
            )
        return smp.losses.FocalLoss(
            mode="multiclass",
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )

    if name == "focal_dice":
        if class_weights is not None:
            raise ValueError(
                "class_weights is not supported with focal_dice loss. "
                "The focal component takes a single scalar weighting, not per-class weights."
            )
        return FocalDiceLoss(
            gamma=focal_gamma,
            alpha=focal_dice_alpha,
            ignore_index=ignore_index,
        )

    if name == "focal_tversky":
        if class_weights is not None:
            raise ValueError(
                "class_weights is not supported with focal_tversky loss. "
                "Class balance is controlled via tversky alpha/beta."
            )
        return FocalTverskyLoss(
            gamma=focal_gamma,
            alpha=focal_tversky_alpha,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            ignore_index=ignore_index,
        )

    if name == "ftl":
        # Focal Tversky Loss (Abraham & Khan, 2018):
        #   FTL = (1 − TverskyIndex) ** gamma
        # smp.losses.TverskyLoss applies the gamma exponent natively. Class
        # balance is controlled via the Tversky alpha/beta knobs (FP/FN
        # weighting), not via per-class weights — paper config for rare-class
        # segmentation is alpha=0.3, beta=0.7, gamma=4/3.
        if class_weights is not None:
            raise ValueError(
                "class_weights is not supported with ftl loss. "
                "Use --tversky-alpha / --tversky-beta to control class balance instead."
            )
        return smp.losses.TverskyLoss(
            mode="multiclass",
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=ftl_gamma,
            ignore_index=ignore_index,
        )

    raise ValueError(f"Unknown loss: {name!r}")  # unreachable


def describe_loss(name, *, focal_gamma=2.0, focal_dice_alpha=0.5,
                  focal_tversky_alpha=0.5,
                  tversky_alpha=0.3, tversky_beta=0.7, ftl_gamma=4.0 / 3.0,
                  has_class_weights=False, ignore_index=IGNORE_INDEX):
    """Short, human-readable summary string suitable for config.json."""
    if name == "ce":
        suffix = " + class weights" if has_class_weights else ""
        return f"CrossEntropyLoss(ignore_index={ignore_index}){suffix}"
    if name == "focal":
        return f"FocalLoss(gamma={focal_gamma}, ignore_index={ignore_index})"
    if name == "focal_dice":
        return (
            f"{focal_dice_alpha:.2f}*FocalLoss(gamma={focal_gamma}) + "
            f"{1.0 - focal_dice_alpha:.2f}*DiceLoss(multiclass), "
            f"ignore_index={ignore_index}"
        )
    if name == "focal_tversky":
        return (
            f"{focal_tversky_alpha:.2f}*FocalLoss(gamma={focal_gamma}) + "
            f"{1.0 - focal_tversky_alpha:.2f}*TverskyLoss(alpha={tversky_alpha}, "
            f"beta={tversky_beta}), ignore_index={ignore_index}"
        )
    if name == "ftl":
        return (
            f"FocalTverskyLoss-Abraham(alpha={tversky_alpha}, beta={tversky_beta}, "
            f"gamma={ftl_gamma:.4f}, ignore_index={ignore_index})"
        )
    raise ValueError(f"Unknown loss: {name!r}")
