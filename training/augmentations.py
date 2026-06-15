"""Albumentations pipelines for AI4Mars training.

Each named pipeline returns an `A.Compose` callable that takes
`image=<HxWx3 uint8>, mask=<HxW uint8 with 255=ignore>` and returns a dict with
`image` (3xHxW float32 tensor in [0,1]) and `mask` (HxW long tensor).

The "none" name corresponds to RQ1/baseline runs — no augmentation, just
normalization and tensor conversion. RQ2 experiments compare it to "basic".
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


_TENSOR_TAIL = [
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
]


_AFFINE_KW = dict(
    interpolation=cv2.INTER_LINEAR,
    mask_interpolation=cv2.INTER_NEAREST,
    fill=0,
    fill_mask=255,
)


def _none():
    return A.Compose(_TENSOR_TAIL)


def _basic():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=0, p=0.3
            ),
            A.Affine(
                rotate=(-8, 8),
                translate_percent=(0.0, 0.05),
                p=0.3,
                **_AFFINE_KW,
            ),
        ]
        + _TENSOR_TAIL
    )


def _strong():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=0, p=0.5
            ),
            A.Affine(
                rotate=(-12, 12),
                translate_percent=(0.0, 0.08),
                scale=(0.85, 1.15),
                p=0.5,
                **_AFFINE_KW,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.3,
            ),
        ]
        + _TENSOR_TAIL
    )


PIPELINES = {
    "none": _none,
    "basic": _basic,
    "strong": _strong,
}


def get_transform(name):
    """Return an Albumentations Compose for the named pipeline.

    Returns None for "none" / None / empty so the dataset's inline
    no-transform path stays bit-identical to historical baseline runs.
    """
    if name in (None, "", "none"):
        return None
    if name not in PIPELINES:
        raise ValueError(
            f"Unknown augmentation pipeline: {name!r}. "
            f"Available: {sorted(PIPELINES)}"
        )
    return PIPELINES[name]()
