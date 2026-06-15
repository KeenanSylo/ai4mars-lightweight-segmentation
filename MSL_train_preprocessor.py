"""Training-time dataset for AI4Mars MSL NCAM.

Pairs each EDR image with its label. The training labels in
`MSL_NAVCAM_TRAINING_SET/labels_op1/train_op1/` already have the rover (mxy)
and range (rng-30m) exclusions baked in as `IGNORE_INDEX=255`, so labels are
loaded as-is here.

Yields `(image, label)`:
    image: torch.float32, shape (3, H, W), values in [0, 1]
    label: torch.int64,   shape (H, W), values in {0..NUM_CLASSES-1, 255}

Optional knobs:
    transform: an Albumentations `Compose` (see `training/augmentations.py`).
        When provided, it is called as `transform(image=<uint8 HxWx3>,
        mask=<uint8 HxW>)`. Pipelines from `training/augmentations.py` end in
        `ToFloat + ToTensorV2`, so the returned image is already a float32
        CHW tensor; the mask is still uint8 HW and is cast to int64 here.
    input_hw: square side length. None = native resolution. Image is bilinear-
        resized; label is nearest-resized so class indices and the 255 ignore
        region are preserved.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MSL_train_preprocessor(Dataset):
    def __init__(self,
                 image_paths,
                 label_paths,
                 transform=None,
                 input_hw=None):
        assert len(image_paths) == len(label_paths), (
            f"Mismatched lengths: images={len(image_paths)} labels={len(label_paths)}"
        )
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.transform = transform
        self.input_hw = input_hw

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Could not read label: {self.label_paths[idx]}")

        if self.input_hw is not None:
            size = (self.input_hw, self.input_hw)
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            out = self.transform(image=image, mask=label)
            image_t = out["image"]
            mask_t = out["mask"]
            if not torch.is_tensor(mask_t):
                mask_t = torch.from_numpy(np.asarray(mask_t))
            return image_t, mask_t.long()

        image_t = torch.from_numpy(
            np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        )
        label_t = torch.from_numpy(label.astype(np.int64))
        return image_t, label_t
