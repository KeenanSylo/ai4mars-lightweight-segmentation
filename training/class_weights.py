"""Compute balanced per-class weights from AI4Mars training labels.

Balanced (sklearn-style) weights:  w_c = N / (K * n_c)
where N = total valid pixels, K = number of classes, n_c = pixel count for class c.
Mean weight = 1.0 by construction.

Rover (mxy) and range (rng-30m) masks are applied first to match the
training-time MSL_train_preprocessor logic, so weights reflect the distribution of
pixels the loss actually sees.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]


def _scan_one(triple, num_classes=4, ignore_index=255):
    label_path, mxy_path, rng_path = triple
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    rover = cv2.imread(mxy_path, cv2.IMREAD_GRAYSCALE)
    rng_im = cv2.imread(rng_path, cv2.IMREAD_GRAYSCALE)
    mask = (rover > 0) | (rng_im > 0)
    label[mask] = ignore_index
    valid = label[label != ignore_index]
    return np.bincount(valid, minlength=num_classes)[:num_classes].astype(np.int64)


def compute_class_pixel_counts(label_paths, mxy_paths, rng_paths,
                               num_classes=4, n_workers=8):
    triples = [(str(a), str(b), str(c)) for a, b, c in zip(label_paths, mxy_paths, rng_paths)]
    counts = np.zeros(num_classes, dtype=np.int64)
    if n_workers <= 1:
        for t in tqdm(triples, desc="scanning labels"):
            counts += _scan_one(t, num_classes=num_classes)
        return counts
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for c in tqdm(ex.map(_scan_one, triples, chunksize=8),
                      total=len(triples), desc="scanning labels"):
            counts += c
    return counts


def balanced_weights_from_counts(counts, num_classes=None):
    counts = np.asarray(counts, dtype=np.float64)
    K = num_classes if num_classes is not None else len(counts)
    total = counts.sum()
    weights = total / (K * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="MSL_NAVCAM_TRAINING_SET")
    ap.add_argument("--output", default="MSL_NAVCAM_TRAINING_SET/class_weights_train.json")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    label_paths = sorted((data_dir / "labels_op1" / "train_op1").iterdir())
    mxy_paths = sorted((data_dir / "images_op1" / "mxy_op1").iterdir())
    rng_paths = sorted((data_dir / "images_op1" / "rng-30m_op1").iterdir())
    assert len(label_paths) == len(mxy_paths) == len(rng_paths), \
        f"Mismatch: labels={len(label_paths)} mxy={len(mxy_paths)} rng={len(rng_paths)}"
    print(f"Scanning {len(label_paths)} labels with {args.n_workers} workers...")

    counts = compute_class_pixel_counts(
        label_paths, mxy_paths, rng_paths,
        num_classes=args.num_classes, n_workers=args.n_workers,
    )
    weights = balanced_weights_from_counts(counts, num_classes=args.num_classes)

    names = CLASS_NAMES[:args.num_classes]
    total = int(counts.sum())
    out = {
        "data_dir": args.data_dir,
        "num_labels_scanned": len(label_paths),
        "total_valid_pixels": total,
        "class_names": names,
        "pixel_counts": {n: int(counts[i]) for i, n in enumerate(names)},
        "fractions": {n: float(counts[i] / total) for i, n in enumerate(names)},
        "balanced_weights": {n: float(weights[i]) for i, n in enumerate(names)},
        "weights_list": [float(w) for w in weights],
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
