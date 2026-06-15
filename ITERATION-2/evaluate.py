# Reconstructed from ./__pycache__/evaluate.cpython-312.pyc
# via PyLingual (https://pylingual.io). Internal filename in bytecode header:
# './evaluate.py'.
# Bytecode version: 3.12.0rc2 (3531). Source timestamp: 2026-05-14 01:51:51 UTC.
#
# Two functions were flagged by the decompiler as "Different control flow" and
# had to be hand-reconstructed against the saved JSON shape in
# ITERATION-2/experiments/003_fpn_512/evaluation_results.json:
#   - ConfidenceAccumulator.to_dict
#   - cm_to_jsonable
# All other functions decompiled cleanly; only cosmetic fixes were applied
# (`!=` spacing, one indentation level on the upsample branch).

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from metrics import (
    NUM_CLASSES,
    IGNORE_INDEX,
    update_confusion_matrix,
    compute_metrics,
    format_report,
    CLASS_NAMES,
)


CONFIDENCE_THRESHOLD_DEFAULT = 0.6
ARCH_MAP = {
    'smp.Unet': smp.Unet,
    'smp.DeepLabV3Plus': smp.DeepLabV3Plus,
    'smp.Segformer': smp.Segformer,
    'smp.FPN': smp.FPN,
}


class MSLGoldTestDataset(Dataset):
    """Pairs each gold-test label with its EDR image.

    Test labels are named <prefix>_merged.png and EDR images are <prefix>.JPG.
    Rover-mask and range-mask exclusions are already baked into the merged
    labels as 255, so no extra masks are applied here.
    """

    def __init__(self, label_dir: Path, edr_dir: Path, skip_fully_ignored: bool = True):
        self.pairs = []
        self.skipped_no_edr = 0
        self.skipped_all_ignore = 0
        for label_file in sorted(label_dir.glob('*.png')):
            prefix = label_file.name.replace('_merged.png', '')
            edr_matches = list(edr_dir.glob(f'{prefix}.*'))
            if not edr_matches:
                self.skipped_no_edr += 1
                continue
            if skip_fully_ignored:
                lbl = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
                if (lbl == IGNORE_INDEX).all():
                    self.skipped_all_ignore += 1
                    continue
            self.pairs.append((edr_matches[0], label_file))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        edr_path, label_path = self.pairs[idx]
        image = cv2.imread(str(edr_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(image), torch.from_numpy(label.astype(np.int64))


def build_model(weights_path: Path, device: torch.device,
                arch: str = 'smp.Unet', encoder_name: str = 'mobilenet_v2'):
    """Build a segmentation model from a saved checkpoint.

    `arch` is the smp class name as recorded in config.json's `model.arch` field
    ("smp.Unet" / "smp.DeepLabV3Plus" / "smp.Segformer"). Defaults to "smp.Unet"
    for backward compatibility with older experiments that didn't record arch.
    """
    if arch not in ARCH_MAP:
        raise SystemExit(
            f'Unknown arch: {arch}. Add it to ARCH_MAP in evaluate.py. '
            f'Available: {sorted(ARCH_MAP)}'
        )
    cls = ARCH_MAP[arch]
    model = cls(encoder_name=encoder_name, encoder_weights=None,
                in_channels=3, classes=NUM_CLASSES)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class ConfidenceAccumulator:
    """Tracks per-pixel max-softmax confidence across an evaluation pass.

    Reports:
      - Overall fraction of valid pixels with max(softmax) < threshold
        (the "stop the rover" signal per the ITERATION-2 reliability constraint)
      - Per-class breakdown (especially big_rock — the safety class)
      - Mean confidence overall and per-class
      - Histogram of max-confidence values
    """

    def __init__(self, num_classes=NUM_CLASSES,
                 threshold=CONFIDENCE_THRESHOLD_DEFAULT,
                 n_bins=20, ignore_index=IGNORE_INDEX):
        self.num_classes = num_classes
        self.threshold = float(threshold)
        self.ignore_index = ignore_index
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.histogram = np.zeros(n_bins, dtype=np.int64)
        self.per_class_count = np.zeros(num_classes, dtype=np.int64)
        self.per_class_conf_sum = np.zeros(num_classes, dtype=np.float64)
        self.per_class_below = np.zeros(num_classes, dtype=np.int64)

    def update(self, max_conf_np, labels_np):
        valid = labels_np != self.ignore_index
        conf = max_conf_np[valid]
        lbl = labels_np[valid]
        if conf.size == 0:
            return
        idx = np.clip(np.digitize(conf, self.bin_edges) - 1, 0, len(self.histogram) - 1)
        np.add.at(self.histogram, idx, 1)
        for c in range(self.num_classes):
            mask = lbl == c
            if not mask.any():
                continue
            conf_c = conf[mask]
            self.per_class_count[c] += int(conf_c.size)
            self.per_class_conf_sum[c] += float(conf_c.sum())
            self.per_class_below[c] += int((conf_c < self.threshold).sum())

    def to_dict(self):
        # Hand-reconstructed: pylingual flagged this with "Different control flow"
        # and produced scrambled key/value pairs. Restored from the 003 saved JSON.
        total = int(self.per_class_count.sum())
        below = int(self.per_class_below.sum())
        conf_total = float(self.per_class_conf_sum.sum())
        denom = max(total, 1)
        return {
            'threshold': self.threshold,
            'overall_low_confidence_fraction': below / denom,
            'overall_mean_confidence': conf_total / denom,
            'per_class_low_confidence_fraction': {
                CLASS_NAMES[c]: float(self.per_class_below[c]) / max(int(self.per_class_count[c]), 1)
                for c in range(self.num_classes)
            },
            'per_class_mean_confidence': {
                CLASS_NAMES[c]: float(self.per_class_conf_sum[c]) / max(int(self.per_class_count[c]), 1)
                for c in range(self.num_classes)
            },
            'per_class_pixel_count': {
                CLASS_NAMES[c]: int(self.per_class_count[c])
                for c in range(self.num_classes)
            },
            'histogram_bin_edges': self.bin_edges.tolist(),
            'histogram_counts': self.histogram.tolist(),
        }


def evaluate_on_loader(model, loader, device,
                       confidence_threshold=CONFIDENCE_THRESHOLD_DEFAULT,
                       eval_input_hw=None):
    """Evaluate `model` on `loader`'s gold-test pairs.

    If `eval_input_hw` is set (Strategy A for sub-native-resolution models):
      1. Resize the input image to (eval_input_hw, eval_input_hw) bilinear.
      2. Forward at the model's training resolution.
      3. Upsample the logits back to the gold label's native resolution
         (bilinear, align_corners=False) BEFORE argmax / softmax.
      4. Compare 1024² predictions against 1024² gold labels.

    This makes IoU comparable across runs trained at different resolutions,
    by paying the upsampling cost the deployment would actually pay.
    """
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_acc = ConfidenceAccumulator(threshold=confidence_threshold)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='eval', leave=False):
            images = images.to(device, non_blocking=True)
            target_h, target_w = labels.shape[-2:]
            if eval_input_hw is not None:
                images = F.interpolate(images, size=(eval_input_hw, eval_input_hw),
                                       mode='bilinear', align_corners=False)
            outputs = model(images)
            if eval_input_hw is not None and (
                outputs.shape[-2] != target_h or outputs.shape[-1] != target_w
            ):
                outputs = F.interpolate(outputs, size=(target_h, target_w),
                                        mode='bilinear', align_corners=False)
            preds = outputs.argmax(dim=1).cpu().numpy()
            max_conf = F.softmax(outputs, dim=1).max(dim=1).values.cpu().numpy()
            labels_np = labels.numpy()
            update_confusion_matrix(cm, preds, labels_np)
            conf_acc.update(max_conf, labels_np)
    return cm, conf_acc


def cm_to_jsonable(cm, conf_acc=None):
    # Hand-reconstructed: pylingual flagged this with "Different control flow"
    # and produced scrambled key/value pairs. Restored from the 003 saved JSON.
    m = compute_metrics(cm)
    out = {
        'confusion_matrix': cm.tolist(),
        'class_names': CLASS_NAMES,
        'pixel_accuracy': m['pixel_acc'],
        'mean_iou': m['miou'],
        'per_class_iou': {n: float(m['iou'][i]) for i, n in enumerate(CLASS_NAMES)},
        'per_class_precision': {n: float(m['precision'][i]) for i, n in enumerate(CLASS_NAMES)},
        'per_class_recall': {n: float(m['recall'][i]) for i, n in enumerate(CLASS_NAMES)},
        'per_class_support': {n: int(m['support'][i]) for i, n in enumerate(CLASS_NAMES)},
    }
    if conf_acc is not None:
        out['confidence'] = conf_acc.to_dict()
    return out


def resolve_paths(args):
    """Return (weights_path, output_path, arch, encoder_name, eval_input_hw) from args.

    If --exp-id is given, paths point into <exp-root>/<exp-id>/ and `arch`,
    `encoder_name`, and `eval_input_hw` are read from that experiment's
    config.json by default (overridable via CLI flags). Otherwise fall back
    to --weights / --output / --arch / --encoder / --input-hw.
    """
    if args.exp_id:
        exp_dir = Path(args.exp_root) / args.exp_id
        weights = exp_dir / 'weights.pth'
        output = exp_dir / 'evaluation_results.json'
        arch = args.arch
        encoder = args.encoder
        eval_input_hw = args.input_hw
        cfg_path = exp_dir / 'config.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            if arch is None:
                arch = cfg.get('model', {}).get('arch', 'smp.Unet')
            if encoder is None:
                encoder = cfg.get('model', {}).get('encoder_name', 'mobilenet_v2')
            if eval_input_hw is None:
                eval_input_hw = cfg.get('data', {}).get('input_hw')
        if arch is None:
            arch = 'smp.Unet'
        if encoder is None:
            encoder = 'mobilenet_v2'
        if not weights.exists():
            raise SystemExit(f'No weights at {weights}. Train first or check --exp-id.')
        return weights, output, arch, encoder, eval_input_hw
    if not args.weights:
        raise SystemExit('Provide either --exp-id or --weights.')
    return (Path(args.weights), Path(args.output),
            args.arch or 'smp.Unet', args.encoder or 'mobilenet_v2', args.input_hw)


def main():
    ap = argparse.ArgumentParser(
        description='Evaluate a trained segmentation model on the AI4Mars gold test set.'
    )
    ap.add_argument('--exp-id', help='Evaluate <exp-root>/<exp-id>/weights.pth and save to that folder.')
    ap.add_argument('--exp-root', default='ITERATION-2/experiments',
                    help="Parent folder containing the experiment. Defaults to the "
                         "current iteration's experiments/ folder.")
    ap.add_argument('--weights', help='Path to a .pth file (used only if --exp-id not given).')
    ap.add_argument('--output', default='evaluation_results.json',
                    help='Output JSON path (used only if --exp-id not given).')
    ap.add_argument('--encoder', help='smp encoder name. If --exp-id is given, read from its config.json by default.')
    ap.add_argument('--arch', help='smp architecture class name (smp.Unet / smp.DeepLabV3Plus / smp.Segformer). '
                                   'If --exp-id is given, read from its config.json by default.')
    ap.add_argument('--variants', nargs='+',
                    default=['masked-gold-min1-100agree',
                            'masked-gold-min2-100agree',
                            'masked-gold-min3-100agree'])
    ap.add_argument('--data-root', default='data/msl/ncam',
                    help='Root containing the AI4Mars MSL NCAM release '
                         '(images/edr, images/mxy, images/rng-30m, labels/test).')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD_DEFAULT,
                    help='Per-pixel max-softmax threshold below which the rover would stop '
                         '(ITERATION-2 reliability constraint). Default 0.6.')
    ap.add_argument('--input-hw', type=int, default=None,
                    help='Strategy-A evaluation resolution: resize EDR to this square before '
                         'inference, then upsample logits to the native gold-label resolution '
                         'before argmax/IoU. None = predict at native resolution. If --exp-id '
                         'is given, defaults to config.json\'s data.input_hw (so a model '
                         'trained at 512² is automatically evaluated at 512² with upsampled output).')
    args = ap.parse_args()

    weights_path, output_path, arch, encoder_name, eval_input_hw = resolve_paths(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading weights from {weights_path}  (arch: {arch}, encoder: {encoder_name})')
    if eval_input_hw is not None:
        print(f'Evaluation input resolution: {eval_input_hw}×{eval_input_hw} '
              f'(Strategy A: predict at {eval_input_hw}², upsample logits to gold-label native size).')
    else:
        print('Evaluation input resolution: native (no downsample).')

    model = build_model(weights_path, device, arch=arch, encoder_name=encoder_name)
    edr_dir = Path(args.data_root) / 'images' / 'edr'
    test_root = Path(args.data_root) / 'labels' / 'test'
    results = {
        'exp_id': args.exp_id,
        'weights': str(weights_path),
        'arch': arch,
        'encoder': encoder_name,
        'confidence_threshold': args.confidence_threshold,
        'eval_input_hw': eval_input_hw,
        'eval_strategy': ('A: resize EDR to eval_input_hw, predict, upsample logits to '
                          'native label size') if eval_input_hw is not None
                         else 'native: predict and compare at gold-label native resolution',
        'variants': {},
    }

    for variant in args.variants:
        label_dir = test_root / variant
        if not label_dir.is_dir():
            print(f'  [skip] {variant}: directory not found')
            continue
        ds = MSLGoldTestDataset(label_dir, edr_dir)
        print(f'\n--- {variant}: usable={len(ds)}  skipped_no_edr={ds.skipped_no_edr}  '
              f'skipped_all_ignore={ds.skipped_all_ignore}')
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == 'cuda'))
        cm, conf_acc = evaluate_on_loader(
            model, loader, device,
            confidence_threshold=args.confidence_threshold,
            eval_input_hw=eval_input_hw,
        )
        print(format_report(cm, title=variant))
        cdict = conf_acc.to_dict()
        print(f"  confidence < {args.confidence_threshold:.2f}: "
              f"{100 * cdict['overall_low_confidence_fraction']:.2f}% of valid pixels  "
              f"(mean confidence overall: {cdict['overall_mean_confidence']:.4f})")
        results['variants'][variant] = cm_to_jsonable(cm, conf_acc=conf_acc)
        results['variants'][variant]['num_images'] = len(ds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f'\nSaved results to {output_path}')


if __name__ == '__main__':
    main()
