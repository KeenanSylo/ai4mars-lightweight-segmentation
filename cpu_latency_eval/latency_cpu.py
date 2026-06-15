"""SGC #2 (latency) — CPU re-measurement harness.

Thesis §4.3 specifies: "single CPU thread under FP32 PyTorch with
torch.set_num_threads(1)" with a discarded warm-up pass. The legacy
sgc_evaluate_{GOOD,BAD}_2.0.py harnesses time inference on whichever
device is available (GPU on this host), which contradicts the spec.

This script measures latency on CPU only, single-threaded FP32, with a
warm-up discard, and writes results to sgc_cpu_latency/results/.
It is self-contained: no imports from sgc_evaluate_*.py. The metrics
module is imported only for class constants (NUM_CLASSES, IGNORE_INDEX);
no IoU is recomputed here.

Run from SPACE_PROJECT root, e.g.:
    python sgc_cpu_latency/latency_cpu.py \\
        --mode baseline \\
        --weights ML_CHOICE/v2_R11_MNv4-S_512.pth \\
        --input-hw 512 \\
        --run-name v2_R11_MNv4-S_512_baseline

Modes: baseline | unshielded_chaos | shielded_chaos
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Allow running from anywhere (e.g. nohup from project root) and still
# locate metrics.py at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import NUM_CLASSES, IGNORE_INDEX


DEFAULT_DATA_ROOT = PROJECT_ROOT / "MSL_NAVCAM_TEST_SET"
DEFAULT_VARIANT = "masked-gold-min3-100agree_op1"
DEFAULT_ENCODER = "tu-mobilenetv4_conv_small"
LATENCY_CAP_MS = 500


class MSLGoldTestDataset(Dataset):
    """Copy of the loader from sgc_evaluate_*.py — kept self-contained on purpose."""

    def __init__(self, label_dir: Path, edr_dir: Path, skip_fully_ignored: bool = True):
        self.pairs = []
        self.skipped_no_edr = 0
        self.skipped_all_ignore = 0
        for label_file in sorted(label_dir.glob("*.png")):
            prefix = label_file.name.replace("_merged.png", "")
            edr_matches = list(edr_dir.glob(f"{prefix}.*"))
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
                encoder_name: str = DEFAULT_ENCODER) -> torch.nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --- Chaos injector — copied from sgc_evaluate_BAD_2.0.py (deepest-feature +
# non-zero-index patches included). Identical semantics; reproduced verbatim
# rather than imported to keep this script standalone.
class ChaoticSpaceRadiationInjector:
    def __init__(self, probability=1.0, max_flips=5):
        self.probability = probability
        self.max_flips = max_flips
        self.strike_count = 0

    def __call__(self, module, module_in, module_out):
        if random.random() > self.probability:
            return module_out

        is_list = isinstance(module_out, (list, tuple))
        if is_list:
            corrupted_out = list(module_out)
            list_idx = len(corrupted_out) - 1
            target_tensor = corrupted_out[list_idx].clone().detach().contiguous()
        else:
            target_tensor = module_out.clone().detach().contiguous()

        raw_bits = target_tensor.view(torch.int32)
        flat_bits = raw_bits.view(-1)
        float_view = target_tensor.view(-1)

        num_flips = random.randint(1, self.max_flips)
        RETRIES = 100
        for _ in range(num_flips):
            target_idx = random.randint(0, flat_bits.numel() - 1)
            for _retry in range(RETRIES):
                if float_view[target_idx].item() != 0.0:
                    break
                target_idx = random.randint(0, flat_bits.numel() - 1)

            if random.random() > 0.5:
                bit_to_flip = random.randint(23, 30)
            else:
                bit_to_flip = random.randint(0, 22)

            flat_bits[target_idx] ^= (1 << bit_to_flip)

        self.strike_count += 1
        corrupted_tensor = flat_bits.view(target_tensor.shape).view(torch.float32)

        if is_list:
            corrupted_out[list_idx] = corrupted_tensor
            return type(module_out)(corrupted_out)
        return corrupted_tensor


class BoundsCheckShield:
    """Activation clamp — verbatim from sgc_evaluate_GOOD_2.0.py."""

    def __init__(self, clip_min=-20.0, clip_max=20.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, module, module_in, module_out):
        is_list = isinstance(module_out, (list, tuple))
        if is_list:
            safe_out = list(module_out)
            for i in range(len(safe_out)):
                tensor = torch.nan_to_num(safe_out[i], nan=0.0,
                                          posinf=self.clip_max, neginf=self.clip_min)
                safe_out[i] = torch.clamp(tensor, self.clip_min, self.clip_max)
            return type(module_out)(safe_out)
        tensor = torch.nan_to_num(module_out, nan=0.0,
                                  posinf=self.clip_max, neginf=self.clip_min)
        return torch.clamp(tensor, self.clip_min, self.clip_max)


def time_single_inference(forward_fn, images, eval_input_hw, target_hw):
    """Time one forward pass on CPU using perf_counter. Returns ms-per-image."""
    if eval_input_hw is not None:
        images = F.interpolate(images, size=(eval_input_hw, eval_input_hw),
                               mode="bilinear", align_corners=False)

    start = time.perf_counter()
    outputs = forward_fn(images)
    end = time.perf_counter()

    batch_ms = (end - start) * 1000.0
    per_image_ms = batch_ms / images.shape[0]
    return per_image_ms, outputs


def forward_baseline(model):
    def _f(images):
        return model(images)
    return _f


def forward_unshielded(model):
    # Identical to baseline forward at the call site — the chaos hook is
    # already registered on model.encoder.
    def _f(images):
        return model(images)
    return _f


def forward_tmr(model_a, model_b, model_c):
    def _f(images):
        out_a = model_a(images)
        out_b = model_b(images)
        out_c = model_c(images)
        pred_a = out_a.argmax(dim=1)
        pred_b = out_b.argmax(dim=1)
        pred_c = out_c.argmax(dim=1)
        stacked = torch.stack([pred_a, pred_b, pred_c], dim=0)
        voted, _ = torch.mode(stacked, dim=0)
        # Return shape (B, H, W) of voted preds; latency-only script never
        # touches the actual values, so returning preds is fine.
        return voted
    return _f


def run_latency_loop(forward_fn, loader, eval_input_hw, warmup_input_shape, desc):
    """Warm-up pass (timing discarded), then time every batch of `loader`."""
    print(f"\n[warm-up] forward on torch.zeros({tuple(warmup_input_shape)}) "
          "— timing discarded")
    warm = torch.zeros(*warmup_input_shape)
    with torch.no_grad():
        warm_start = time.perf_counter()
        _ = forward_fn(warm)
        warm_end = time.perf_counter()
    print(f"[warm-up] {(warm_end - warm_start) * 1000:.1f} ms (discarded)")

    latencies_ms = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            target_hw = labels.shape[-2:]
            per_image_ms, _ = time_single_inference(
                forward_fn, images, eval_input_hw, target_hw,
            )
            latencies_ms.append(per_image_ms)

    return latencies_ms


def main():
    ap = argparse.ArgumentParser(
        description="CPU-only single-thread FP32 latency measurement (thesis §4.3)"
    )
    ap.add_argument("--mode", required=True,
                    choices=["baseline", "unshielded_chaos", "shielded_chaos"])
    ap.add_argument("--weights", required=True, help="Path to .pth checkpoint.")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--variant", default=DEFAULT_VARIANT,
                    help="Gold-test variant subdirectory (default: min3-100agree).")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--input-hw", type=int, required=True,
                    help="Strategy-A square input resolution (e.g. 512 or 1024).")
    ap.add_argument("--max-flips", type=int, default=50,
                    help="Bit flips per forward pass for chaos modes "
                         "(matches sgc_evaluate_*.py calibration for MNv4-S).")
    ap.add_argument("--run-name", required=True,
                    help="Used to name the output JSON.")
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of frames timed (smoke test only).")
    args = ap.parse_args()

    # ---- HARD-FORCE CPU SINGLE-THREAD FP32 ----
    device = torch.device("cpu")
    torch.set_num_threads(1)
    # Also pin interop threads to 1 so OpenMP / MKL pools cannot multi-thread
    # behind our back. set_num_interop_threads must be called before any
    # parallel work starts, which is the case here.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialized — fine, just means a previous call set it.
        pass

    print("=" * 60)
    print("MEASURING ON CPU, single-threaded FP32")
    print("=" * 60)
    print(f"  device              : {device}")
    print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    print(f"  mode                : {args.mode}")
    print(f"  weights             : {args.weights}")
    print(f"  encoder             : {args.encoder}")
    print(f"  input_hw            : {args.input_hw}")
    print(f"  variant             : {args.variant}")
    print(f"  run_name            : {args.run_name}")
    if args.mode != "baseline":
        print(f"  max_flips           : {args.max_flips}")
    print("=" * 60)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"Weights file not found: {weights_path}")

    # ---- BUILD MODEL(S) AND REGISTER HOOKS ----
    if args.mode == "baseline":
        model = build_model(weights_path, device, encoder_name=args.encoder)
        n_params = count_params(model)
        forward_fn = forward_baseline(model)
    elif args.mode == "unshielded_chaos":
        model = build_model(weights_path, device, encoder_name=args.encoder)
        n_params = count_params(model)
        model.encoder.register_forward_hook(
            ChaoticSpaceRadiationInjector(probability=1.0, max_flips=args.max_flips)
        )
        forward_fn = forward_unshielded(model)
    elif args.mode == "shielded_chaos":
        model_a = build_model(weights_path, device, encoder_name=args.encoder)
        model_b = build_model(weights_path, device, encoder_name=args.encoder)
        model_c = build_model(weights_path, device, encoder_name=args.encoder)
        n_params = count_params(model_a)  # single-arch param count
        shield = BoundsCheckShield(clip_min=-20.0, clip_max=20.0)
        for m in (model_a, model_b, model_c):
            m.encoder.register_forward_hook(
                ChaoticSpaceRadiationInjector(probability=1.0, max_flips=args.max_flips)
            )
            m.encoder.register_forward_hook(shield)
        forward_fn = forward_tmr(model_a, model_b, model_c)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    print(f"  param count (single model): {n_params:,} ({n_params / 1e6:.2f}M)")

    # ---- BUILD DATASET ----
    data_root = Path(args.data_root)
    edr_dir = data_root / "images_op1" / "edr_op1"
    label_dir = data_root / "labels_op1" / "test_op1" / args.variant
    if not edr_dir.is_dir():
        raise SystemExit(f"EDR dir not found: {edr_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"Label dir not found: {label_dir}")

    ds = MSLGoldTestDataset(label_dir, edr_dir)
    print(f"  dataset: usable={len(ds)}  skipped_no_edr={ds.skipped_no_edr}  "
          f"skipped_all_ignore={ds.skipped_all_ignore}")
    if len(ds) == 0:
        raise SystemExit("Empty dataset — nothing to measure.")
    if args.limit is not None:
        ds.pairs = ds.pairs[:args.limit]
        print(f"  [LIMIT] truncated to first {len(ds)} frames (smoke test mode)")

    # num_workers=0 keeps the worker collation in this process — fairer for
    # latency since pin_memory + multiproc loaders artificially hide CPU cost.
    # pin_memory is meaningless on CPU.
    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ---- RUN ----
    warmup_shape = (1, 3, args.input_hw, args.input_hw)
    wall_start = time.perf_counter()
    latencies_ms = run_latency_loop(
        forward_fn, loader, eval_input_hw=args.input_hw,
        warmup_input_shape=warmup_shape, desc=args.run_name,
    )
    wall_end = time.perf_counter()
    wall_clock_s = wall_end - wall_start

    avg_lat = float(np.mean(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    latency_pass = max_lat <= LATENCY_CAP_MS

    # ---- REPORT ----
    print()
    print("=" * 60)
    print(f"  RESULTS — {args.run_name}")
    print("=" * 60)
    print(f"  frames timed        : {len(latencies_ms)}")
    print(f"  avg latency (ms)    : {avg_lat:.2f}")
    print(f"  p95 latency (ms)    : {p95_lat:.2f}")
    print(f"  p99 latency (ms)    : {p99_lat:.2f}")
    print(f"  max latency (ms)    : {max_lat:.2f}")
    print(f"  SGC #2 cap          : {LATENCY_CAP_MS} ms (max)")
    print(f"  SGC #2 verdict      : {'PASS' if latency_pass else 'FAIL'}")
    print(f"  wall-clock (s)      : {wall_clock_s:.1f}")
    print("=" * 60)

    # ---- WRITE JSON ----
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.run_name}_cpu_latency.json"

    out = {
        "run_name": args.run_name,
        "mode": args.mode,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "dtype": "fp32",
        "weights": str(weights_path),
        "encoder": args.encoder,
        "input_hw": args.input_hw,
        "variant": args.variant,
        "param_count": n_params,
        "max_flips": args.max_flips if args.mode != "baseline" else None,
        "frames_timed": len(latencies_ms),
        "warmup_discarded": True,
        "sgc2_latency_avg_ms": avg_lat,
        "sgc2_latency_p95_ms": p95_lat,
        "sgc2_latency_p99_ms": p99_lat,
        "sgc2_latency_max_ms": max_lat,
        "sgc2_cap_ms": LATENCY_CAP_MS,
        "sgc2_latency_pass": latency_pass,
        "wall_clock_seconds": wall_clock_s,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    main()
