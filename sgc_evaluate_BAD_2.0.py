"""SGC evaluation harness — baseline mode.

Iteration 4: The Control Group (Unshielded MBU Chaos)
  - SGC #1 (lightweight): param count vs the 3M cap
  - SGC #2 (latency): max inference time per frame vs 500ms cap
  - SGC #3 (memory): peak pure-ML tensor RAM vs 256MB cap
  - SGC #4 (Survivability): Chaotic Radiation ON, All Shields OFF

Default checkpoint: ML_Model_512.pth at project root.
Run from SPACE_PROJECT root:
    python sgc_evaluate.py
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random

from metrics import (
    NUM_CLASSES,
    IGNORE_INDEX,
    CLASS_NAMES,
    update_confusion_matrix,
    compute_metrics,
    format_report,
)


DEFAULT_WEIGHTS = "ML_Model_512.pth"
DEFAULT_DATA_ROOT = "MSL_NAVCAM_TEST_SET"
DEFAULT_VARIANTS = [
    "masked-gold-min1-100agree_op1",
    "masked-gold-min2-100agree_op1",
    "masked-gold-min3-100agree_op1",
]
DEFAULT_ENCODER = "tu-mobilenetv3_small_100"
EVAL_INPUT_HW = 512

# --- SPACE-GRADE CONSTRAINTS ---
LIGHTWEIGHT_PARAM_CAP = 3_000_000
LATENCY_CAP_MS = 500       
RAM_CAP_MB = 256           
# -------------------------------


class MSLGoldTestDataset(Dataset):
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


# --- PROBLEM CREATOR (CHAOTIC RADIATION: MULTIPLE BIT UPSETS) ---
class ChaoticSpaceRadiationInjector:
    def __init__(self, probability=1.0, max_flips=5):
        """
        probability: 1.0 means every frame gets hit.
        max_flips: Simulates a physical particle streak flipping a cluster of bits.
        """
        self.probability = probability
        self.max_flips = max_flips
        self.strike_count = 0

    def __call__(self, module, module_in, module_out):
        if random.random() > self.probability:
            return module_out

        is_list = isinstance(module_out, (list, tuple))
        if is_list:
            corrupted_out = list(module_out)
            # FIX: always target the DEEPEST encoder feature (the last element of
            # the list). For smp.DeepLabV3Plus, this is the ASPP input — guaranteed
            # to propagate through the decoder. Previously this used
            # random.randint(0, len-1) which often picked unused stage outputs
            # (encoders return ~6 feature maps but the DLV3+ decoder consumes
            # only 2), producing a false "PASS" verdict because the injected
            # faults never reached the prediction head.
            list_idx = len(corrupted_out) - 1
            target_tensor = corrupted_out[list_idx].clone().detach().contiguous()
        else:
            target_tensor = module_out.clone().detach().contiguous()
        
        raw_bits = target_tensor.view(torch.int32)
        flat_bits = raw_bits.view(-1)
        
        # Shotgun blast: flip 1 to max_flips bits simultaneously
        num_flips = random.randint(1, self.max_flips)
        # Cache the float view once — used for non-zero index selection below.
        # Modern compact CNNs (e.g., MNv4-Conv-Small) produce sparse deep
        # features after ReLU + BatchNorm. Naive random index selection often
        # lands on zero-valued positions where any bit flip just produces a
        # subnormal float (~1e-43) that is effectively still zero, yielding a
        # silent null injection. We resample up to RETRIES times to find a
        # non-zero target so the injected fault has measurable downstream
        # impact.
        float_view = target_tensor.view(-1)
        RETRIES = 100
        for _ in range(num_flips):
            target_idx = random.randint(0, flat_bits.numel() - 1)
            for _retry in range(RETRIES):
                if float_view[target_idx].item() != 0.0:
                    break
                target_idx = random.randint(0, flat_bits.numel() - 1)

            # 50/50 mix: Exponents (Math Explosions) OR Fractions (Silent Corruption)
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
        else:
            return corrupted_tensor


def evaluate_on_loader(model, loader, device, eval_input_hw=EVAL_INPUT_HW):
    """Strategy-A eval: resize input, predict, upsample, score. Includes latency tracking."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    latencies_ms = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            target_h, target_w = labels.shape[-2:]
            
            if eval_input_hw is not None:
                images = F.interpolate(
                    images, size=(eval_input_hw, eval_input_hw),
                    mode="bilinear", align_corners=False,
                )
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            outputs = model(images)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_latency_ms = (end_time - start_time) * 1000
            per_image_latency_ms = batch_latency_ms / images.shape[0]
            latencies_ms.append(per_image_latency_ms)

            if outputs.shape[-2] != target_h or outputs.shape[-1] != target_w:
                outputs = F.interpolate(
                    outputs, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.numpy()
            update_confusion_matrix(cm, preds, labels_np)
            
    avg_latency = float(np.mean(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    return cm, avg_latency, max_latency


def cm_to_jsonable(cm):
    m = compute_metrics(cm)
    return {
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "pixel_accuracy": m["pixel_acc"],
        "mean_iou": m["miou"],
        "per_class_iou": {n: float(m["iou"][i]) for i, n in enumerate(CLASS_NAMES)},
        "per_class_precision": {n: float(m["precision"][i]) for i, n in enumerate(CLASS_NAMES)},
        "per_class_recall": {n: float(m["recall"][i]) for i, n in enumerate(CLASS_NAMES)},
        "per_class_support": {n: int(m["support"][i]) for i, n in enumerate(CLASS_NAMES)},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Unshielded Chaotic Radiation Stress Test (Control Group)"
    )
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help=f"Path to .pth checkpoint. Default: {DEFAULT_WEIGHTS}")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help=f"Test-set root. Default: {DEFAULT_DATA_ROOT}")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                    help="Gold-test variant subdirectory names.")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help=f"smp encoder name. Default: {DEFAULT_ENCODER}")
    ap.add_argument("--input-hw", type=int, default=EVAL_INPUT_HW,
                    help=f"Strategy-A square input resolution. Default: {EVAL_INPUT_HW}.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Batch size. MUST be 1 for accurate latency simulation.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-id", default="unshielded_chaos_fp32",
                    help="Subfolder under --results-root.")
    ap.add_argument("--max-flips", type=int, default=5,
                    help="Max bit flips per forward pass (1..max_flips uniform). "
                         "Default 5 matches the thesis Section 3.6 calibration for "
                         "MobileNetV3-Small (96-channel deep feature). For wider "
                         "encoders, scale by channel-width ratio.")
    ap.add_argument("--results-root", default="sgc_results",
                    help="Parent folder for SGC harness outputs.")
    args = ap.parse_args()

    out_dir = Path(args.results_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    # CPU single-thread FP32 — matches thesis §4.3 latency spec.
    # Previously this was GPU-when-available, which contradicted the spec and
    # produced optimistic per-frame timings; SGC #2 must be measured on the
    # same device class as the deployment target (rad-hard space CPU).
    device = torch.device("cpu")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialised by an earlier call — fine.
        pass
    print("=" * 60)
    print("MEASURING ON CPU, single-threaded FP32")
    print(f"  device                         : {device}")
    print(f"  torch.get_num_threads()        : {torch.get_num_threads()}")
    print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    print("=" * 60)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"Weights file not found: {weights_path}")
    print(f"Loading {weights_path}  (arch=smp.DeepLabV3Plus, encoder={args.encoder})")
    model = build_model(weights_path, device, encoder_name=args.encoder)

    # --- ARM THE CHAOTIC RADIATION GUN (NO SHIELDS) ---
    radiation_gun = ChaoticSpaceRadiationInjector(probability=1.0, max_flips=args.max_flips)
    print(f"  Radiation gun armed: max_flips={args.max_flips} (per forward pass)")
    model.encoder.register_forward_hook(radiation_gun)

    # Warm-up pass — timing discarded. Required by the thesis §4.3 latency
    # convention so that the first-frame cold start (lazy MKLDNN pool init,
    # first-cache-miss page faults, etc.) does not pollute the max-latency
    # statistic that SGC #2 evaluates.
    warm = torch.zeros(1, 3, args.input_hw, args.input_hw)
    with torch.no_grad():
        _ = model(warm)
    print("  Warm-up forward complete (timing discarded)")

    print("\n=======================================================")
    print("[!] DANGER: Chaotic Multiple Bit Upsets (MBUs) ACTIVE.")
    print("[!] WARNING: All Software Mitigations (TMR, Clamping) are OFF.")
    print("=======================================================\n")

    n_params = count_params(model)
    lightweight_pass = n_params <= LIGHTWEIGHT_PARAM_CAP
    print(f"Param count: {n_params:,}  ({n_params / 1e6:.2f}M)  "
          f"-> SGC #1 (lightweight, <={LIGHTWEIGHT_PARAM_CAP / 1e6:.0f}M): "
          f"{'PASS' if lightweight_pass else 'FAIL'}")

    data_root = Path(args.data_root)
    edr_dir = data_root / "images_op1" / "edr_op1"
    test_root = data_root / "labels_op1" / "test_op1"
    if not edr_dir.is_dir():
        raise SystemExit(f"EDR dir not found: {edr_dir}")
    if not test_root.is_dir():
        raise SystemExit(f"Test labels root not found: {test_root}")

    results = {
        "run_id": args.run_id,
        "mode": "unshielded_chaos_fp32",
        "weights": str(weights_path),
        "arch": "smp.DeepLabV3Plus",
        "encoder": args.encoder,
        "eval_input_hw": args.input_hw,
        "param_count": n_params,
        "sgc_lightweight_pass": lightweight_pass,
        "variants": {},
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for variant in args.variants:
        label_dir = test_root / variant
        if not label_dir.is_dir():
            print(f"  [skip] {variant}: directory not found at {label_dir}")
            continue
        ds = MSLGoldTestDataset(label_dir, edr_dir)
        print(f"\n--- {variant}: usable={len(ds)}  skipped_no_edr={ds.skipped_no_edr}  "
              f"skipped_all_ignore={ds.skipped_all_ignore}")
        if len(ds) == 0:
            continue
        
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        
        try:
            # Evaluate metrics and latency
            cm, avg_lat, max_lat = evaluate_on_loader(model, loader, device, eval_input_hw=args.input_hw)
            
            # --- MEMORY CHECKS ---
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            total_python_ram_mb = mem_info.rss / (1024 * 1024) 
            
            if device.type == "cuda":
                peak_ram_bytes = torch.cuda.max_memory_allocated(device)
                true_ml_ram_mb = peak_ram_bytes / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                true_ml_ram_mb = total_python_ram_mb

            # --- CONSTRAINT LOGIC ---
            latency_pass = max_lat <= LATENCY_CAP_MS
            ram_pass = true_ml_ram_mb <= RAM_CAP_MB

            print(format_report(cm, title=variant))
            print(f"--- Space-Grade Constraints (SGC) for {variant} ---")
            print(f"  Latency (Max):    {max_lat:.2f} ms | Pass: {latency_pass}")
            print(f"  True ML RAM:      {true_ml_ram_mb:.2f} MB | Pass: {ram_pass}\n")

            results["variants"][variant] = cm_to_jsonable(cm)
            results["variants"][variant].update({
                "num_images": len(ds),
                "sgc_latency_avg_ms": avg_lat,
                "sgc_latency_max_ms": max_lat,
                "sgc_latency_pass": latency_pass,
                "sgc_true_ml_ram_mb": true_ml_ram_mb,
                "sgc_ram_pass": ram_pass
            })
            
        except RuntimeError as e:
            # Catch the inevitable NaN/Infinity explosions
            print(f"\n[CRITICAL FAILURE] The unshielded model crashed mathematically on variant {variant}!")
            print(f"Error caught: {e}\n")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()