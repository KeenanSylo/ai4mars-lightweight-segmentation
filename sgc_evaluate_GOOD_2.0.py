"""SGC evaluation harness — The Ultimate Gauntlet

Iteration 5: The A/B Verification (Shielded Chaos)
  - SGC #1 (lightweight): param count vs the 3M cap
  - SGC #2 (latency): max inference time per frame vs 500ms cap
  - SGC #3 (memory): peak pure-ML tensor RAM vs 256MB cap
  - SGC #4 (Survivability): Chaotic MBU Radiation ON + All Shields ON

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


# --- PROBLEM CREATOR (CHAOTIC RADIATION: MBUs) ---
class ChaoticSpaceRadiationInjector:
    def __init__(self, probability=1.0, max_flips=5):
        self.probability = probability
        self.max_flips = max_flips

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
        
        num_flips = random.randint(1, self.max_flips)
        # Cache the float view once — used for non-zero index selection below.
        # Modern compact CNNs (e.g., MNv4-Conv-Small) produce sparse deep
        # features after ReLU + BatchNorm. Naive random index selection often
        # lands on zero-valued positions where any bit flip just produces a
        # subnormal float (~1e-43) that is effectively still zero, yielding a
        # silent null injection. We resample up to RETRIES times to find a
        # non-zero target so the injected fault has measurable downstream
        # impact (and the TMR shield has a real signal to recover from).
        float_view = target_tensor.view(-1)
        RETRIES = 100
        for _ in range(num_flips):
            target_idx = random.randint(0, flat_bits.numel() - 1)
            for _retry in range(RETRIES):
                if float_view[target_idx].item() != 0.0:
                    break
                target_idx = random.randint(0, flat_bits.numel() - 1)

            # Mix Exponents (Explosions) and Fractions (Silent Corruption)
            if random.random() > 0.5:
                bit_to_flip = random.randint(23, 30)
            else:
                bit_to_flip = random.randint(0, 22)

            flat_bits[target_idx] ^= (1 << bit_to_flip)
            
        corrupted_tensor = flat_bits.view(target_tensor.shape).view(torch.float32)
        
        if is_list:
            corrupted_out[list_idx] = corrupted_tensor
            return type(module_out)(corrupted_out)
        else:
            return corrupted_tensor


# --- PROBLEM SOLVER 1 (ALWAYS-ON CLAMP) ---
class BoundsCheckShield:
    def __init__(self, clip_min=-20.0, clip_max=20.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, module, module_in, module_out):
        is_list = isinstance(module_out, (list, tuple))
        if is_list:
            safe_out = list(module_out)
            for i in range(len(safe_out)):
                tensor = torch.nan_to_num(safe_out[i], nan=0.0, posinf=self.clip_max, neginf=self.clip_min)
                safe_out[i] = torch.clamp(tensor, self.clip_min, self.clip_max)
            return type(module_out)(safe_out)
        else:
            tensor = torch.nan_to_num(module_out, nan=0.0, posinf=self.clip_max, neginf=self.clip_min)
            return torch.clamp(tensor, self.clip_min, self.clip_max)


# --- PROBLEM SOLVER 2 (ALWAYS-ON TMR VOTER) ---
def evaluate_on_loader_tmr(model_a, model_b, model_c, loader, device, eval_input_hw=EVAL_INPUT_HW):
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
            
            out_a = model_a(images)
            out_b = model_b(images)
            out_c = model_c(images)
            
            if out_a.shape[-2] != target_h or out_a.shape[-1] != target_w:
                out_a = F.interpolate(out_a, size=(target_h, target_w), mode="bilinear", align_corners=False)
                out_b = F.interpolate(out_b, size=(target_h, target_w), mode="bilinear", align_corners=False)
                out_c = F.interpolate(out_c, size=(target_h, target_w), mode="bilinear", align_corners=False)
            
            pred_a = out_a.argmax(dim=1)
            pred_b = out_b.argmax(dim=1)
            pred_c = out_c.argmax(dim=1)
            
            stacked_preds = torch.stack([pred_a, pred_b, pred_c], dim=0)
            voted_preds, _ = torch.mode(stacked_preds, dim=0)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_latency_ms = (end_time - start_time) * 1000
            per_image_latency_ms = batch_latency_ms / images.shape[0]
            latencies_ms.append(per_image_latency_ms)

            preds = voted_preds.cpu().numpy()
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
    }


def main():
    ap = argparse.ArgumentParser(description="Space-Grade Chaotic TMR + Shield Evaluation")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--input-hw", type=int, default=EVAL_INPUT_HW)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-id", default="shielded_chaos_fp32")
    ap.add_argument("--max-flips", type=int, default=5,
                    help="Max bit flips per forward pass (1..max_flips uniform). "
                         "Default 5 matches the thesis Section 3.6 calibration for "
                         "MobileNetV3-Small (96-channel deep feature). For wider "
                         "encoders, scale by channel-width ratio.")
    ap.add_argument("--results-root", default="sgc_results")
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

    print("\n--- INITIALIZING HARDENED ROVER ARCHITECTURE ---")
    model_a = build_model(weights_path, device, encoder_name=args.encoder)
    model_b = build_model(weights_path, device, encoder_name=args.encoder)
    model_c = build_model(weights_path, device, encoder_name=args.encoder)
    print("[+] Models A, B, and C loaded into RAM.")

    # 1. APPLY CHAOTIC RADIATION TO ALL MODELS
    print(f"  Radiation gun armed on all 3 TMR models: max_flips={args.max_flips} (per forward pass)")
    model_a.encoder.register_forward_hook(ChaoticSpaceRadiationInjector(max_flips=args.max_flips))
    model_b.encoder.register_forward_hook(ChaoticSpaceRadiationInjector(max_flips=args.max_flips))
    model_c.encoder.register_forward_hook(ChaoticSpaceRadiationInjector(max_flips=args.max_flips))
    print("[!] DANGER: Chaotic Multiple Bit Upsets (MBUs) active on ALL models.")

    # 2. APPLY THE CLAMP SHIELD TO ALL MODELS (Catch Explosions)
    shield = BoundsCheckShield(clip_min=-20.0, clip_max=20.0)
    model_a.encoder.register_forward_hook(shield)
    model_b.encoder.register_forward_hook(shield)
    model_c.encoder.register_forward_hook(shield)
    print("[+] LAYER 1 DEFENSE: Bounds Checking deployed to catch Exponent explosions.")
    print("[+] LAYER 2 DEFENSE: TMR Voter ready to erase Fractional corruptions.\n")

    # Warm-up pass on all three models — timing discarded. Required by the
    # thesis §4.3 latency convention so that the first-frame cold start
    # (lazy MKLDNN pool init, page faults) does not pollute SGC #2 max-latency.
    warm = torch.zeros(1, 3, args.input_hw, args.input_hw)
    with torch.no_grad():
        _ = model_a(warm)
        _ = model_b(warm)
        _ = model_c(warm)
    print("[+] Warm-up forward on all 3 TMR copies complete (timing discarded)\n")

    n_params = count_params(model_a)
    lightweight_pass = n_params <= LIGHTWEIGHT_PARAM_CAP
    print(f"Param count (Base Arch): {n_params / 1e6:.2f}M  -> SGC #1 PASS: {lightweight_pass}")

    data_root = Path(args.data_root)
    edr_dir = data_root / "images_op1" / "edr_op1"
    test_root = data_root / "labels_op1" / "test_op1"

    results = {"run_id": args.run_id, "mode": "chaos_gauntlet", "variants": {}}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for variant in args.variants:
        label_dir = test_root / variant
        if not label_dir.is_dir(): continue
            
        ds = MSLGoldTestDataset(label_dir, edr_dir)
        if len(ds) == 0: continue
        
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        
        try:
            # This executes the models and runs the TMR Voter
            cm, avg_lat, max_lat = evaluate_on_loader_tmr(model_a, model_b, model_c, loader, device, eval_input_hw=args.input_hw)
            
            if device.type == "cuda":
                peak_ram_bytes = torch.cuda.max_memory_allocated(device)
                true_ml_ram_mb = peak_ram_bytes / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                true_ml_ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 

            latency_pass = max_lat <= LATENCY_CAP_MS
            ram_pass = true_ml_ram_mb <= RAM_CAP_MB

            print(format_report(cm, title=variant))
            print(f"--- Space-Grade Constraints (SGC) for {variant} ---")
            print(f"  Latency (3 Models + Voter): {max_lat:.2f} ms | Pass: {latency_pass}")
            print(f"  True ML RAM (3 Models):     {true_ml_ram_mb:.2f} MB | Pass: {ram_pass}\n")
            
            # --- FIXED: PACKAGING RESULTS FOR JSON ---
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
            print(f"\n[CRITICAL FAILURE] The shielded model crashed on variant {variant}!")
            print(f"Error caught: {e}\n")

    # --- FIXED: WRITING THE FINAL JSON FILE ---
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()