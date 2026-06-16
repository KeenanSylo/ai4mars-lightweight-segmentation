# Big Rock Segmentation on AI4Mars under Space-Grade Constraints

Companion code for the bachelor's thesis *Terrain Detection for Autonomous Mars Rovers - Optimizing Big Rock Segmentation via Machine Learning under Space-grade Constraints* (Linnaeus University, course 2DV50E, VT 2026).

The thesis investigates how architectural and training-related design choices affect rare-class terrain segmentation on the AI4Mars dataset under constrained deployment conditions. A lightweight DeepLabv3+ with a MobileNetV4-Conv-Small encoder at 512×512 input, trained with focal-Tversky loss and basic augmentation, reaches a Big Rock IoU of 0.446 on the gold expert test set at the `min3-100agree` variant, on 3.00 M parameters. A simulated radiation fault injector and a shielding stack (activation clamp + three-copy majority vote) recover the rare-class accuracy to within 0.0003 of the baseline under simulated bit-flip faults.

## Repository contents

```
DLV3+_MNv4_small_trainer.py        # main trainer (DeepLabv3+ + MobileNetV4-Conv-Small)
DLV3+_MNV3_small_trainer.py        # sister trainer for the MobileNetV3-Small encoder
MSL_train_preprocessor.py          # PyTorch Dataset for AI4Mars Curiosity NAVCAM
metrics.py                         # per-class IoU / precision / recall / F1 from confusion matrix
sgc_evaluate_GOOD_2.0.py           # constraint evaluation + fault injection + shielded TMR
sgc_evaluate_BAD_2.0.py            # unshielded comparator
training/                          # losses, augmentations, class-weights helpers
cpu_latency_eval/                  # single-thread CPU latency measurement (FP32, warm-up discarded)
architecture_sweep/                # stage 1: cross-architecture sweep (15 cells, weights + configs)
training_configuration_sweep/      # stage 2: training-configuration sweep (16 runs, weights + configs)
final_model/                       # final chosen models + per-constraint scorecards
docs/                              # evaluation protocol notes
thesis_scope.md                    # scope map of which experiments are in scope for the thesis
requirements.txt
```

## Dataset

This repository **does not** redistribute the AI4Mars dataset. Download it directly from NASA's Jet Propulsion Laboratory:

> Swan, R. M., Atha, D., Leopold, H. A., Gildner, M., Oij, S., Chiu, C., & Ono, M. (2021).
> *AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars.*
> CVPR Workshops 2021. <https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars>

Place the Curiosity NAVCAM subset such that the dataset paths used by `MSL_train_preprocessor.py` resolve correctly. The trainer scripts accept dataset roots through their command-line flags.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12 and PyTorch with CUDA support on the training side, CPU only on the latency-measurement side.

## Reproducing the headline result (Big Rock IoU 0.446)

The chosen final model is **Run 11** with DeepLabv3+ on MobileNetV4-Conv-Small at 512×512 input, trained for 25 epochs with focal-Tversky (`alpha=0.3`, `beta=0.7`) and basic augmentation. The corresponding checkpoint is `final_model/v2_R11_MNv4-S_512.pth`.

To re-evaluate it on the gold expert test set:

```bash
python sgc_evaluate_GOOD_2.0.py \
    --weights final_model/v2_R11_MNv4-S_512.pth \
    --encoder tu-mobilenetv4_conv_small \
    --input-hw 512
```

To re-train from scratch on a single GPU, see the `config.json` inside the matching run folder under `training_configuration_sweep/experiments/training_run_11_focal_+_tversky_512p_aug_B/` for the exact CLI flags.

For the CPU-only latency measurement reported in Table 5.6 of the thesis:

```bash
python cpu_latency_eval/latency_cpu.py --weights final_model/v2_R11_MNv4-S_512.pth
```

## Citation

If you use this code or any of the released model checkpoints, please cite the thesis:

```bibtex
@thesis{arsianto-devanaboina-2026,
  title  = {Terrain Detection for Autonomous Mars Rovers --
            Optimizing Big Rock Segmentation via Machine Learning under
            Space-grade Constraints},
  author = {Arsianto, Keenan Syahlevi and Devanaboina, Yashwanth Krishna},
  school = {Linnaeus University},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```

A `CITATION.cff` is also included for tools that read it automatically.

## Acknowledgments

- The **AI4Mars** dataset is the work of Swan et al. at NASA's Jet Propulsion Laboratory.
- The **`segmentation_models_pytorch`** library by Pavel Iakubovskii provides the DeepLabv3+ and FPN decoder implementations.
- The **`timm`** library by Ross Wightman provides the MobileNetV3-Small and MobileNetV4-Conv-Small encoders.
- The **Albumentations** library by Buslaev et al. provides the augmentation pipelines.
- The fault-injection methodology follows the bit-flip propagation analysis of Li et al. (2017), the resilience framework of Reagen et al. (2018), and the robust-ML survey of Shafique et al. (2020).

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

Note that the AI4Mars dataset itself is distributed by NASA JPL under its own terms.
