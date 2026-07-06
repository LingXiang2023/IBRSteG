<div align="center">

# IBRSteG

### Learning a Generalizable Steganography Framework for 3D Gaussian Splatting

<p>
  <a href="https://arxiv.org/abs/2606.30024"><img src="https://img.shields.io/badge/arXiv-2606.30024-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/IEEE-TMM%202026-00629B.svg" alt="IEEE TMM">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/3DGS-Steganography-8A2BE2.svg" alt="3DGS Steganography">
  <img src="https://img.shields.io/badge/License-TBD-lightgrey.svg" alt="License">
  <a href="https://github.com/LingXiang2023/IBRSteG"><img src="https://img.shields.io/github/stars/LingXiang2023/IBRSteG?style=social" alt="GitHub stars"></a>
</p>

**Fanye Kong**<sup>\*</sup> &nbsp;·&nbsp; **Hongyu Xia**<sup>\*</sup> &nbsp;·&nbsp; Yu Zheng &nbsp;·&nbsp; Boyang Gong &nbsp;·&nbsp; Jie Zhou &nbsp;·&nbsp; Jiwen Lu

*Department of Automation, Tsinghua University*

<sub><sup>\*</sup> Equal contribution &nbsp;•&nbsp; Corresponding author: Yu Zheng</sub>

<p>
  <a href="https://arxiv.org/abs/2606.30024"><b>📄 Paper</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/LingXiang2023/IBRSteG"><b>💻 Code</b></a> &nbsp;|&nbsp;
  <a href="#-citation"><b>🔖 Citation</b></a>
</p>

</div>

---

Official PyTorch implementation of **IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting**, accepted by **IEEE Transactions on Multimedia (TMM)**.

IBRSteG hides a secret 3D Gaussian scene inside a cover 3D Gaussian scene in a **feed-forward** way — no per-scene optimization. Unlike prior 3DGS steganography whose parameters are rigidly tied to a fixed scene pair, IBRSteG treats embedding as a *scene-agnostic* function that generalizes to unseen scenes. It reconstructs cover/secret scenes with a frozen **GPS-Gaussian+** backbone, converts them into structured **Gaussian Attribute Maps (GAM)** — depth, color, rotation, opacity — and trains a **Gaussian Attributes Steganographer (GAS)** to embed and extract secrets directly in this 2D attribute domain.

> [!NOTE]
> This directory is the **cleaned open-source release**. It is self-contained and independent from the experimental folders in the parent research workspace. `benchmark4` directories elsewhere are internal comparison baselines and are not used here.

## ✨ Highlights

- **📌 TMM 2026** — accepted by IEEE Transactions on Multimedia.
- **🌐 Generalizable 3DGS steganography** — hides a full secret 3D scene inside a cover scene *without* per-scene optimization or retraining.
- **⚡ Feed-forward & fast** — end-to-end embedding + extraction in **~2 seconds/scene**, vs. hours for scene-specific baselines.
- **🧩 GPS-Gaussian+ backbone** — the reconstruction backbone is frozen; only the lightweight GAS module is trained.
- **✅ Ready to run** — includes an inference checkpoint (`model_zoo/ibrsteg_test_weight.pth`), THU/THumanMV training & testing scripts, and reproducibility commands.

## 📑 Table of Contents

- [Method Overview](#-method-overview)
- [Results](#-results)
- [Repository Layout](#-repository-layout)
- [Installation](#-installation)
- [Data](#-data)
- [Checkpoints](#-checkpoints)
- [Testing on THU](#-testing-on-thu)
- [Training on THU](#-training-on-thu)
- [Notes for Open-Source Users](#-notes-for-open-source-users)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)

## 🧠 Method Overview

IBRSteG runs in three stages:

1. **Input & GAM Generation.** Cover and secret 3DGS scenes are reconstructed from dual-view images using a *frozen* GPS-Gaussian+ model, then mapped by `M(·)` into pixel-aligned **Gaussian Attribute Maps (GAM)** encoding `{depth, color, rotation, opacity}`. Each pixel deterministically corresponds to exactly one Gaussian, giving a clean 2D representation compatible with convolutional learning.
2. **Embedding (GAS Encoder).** A U-Net fuses the cover and secret GAMs, feeding four parallel heads (**Depth / RGB / Rotation / Opacity**). A residual connection adds the cover depth so the Depth head learns a *refinement* rather than predicting depth from scratch. The heads are combined into a **stego GAM**, which the inverse mapping `M⁻¹` lifts back into a stego 3DGS scene.
3. **Transmission & Extraction (GAS Decoder).** After transmission, the stego scene is re-mapped to a stego GAM; the decoder recovers the secret GAM, and `M⁻¹` reconstructs the secret 3DGS. The shared camera parameters `P` (<1 KB) act as a compact geometric key required for coherent recovery.

Training is fully end-to-end and supervised with a **3D rendering loss** (pixel + SSIM + optional Chamfer) plus a **2D GAM loss** over the four attributes.

## 📊 Results

Quantitative comparison across datasets — higher ↑ is better, lower ↓ is better. *Vanilla Upper Bound* = GPS-Gaussian+ reconstruction without any embedding (reference ceiling).

| Dataset | Method | \| Cover/Stego PSNR↑ | SSIM↑ | LPIPS↓ | \| Secret/Recovered PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|--:|--:|--:|--:|--:|--:|
| **THuman MV** | Vanilla Upper Bound | 33.05 | 0.961 | 0.156 | 33.05 | 0.961 | 0.156 |
| | Gaussian + StegaNeRF | 30.93 | 0.958 | 0.210 | 17.77 | 0.763 | 0.560 |
| | Gaussian + Weng et al. | 14.65 | 0.604 | 0.606 | 20.66 | 0.781 | 0.420 |
| | **IBRSteG (ours)** | **32.98** | **0.960** | **0.171** | **32.40** | **0.958** | **0.180** |
| **ENeRF** | Vanilla Upper Bound | 20.93 | 0.510 | 0.379 | 20.93 | 0.510 | 0.379 |
| | **IBRSteG (ours)** | **20.94** | **0.533** | **0.393** | **21.01** | **0.537** | **0.400** |
| **DyNeRF** | Vanilla Upper Bound | 28.16 | 0.909 | 0.186 | 28.16 | 0.909 | 0.186 |
| | **IBRSteG (ours)** | **23.73** | **0.871** | **0.314** | **24.29** | **0.880** | **0.309** |


## 🗂 Repository Layout

```text
IBRSteG_open_source/
├── train_thu.py                    # full GAS training on processed THU/THumanMV data
├── test_thu.py                     # random cover/secret evaluation on THU/THumanMV
├── ibrsteg/                        # public IBRSteG wrappers and losses
├── stegamodels/Conet.py            # GAS network implementation, legacy alias preserved
├── lib/, core/, gaussian_renderer/ # GPS-Gaussian+ backbone code
├── data_process/                   # GPS-Gaussian+ THU/custom-data preprocessing helpers
├── config/stage.yaml               # default training/testing config
├── model_zoo/
│   ├── gps_plus_final.pth          # frozen GPS-Gaussian+ backbone weight
│   └── ibrsteg_test_weight.pth     # inference-only IBRSteG/GAS test weight
├── scripts/
│   ├── train_thu.sh
│   └── test_thu.sh
└── tools/
    ├── build_rasterizer.py
    ├── export_inference_checkpoint.py
    └── run_with_local_rasterizer.py
```

## ⚙️ Installation

The recommended environment is the existing **`gps_plus`** Conda environment. The release code, local rasterizer build, THU training smoke test, random-test smoke test, `per_subject100`, and `per_scene100` checks were all verified with `gps_plus`.

If `gps_plus` already exists on your machine, use it directly:

```bash
cd IBRSteG_open_source
conda activate gps_plus
```

> [!TIP]
> The commands below use `conda run -n gps_plus ...` intentionally, so they don't depend on shell activation state.

<details>
<summary><b>Recreate the environment from scratch (fallback)</b></summary>

```bash
cd IBRSteG_open_source
conda env create -f environment.yml
conda activate gps_plus
```
</details>

<details>
<summary><b>Differentiable Gaussian rasterizer (self-contained local build)</b></summary>

You need the differentiable Gaussian rasterizer used by 3DGS/GPS-Gaussian+. If the active environment already has a compatible `diff_gaussian_rasterization`, the plain `python train_thu.py` / `python test_thu.py` commands work directly.

For a self-contained release build, a compatible rasterizer source tree is bundled under `third_party/`. Build it locally with:

```bash
conda run -n gps_plus python tools/build_rasterizer.py \
  --allow-cuda-mismatch \
  --cxx-standard c++17
```

Then launch training/testing through the local-rasterizer runner:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --help
```
</details>

**Optional dependencies**

- **Chamfer regularization** → install `pytorch3d`. If missing, `train_thu.py` warns and disables Chamfer automatically. Disable explicitly with `python train_thu.py --no-chamfer ...`.
- **LPIPS evaluation** → install `lpips`. If omitted, LPIPS is reported as `0.0`; PSNR and SSIM still run.

> [!WARNING]
> If `diff_gaussian_rasterization` raises an `undefined symbol` import error, the extension was compiled against a different PyTorch ABI. Rebuild it in the active PyTorch/CUDA environment, or use the local build command above and validate it with the smoke tests. On systems where PyTorch was built with a newer CUDA runtime than the installed `nvcc`, prefer the local build.

> [!NOTE]
> On the verified workstation, avoid prefixing commands with extra environment variables such as `CUDA_VISIBLE_DEVICES=0`; the explicit `conda run -n gps_plus ...` form was the most stable path.

## 📁 Data

This release expects the processed THU/THumanMV layout used by GPS-Gaussian+:

```text
data/thu_processed/
├── train/
│   ├── img/<sample_name>/{0,1,2,3,4,5}.jpg
│   ├── mask/<sample_name>/{0,1,...}.jpg or .png
│   └── parameter/<sample_name>/
│       ├── 0_1.json
│       ├── 2_intrinsic.npy
│       ├── 2_extrinsic.npy
│       └── ...
├── val/
│   ├── img/
│   ├── mask/
│   └── parameter/
└── test/
    ├── s1a4_process/
    │   ├── img/
    │   ├── mask/
    │   └── parameter/
    └── ...
```


<details>
<summary><b>Preprocessing raw THU/THumanMV captures</b></summary>

If you start from raw captures, use the preprocessing helpers:

```bash
cd data_process
python step_0rect.py -i s1a1 -t train
python step_1.py    -i s1a1 -t train
python step_0rect.py -i s3a5 -t val
python step_1.py    -i s3a5 -t val
python step_0rect.py -i s1a6 -t test
python step_1.py    -i s1a6 -t test
cd ..
```

> [!IMPORTANT]
> Before running these, edit `data_root` and `processed_data_root` inside the preprocessing files to match your raw and processed data locations.
</details>

## 💾 Checkpoints

| File | Purpose |
| --- | --- |
| `model_zoo/gps_plus_final.pth` | Frozen GPS-Gaussian+ backbone checkpoint |
| `model_zoo/ibrsteg_test_weight.pth` | Inference-only IBRSteG/GAS checkpoint for testing |

The provided test weight was exported from the development checkpoint `rebuttal_other/experiments/09260_0926/ckpt/09260_final.pth`. The original full checkpoint (~489 MB, with optimizer/scheduler state) is trimmed to an inference-only weight (~163 MB, GAS parameters + minimal metadata).

<details>
<summary><b>Regenerate the inference checkpoint</b></summary>

```bash
python tools/export_inference_checkpoint.py \
  --input /path/to/09260_final.pth \
  --output model_zoo/ibrsteg_test_weight.pth
```

Robustness experiments can use the fuller training-state checkpoint (`09260_final.pth` or `09260_latest.pth`). It is intentionally *not* required by the standard test command.
</details>

## 🧪 Testing on THU

**Quick smoke test:**

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --gas-checkpoint model_zoo/ibrsteg_test_weight.pth \
  --test-seqs s1a4 s1a5 s1a6 s2a4 s3a5 \
  --views 2 3 \
  --num-rounds 5 \
  --output-dir results/thu_smoke \
  --skip-lpips
```

**Full random-pair test:**

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --gas-checkpoint model_zoo/ibrsteg_test_weight.pth \
  --test-seqs s1a4 s1a5 s1a6 s2a4 s3a5 \
  --views 2 3 \
  --num-rounds 1000 \
  --output-dir results/thu_test \
  --save-images
```

<details>
<summary><b>Release protocols: <code>per_subject100</code> and <code>per_scene100</code></b></summary>

**Subject-level protocol** — discover every test scene matching `sXaY_process` with `Y >= 4`, group by subject `s`, take 100 cover rounds per subject, then average:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --gas-checkpoint model_zoo/ibrsteg_test_weight.pth \
  --protocol per_subject100 \
  --per-subject-rounds 100 \
  --min-action-id 4 \
  --views 2 3 \
  --output-dir results/thu_per_subject100 \
  --num-workers 0 \
  --skip-lpips
```

When `--test-seqs` is omitted in `per_subject100`, the script discovers scenes from `<data-root>/test`. In the original processed THU split, this selects `s1a4_process`, `s1a5_process`, `s1a6_process`, `s2a4_process`, `s3a5_process`, then evaluates `s1`, `s2`, `s3` for 300 total rounds.

**Scene-level variant** — each selected `sXaY` scene contributes exactly 100 rounds:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --gas-checkpoint model_zoo/ibrsteg_test_weight.pth \
  --protocol per_scene100 \
  --per-scene-rounds 100 \
  --min-action-id 4 \
  --views 2 3 \
  --output-dir results/thu_per_scene100 \
  --num-workers 0 \
  --skip-lpips
```

Or use the wrapper after activating `gps_plus`:

```bash
bash scripts/test_thu.sh \
  --data-root /path/to/thu_processed \
  --protocol per_subject100 \
  --per-subject-rounds 100 \
  --min-action-id 4 \
  --views 2 3 \
  --output-dir results/thu_per_subject100 \
  --num-workers 0 \
  --skip-lpips
```
</details>

**Outputs**

- `metrics.json` — full configuration, per-round rows, and summary means/stds
- `metrics.csv` — per-round metrics
- `scene_metrics.csv` — 100-round average for each selected `sXaY` scene
- `subject_metrics.csv` — average for each selected subject `s`
- Optional images under `images/{stego, cover_gt, recovered_secret, secret_gt}`

## 🏋️ Training on THU

Train GAS while keeping GPS-Gaussian+ frozen:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  train_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --output-dir experiments/ibrsteg_thu \
  --experiment-name ibrsteg_thu \
  --num-workers 8
```

<details>
<summary><b>Resume training / wrapper script</b></summary>

**Resume:**

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  train_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --resume experiments/ibrsteg_thu/ckpt/ibrsteg_thu_latest.pth \
  --output-dir experiments/ibrsteg_thu_resume
```

**Wrapper:**

```bash
bash scripts/train_thu.sh \
  --data-root /path/to/thu_processed \
  --experiment-name ibrsteg_thu
```
</details>

The training loop optimizes:

- cover/stego rendering fidelity
- secret/recovered rendering fidelity
- optional left/right Chamfer consistency
- 2D GAM attribute loss using the weights in `config/stage.yaml`

## 🙏 Acknowledgements

This project was conducted at **Tsinghua University**.

This codebase builds on **GPS-Gaussian+**. We sincerely thank **Boyao Zhou**, author of GPS-Gaussian / GPS-Gaussian+, and the GPS-Gaussian+ team for their excellent work and released codebase. We also thank **Tsinghua University** for the THU/THumanMV data and research environment.

Special thanks to the **Spark Program / 星火计划**. The author is a member of the Spark Program, and this project benefited from its support and research atmosphere.

## 🔖 Citation

If this project is useful for your work, please cite IBRSteG:

```bibtex
@article{kong2026ibrsteg,
  title   = {IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting},
  author  = {Kong, Fanye and Xia, Hongyu and Zheng, Yu and Gong, Boyang and Zhou, Jie and Lu, Jiwen},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026}
}

@misc{kong2026ibrsteg_arxiv,
  title         = {IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting},
  author        = {Kong, Fanye and Xia, Hongyu and Zheng, Yu and Gong, Boyang and Zhou, Jie and Lu, Jiwen},
  year          = {2026},
  eprint        = {2606.30024},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.30024}
}
```

---

<div align="center">
<sub>⭐ If you find IBRSteG helpful, consider giving the repo a star.</sub>
</div>
```

