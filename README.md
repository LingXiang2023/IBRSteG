# IBRSteG: 3D Gaussian Splatting Steganography

Official PyTorch implementation of **IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting**.

**Accepted by IEEE Transactions on Multimedia (TMM).**

**Paper:** [arXiv:2606.30024](https://arxiv.org/abs/2606.30024)  
**Institution:** Tsinghua University  
**Codebase:** built on GPS-Gaussian+ with an open-source cleanup for THU/THumanMV training and testing

IBRSteG hides a secret 3D Gaussian scene inside a cover 3D Gaussian scene in a feed-forward way. The code uses **GPS-Gaussian+** as the frozen generalizable 3DGS backbone and trains a **Gaussian Attributes Steganographer (GAS)** on **Gaussian Attribute Maps (GAM)** containing depth, color, rotation, and opacity.

## Highlights

- **TMM 2026:** accepted by IEEE Transactions on Multimedia.
- **Generalizable 3DGS steganography:** hides a secret 3D Gaussian scene inside a cover 3D Gaussian scene without per-scene optimization.
- **Tsinghua University research release:** includes THU/THumanMV training and testing scripts, processed-data protocol, and reproducibility commands.
- **GPS-Gaussian+ backbone:** uses a frozen GPS-Gaussian+ model and trains only the IBRSteG/GAS module.
- **Ready-to-test checkpoint:** includes `model_zoo/ibrsteg_test_weight.pth` for THU evaluation.

## News

- **2026-06-29:** Paper released on arXiv: [arXiv:2606.30024](https://arxiv.org/abs/2606.30024).
- **TMM:** The paper is accepted by **IEEE Transactions on Multimedia (TMM)**.
- This directory is the cleaned release version. It is independent from the experimental folders in the parent workspace.
- A test checkpoint is provided at `model_zoo/ibrsteg_test_weight.pth`.
- The full training-state checkpoint from the robustness experiments is not attached by default. See [Checkpoints](#checkpoints).

## Release Status

The open-source directory is ready for public release after choosing how to host the model weights:

- The source code, README, license, THU training/testing scripts, test checkpoint, and GPS-Gaussian+ backbone checkpoint are present.
- All smoke tests and THU reproduction checks in this release were verified in the `gps_plus` Conda environment.
- `benchmark4` and other comparison-baseline folders from the parent workspace are intentionally excluded.
- Generated local build artifacts such as `.local_build/`, `__pycache__/`, `data/`, `experiments/`, and `results/` are ignored and should not be uploaded.
- For GitHub, consider Git LFS or an external download link for `model_zoo/*.pth` files because the bundled weights are large.

## Method Naming

The paper and public code use the following names:

| Paper name | Code path | Notes |
| --- | --- | --- |
| IBRSteG | this repository | Generalizable 3DGS steganography framework |
| GAS, Gaussian Attributes Steganographer | `ibrsteg/gas.py`, `stegamodels/Conet.py` | The internal legacy alias `Conet` is kept only for checkpoint compatibility |
| GAM, Gaussian Attribute Map | tensors produced by frozen GPS-Gaussian+ | Contains depth, RGB/color, rotation, scale, opacity, and valid mask |
| cover scene | `cover_*` variables | Scene that should remain visually unchanged |
| secret scene | `secret_*` variables | Scene to hide and later recover |
| stego scene | `stego_*` variables | Cover-looking scene carrying the secret |
| recovered scene | `recovered_*` variables | Decoded secret scene |

## Repository Layout

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

## Installation

The recommended environment is the existing **`gps_plus`** Conda environment. The release code, local rasterizer build, THU training smoke test, random test smoke test, `per_subject100`, and `per_scene100` checks were all verified with `gps_plus`.

If `gps_plus` already exists on your machine, use it directly:

```bash
cd IBRSteG_open_source
conda activate gps_plus
```

If you need to recreate the environment, a Conda environment file is provided as a fallback:

```bash
cd IBRSteG_open_source
conda env create -f environment.yml
conda activate gps_plus
```

The commands below use `conda run -n gps_plus` intentionally, so they do not depend on shell activation state.

You also need the differentiable Gaussian rasterizer used by 3DGS/GPS-Gaussian+. If the active environment already has a compatible `diff_gaussian_rasterization`, the normal `python train_thu.py` and `python test_thu.py` commands work directly.

For a self-contained release build, this repository includes a compatible rasterizer source tree under `third_party/`. Build it locally with:

```bash
conda run -n gps_plus python tools/build_rasterizer.py \
  --allow-cuda-mismatch \
  --cxx-standard c++17
```

Then launch training or testing through the local-rasterizer runner:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  test_thu.py -- \
  --help
```

For training with Chamfer regularization, install `pytorch3d`. If it is not installed, `train_thu.py` will warn and disable Chamfer. You can also disable it explicitly:

```bash
python train_thu.py --no-chamfer ...
```

For LPIPS evaluation in `test_thu.py`, install `lpips`. If omitted, LPIPS is reported as `0.0`; PSNR and SSIM still run.

If `diff_gaussian_rasterization` raises an `undefined symbol` import error, rebuild the extension in the active PyTorch/CUDA environment. This usually means the extension was compiled against a different PyTorch ABI. On systems where PyTorch was built with a newer CUDA runtime than the installed `nvcc`, use the local build command above and validate it with the smoke tests below.

On the verified workstation, avoid prefixing the commands with extra environment variables such as `CUDA_VISIBLE_DEVICES=0`; using the explicit `conda run -n gps_plus ...` commands below was the most stable path.

## Data

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

The default source views are `0` and `1`; novel target views are `2` and `3` for evaluation and `2,3,4,5` for training supervision.

If you start from raw THU/THumanMV captures, use the preprocessing helpers:

```bash
cd data_process
python step_0rect.py -i s1a1 -t train
python step_1.py -i s1a1 -t train
python step_0rect.py -i s3a5 -t val
python step_1.py -i s3a5 -t val
python step_0rect.py -i s1a6 -t test
python step_1.py -i s1a6 -t test
cd ..
```

Before running them, edit `data_root` and `processed_data_root` in the preprocessing files to match your raw and processed data locations.

## Checkpoints

Bundled files:

| File | Purpose |
| --- | --- |
| `model_zoo/gps_plus_final.pth` | Frozen GPS-Gaussian+ backbone checkpoint |
| `model_zoo/ibrsteg_test_weight.pth` | Inference-only IBRSteG/GAS checkpoint for testing |

The provided test weight was exported from the development checkpoint `rebuttal_other/experiments/09260_0926/ckpt/09260_final.pth` in the original workspace.

The original full checkpoint is about 489 MB and includes optimizer/scheduler state. The exported test weight is about 163 MB and contains only GAS model parameters plus minimal metadata. To regenerate it:

```bash
python tools/export_inference_checkpoint.py \
  --input /path/to/09260_final.pth \
  --output model_zoo/ibrsteg_test_weight.pth
```

Robustness experiments can use the fuller training-state checkpoint from `rebuttal_other/experiments/09260_0926/ckpt/09260_final.pth` or `09260_latest.pth`. That checkpoint is intentionally not required by the standard test command.

## Test On THU

Run a short smoke test:

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

Run the full random-pair test:

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

For the THU subject-level protocol used in the release checks, discover every test scene matching `sXaY_process` with `Y >= 4`, group them by subject `s`, take 100 cover rounds per subject, and then average:

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

When `--test-seqs` is omitted in `per_subject100`, the script discovers scenes from `<data-root>/test`. In the original processed THU split used here, this selects `s1a4_process`, `s1a5_process`, `s1a6_process`, `s2a4_process`, and `s3a5_process`, then evaluates `s1`, `s2`, and `s3` for 300 total rounds.

If you want each selected `sXaY` scene to contribute exactly 100 rounds, use the scene-level variant:

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

You can also use the wrapper after activating `gps_plus`:

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

Outputs:

- `metrics.json`: full configuration, per-round rows, and summary means/stds
- `metrics.csv`: per-round metrics
- `scene_metrics.csv`: 100-round average for each selected `sXaY` scene
- `subject_metrics.csv`: average for each selected subject `s`
- optional images under `images/stego`, `images/cover_gt`, `images/recovered_secret`, and `images/secret_gt`

## Train On THU

Train GAS while freezing GPS-Gaussian+:

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

Resume training:

```bash
conda run -n gps_plus python tools/run_with_local_rasterizer.py \
  --rasterizer-build .local_build/diff_gaussian_rasterization \
  train_thu.py -- \
  --data-root /path/to/thu_processed \
  --gps-checkpoint model_zoo/gps_plus_final.pth \
  --resume experiments/ibrsteg_thu/ckpt/ibrsteg_thu_latest.pth \
  --output-dir experiments/ibrsteg_thu_resume
```

Or use the wrapper:

```bash
bash scripts/train_thu.sh \
  --data-root /path/to/thu_processed \
  --experiment-name ibrsteg_thu
```

The training loop optimizes:

- cover/stego rendering fidelity
- secret/recovered rendering fidelity
- optional left/right Chamfer consistency
- 2D GAM attribute loss using the weights in `config/stage.yaml`

## Notes For Open-Source Users

- `stegamodels/Conet.py` still contains `Conet = GaussianAttributesSteganographer` as a compatibility alias. New code should import `GaussianAttributesSteganographer` from `ibrsteg`.
- `config/stage.yaml` uses relative paths by default. Prefer command-line overrides for dataset and checkpoint paths.
- `data/`, `experiments/`, `results/`, and logs are ignored by `.gitignore`.
- `benchmark4` directories in the parent workspace are comparison baselines and are not used by this release.

## Acknowledgements

This project was conducted at **Tsinghua University**.

This codebase builds on **GPS-Gaussian+**. We sincerely thank **Boyao Zhou**, the author of GPS-Gaussian/GPS-Gaussian+, and the GPS-Gaussian+ team for their excellent work and released codebase.

We also thank **Tsinghua University** for the THU/THumanMV data and research environment that made this project possible.

Special thanks to the **Spark Program / 星火计划**. The author is a member of the Spark Program, and this project benefited from its support and research atmosphere.

## Citation

If this project is useful for your work, please cite IBRSteG:

```bibtex
@article{kong2026ibrsteg,
  title={IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting},
  author={Kong, Fanye and Xia, Hongyu and Zheng, Yu and Gong, Boyang and Zhou, Jie and Lu, Jiwen},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}

@misc{kong2026ibrsteg_arxiv,
  title={IBRSteG: Learning a Generalizable Steganography Framework for 3D Gaussian Splatting},
  author={Kong, Fanye and Xia, Hongyu and Zheng, Yu and Gong, Boyang and Zhou, Jie and Lu, Jiwen},
  year={2026},
  eprint={2606.30024},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.30024}
}

```

