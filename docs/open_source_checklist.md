# Open Source Checklist

- [x] Use an independent release directory: `IBRSteG_open_source/`.
- [x] Exclude benchmark/comparison-baseline code from `rebuttal_other/benchmark4`.
- [x] Keep GPS-Gaussian+ backbone code needed for THU training/testing.
- [x] Rename public model API from legacy `Conet` to `GaussianAttributesSteganographer`.
- [x] Keep `Conet` alias for old checkpoint compatibility.
- [x] Remove machine-specific defaults from public training/testing commands.
- [x] Recommend and verify the existing `gps_plus` Conda environment.
- [x] Provide `train_thu.py` for complete THU training.
- [x] Provide `test_thu.py` for complete THU testing.
- [x] Provide `model_zoo/ibrsteg_test_weight.pth`.
- [x] Document the fuller robustness/training-state checkpoint without requiring it for normal tests.
- [x] Acknowledge Boyao Zhou, GPS-Gaussian+, Tsinghua University, and the Spark Program / 星火计划.
- [x] Add paper link: https://arxiv.org/abs/2606.30024.
- [x] Highlight IEEE Transactions on Multimedia (TMM) acceptance.
- [x] Run a training smoke test on `/opt/Data_zy/gswater/data_processed`.
- [x] Run THU testing smoke and `per_subject100`/`per_scene100` release checks.
- [x] Remove generated local build artifacts such as `.local_build/`.
- [ ] Before publishing, confirm whether large `.pth` files should be tracked directly or hosted externally.
