#!/usr/bin/env python3
"""Run an IBRSteG entry point with a locally built rasterizer package."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_LIB = ROOT / ".local_build" / "diff_gaussian_rasterization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize PyTorch/CUDA first, add a locally built "
            "diff_gaussian_rasterization package, then run a script."
        )
    )
    parser.add_argument("script", help="Script to run, for example train_thu.py or test_thu.py.")
    parser.add_argument(
        "--rasterizer-build",
        default=str(DEFAULT_BUILD_LIB),
        help="Directory created by tools/build_rasterizer.py.",
    )
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dir = Path(args.rasterizer_build).resolve()
    package_init = build_dir / "diff_gaussian_rasterization" / "__init__.py"
    if not package_init.exists():
        raise FileNotFoundError(
            f"Local rasterizer package not found at {build_dir}. "
            "Run tools/build_rasterizer.py first."
        )

    import torch

    if torch.cuda.is_available():
        torch.cuda.init()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(build_dir))
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = ROOT / script_path

    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    sys.argv = [str(script_path)] + script_args
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
