#!/usr/bin/env python3
"""Build the local diff-gaussian-rasterization extension for this environment."""

from __future__ import annotations

import argparse
import os
import re
import runpy
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RASTERIZER_SRC = ROOT / "third_party" / "diff-gaussian-rasterization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the bundled diff-gaussian-rasterization extension."
    )
    parser.add_argument(
        "--build-lib",
        default=str(ROOT / ".local_build" / "diff_gaussian_rasterization"),
        help="Directory that will receive the built Python package.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(ROOT / ".local_build" / "diff_gaussian_rasterization_src"),
        help="Scratch source directory used for compilation.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the extension into the active Python environment after building.",
    )
    parser.add_argument(
        "--allow-cuda-mismatch",
        action="store_true",
        help=(
            "Bypass PyTorch's CUDA version check. This is useful on machines where "
            "PyTorch was built with a newer CUDA runtime but only an older nvcc is "
            "available. Use only after validating a smoke test."
        ),
    )
    parser.add_argument(
        "--cxx-standard",
        default=None,
        help=(
            "Override the C++ standard passed to cxx/nvcc, for example c++17. "
            "PyTorch 2.12 defaults to c++20, which CUDA 11.8 nvcc cannot parse."
        ),
    )
    return parser.parse_args()


def prepare_source(work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(
        RASTERIZER_SRC,
        work_dir,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    return work_dir


def patch_setup_for_standard(source_dir: Path, cxx_standard: str) -> None:
    setup_path = source_dir / "setup.py"
    setup_text = setup_path.read_text()
    if "-std=c++" in setup_text:
        setup_path.write_text(re.sub(r"-std=c\+\+\d+", f"-std={cxx_standard}", setup_text))
        return

    needle = (
        'extra_compile_args={"nvcc": ["-I" + os.path.join('
        'os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}'
    )
    replacement = (
        'extra_compile_args={\n'
        f'                "cxx": ["-std={cxx_standard}"],\n'
        f'                "nvcc": ["-std={cxx_standard}", "-I" + os.path.join('
        'os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")],\n'
        "            }"
    )
    if needle not in setup_text:
        raise RuntimeError(f"Could not patch {setup_path}; setup.py format is unexpected.")
    setup_path.write_text(setup_text.replace(needle, replacement))


def main() -> None:
    args = parse_args()
    if not RASTERIZER_SRC.exists():
        raise FileNotFoundError(f"Rasterizer source not found: {RASTERIZER_SRC}")
    build_source = prepare_source(Path(args.work_dir).resolve())

    import torch
    import torch.utils.cpp_extension as cpp_extension

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}, torch CUDA: {torch.version.cuda}")
    print(f"CUDA_HOME: {cpp_extension.CUDA_HOME}")
    print(f"nvcc: {shutil.which('nvcc')}")

    if args.allow_cuda_mismatch:
        cpp_extension._check_cuda_version = lambda *unused_args, **unused_kwargs: None
        print("Bypassing PyTorch CUDA version check.")

    if args.cxx_standard:
        patch_setup_for_standard(build_source, args.cxx_standard)
        print(f"Using -std={args.cxx_standard} for cxx/nvcc.")

    build_lib = Path(args.build_lib).resolve()
    build_lib.parent.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(build_source)
        if args.install:
            sys.argv = ["setup.py", "install"]
        else:
            sys.argv = ["setup.py", "build_ext", "--build-lib", str(build_lib)]
        runpy.run_path("setup.py", run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    if not args.install:
        init_src = build_source / "diff_gaussian_rasterization" / "__init__.py"
        package_dir = build_lib / "diff_gaussian_rasterization"
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(init_src, package_dir / "__init__.py")
        print(f"Built local package at: {build_lib}")
        print("Use it with:")
        print(f"  export PYTHONPATH={build_lib}:$PYTHONPATH")


if __name__ == "__main__":
    main()
