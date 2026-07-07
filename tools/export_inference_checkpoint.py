from __future__ import annotations

import argparse
from pathlib import Path

import torch


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an inference-only IBRSteG checkpoint.")
    parser.add_argument("--input", required=True, help="Training checkpoint containing `conet` or `gas`.")
    parser.add_argument("--output", required=True, help="Output inference checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint = safe_torch_load(input_path)
    if "gas" in checkpoint:
        gas_state = checkpoint["gas"]
    elif "conet" in checkpoint:
        gas_state = checkpoint["conet"]
    else:
        gas_state = checkpoint

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gas": gas_state,
            "conet": gas_state,
            "source_checkpoint": str(input_path),
            "total_steps": checkpoint.get("total_steps") if isinstance(checkpoint, dict) else None,
        },
        output_path,
    )
    print(f"saved inference checkpoint to {output_path}")


if __name__ == "__main__":
    main()
