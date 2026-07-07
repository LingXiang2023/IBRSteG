"""Gaussian Attributes Steganographer (GAS).

The training checkpoints produced during development used the module name
``Conet``.  The public API exposes the paper terminology while keeping the
state-dict layout unchanged for checkpoint compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from stegamodels.Conet import GaussianAttributesSteganographer


def load_gas_state_dict(
    gas: GaussianAttributesSteganographer,
    checkpoint: Mapping[str, object],
    strict: bool = True,
) -> None:
    """Load a GAS checkpoint from either the open-source or legacy key names."""

    if "gas" in checkpoint:
        state_dict = checkpoint["gas"]
    elif "conet" in checkpoint:
        state_dict = checkpoint["conet"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a state dict or a checkpoint containing `gas`/`conet`.")

    normalized = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value

    gas.load_state_dict(normalized, strict=strict)
