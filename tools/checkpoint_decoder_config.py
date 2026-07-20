#!/usr/bin/env python3
"""Print decoder architecture values needed to load a SoundStream checkpoint."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    package = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(package, dict) or "config" not in package:
        raise ValueError(
            f"Checkpoint does not contain the model config required for a safe handoff: {checkpoint}"
        )

    config = pickle.loads(package["config"])
    mode = str(config.get("decoder_upsample_mode", "convtranspose"))
    kernel_min = int(config.get("decoder_linear_upsample_kernel_min", 0))
    if mode not in ("convtranspose", "linear"):
        raise ValueError(f"Unsupported decoder_upsample_mode={mode!r}")
    if kernel_min < 0:
        raise ValueError(f"Invalid decoder_linear_upsample_kernel_min={kernel_min}")

    # One raw value per line keeps Bash mapfile parsing unambiguous.
    print(mode)
    print(kernel_min)


if __name__ == "__main__":
    main()
