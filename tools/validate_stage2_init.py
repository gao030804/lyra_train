#!/usr/bin/env python3
"""Reject a Stage-2 handoff whose fixed-validation codec state is unhealthy."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["metric", "value"]:
            raise ValueError(f"Unexpected validation report header in {path}: {reader.fieldnames}")
        for row in reader:
            metrics[row["metric"]] = float(row["value"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--min-q00-active-ratio", type=float, default=0.70)
    parser.add_argument("--min-q00-perplexity", type=float, default=50.0)
    parser.add_argument("--max-recon-clip-fraction", type=float, default=1e-3)
    args = parser.parse_args()

    report = args.report.expanduser().resolve()
    metrics = read_metrics(report)

    def finite(key: str) -> float:
        if key not in metrics:
            raise KeyError(f"Metric {key!r} is missing from {report}")
        value = metrics[key]
        if not math.isfinite(value):
            raise ValueError(f"Metric {key!r} is not finite in {report}: {value}")
        return value

    checks = {
        "aligned_si_sdr_finite": math.isfinite(finite("aligned_si_sdr")),
        "q00_active_ratio": finite("codebook_q00_active_ratio") >= args.min_q00_active_ratio,
        "q00_perplexity": finite("codebook_q00_perplexity") >= args.min_q00_perplexity,
        "q00_validation_eligible": finite("q00_validation_eligible") >= 0.5,
        "q01_validation_eligible": finite("q01_validation_eligible") >= 0.5,
        "rvq_validation_eligible": finite("rvq_validation_eligible") >= 0.5,
        "collapsed_quantizers": finite("codebook_collapsed_quantizers") < 0.5,
        "recon_clip_fraction": finite("recon_clip_fraction") <= args.max_recon_clip_fraction,
    }

    print("Stage-2 initialization preflight")
    print(f"report={report}")
    print(
        "q00_active_ratio="
        f"{metrics['codebook_q00_active_ratio']:.4f} "
        f"(required>={args.min_q00_active_ratio:.4f})"
    )
    print(
        "q00_perplexity="
        f"{metrics['codebook_q00_perplexity']:.2f} "
        f"(required>={args.min_q00_perplexity:.2f})"
    )
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(
            "Stage-2 initialization rejected: unhealthy fixed-validation checkpoint; "
            f"failed={','.join(failed)}"
        )

    print("Stage-2 initialization accepted.")


if __name__ == "__main__":
    main()
