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
    parser.add_argument("--min-aligned-si-sdr", type=float, default=0.0)
    parser.add_argument("--min-aligned-correlation", type=float, default=0.65)
    parser.add_argument("--max-click-excess", type=float, default=0.5)
    parser.add_argument("--min-voiced-hf-ratio-db", type=float, default=-1.5)
    parser.add_argument("--max-voiced-hf-ratio-db", type=float, default=1.0)
    parser.add_argument("--max-quiet-hf-excess-db", type=float, default=1.0)
    parser.add_argument("--max-ac320-isolated", type=float, default=0.10)
    parser.add_argument("--max-comb-median-excess-db", type=float, default=8.0)
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
        "aligned_si_sdr": finite("aligned_si_sdr") >= args.min_aligned_si_sdr,
        "aligned_correlation": finite("aligned_correlation") >= args.min_aligned_correlation,
        "click_excess": finite("click_excess") <= args.max_click_excess,
        "voiced_hf_lower": finite("voiced_hf_energy_ratio_db") >= args.min_voiced_hf_ratio_db,
        "voiced_hf_upper": finite("voiced_hf_energy_ratio_db") <= args.max_voiced_hf_ratio_db,
        "quiet_hf_excess": finite("quiet_hf_excess_db") <= args.max_quiet_hf_excess_db,
        "ac320_isolated": finite("ac_320_isolated") <= args.max_ac320_isolated,
        "comb_median_excess": finite("comb_median_excess_db") <= args.max_comb_median_excess_db,
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
    print(
        "handoff_quality="
        f"aligned_si_sdr={metrics['aligned_si_sdr']:.3f}, "
        f"aligned_corr={metrics['aligned_correlation']:.3f}, "
        f"click_excess={metrics['click_excess']:+.4f}, "
        f"voiced_hf_ratio_db={metrics['voiced_hf_energy_ratio_db']:+.3f}, "
        f"quiet_hf_excess_db={metrics['quiet_hf_excess_db']:+.3f}, "
        f"ac320={metrics['ac_320_isolated']:+.4f}, "
        f"comb_median_db={metrics['comb_median_excess_db']:+.3f}"
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
