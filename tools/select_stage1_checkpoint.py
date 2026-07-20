#!/usr/bin/env python3
"""Choose a new Stage-1 checkpoint or an established fallback from matched validation."""

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


def require(metrics: dict[str, float], key: str, report: Path) -> float:
    if key not in metrics:
        raise KeyError(f"Metric {key!r} is missing from {report}")
    value = metrics[key]
    if not math.isfinite(value):
        raise ValueError(f"Metric {key!r} is not finite in {report}: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-checkpoint", type=Path, required=True)
    parser.add_argument("--fallback-checkpoint", type=Path, required=True)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--fallback-report", type=Path, required=True)
    parser.add_argument("--decision-report", type=Path, required=True)
    parser.add_argument("--max-si-sdr-drop", type=float, default=0.15)
    parser.add_argument("--max-correlation-drop", type=float, default=0.01)
    parser.add_argument("--max-voiced-hf-drop-db", type=float, default=0.10)
    parser.add_argument("--max-voiced-hf-rise-db", type=float, default=0.50)
    parser.add_argument("--max-quiet-hf-rise-db", type=float, default=0.50)
    parser.add_argument("--max-ac320-rise", type=float, default=0.01)
    parser.add_argument("--max-click-rise", type=float, default=1.0)
    parser.add_argument("--max-recon-clip-fraction", type=float, default=1e-3)
    parser.add_argument("--min-q00-active-ratio", type=float, default=0.70)
    parser.add_argument("--min-q00-perplexity", type=float, default=50.0)
    args = parser.parse_args()

    new_checkpoint = args.new_checkpoint.expanduser().resolve()
    fallback_checkpoint = args.fallback_checkpoint.expanduser().resolve()
    new_report = args.new_report.expanduser().resolve()
    fallback_report = args.fallback_report.expanduser().resolve()
    decision_report = args.decision_report.expanduser().resolve()

    for checkpoint in (new_checkpoint, fallback_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    new = read_metrics(new_report)
    old = read_metrics(fallback_report)
    checks: list[tuple[str, float, float, str, bool]] = []

    def at_least(name: str, new_key: str, old_key: str, tolerance: float) -> None:
        new_value = require(new, new_key, new_report)
        old_value = require(old, old_key, fallback_report)
        limit = old_value - tolerance
        checks.append((name, new_value, old_value, f">= {limit:.6f}", new_value >= limit))

    def at_most(name: str, new_key: str, old_key: str, tolerance: float) -> None:
        new_value = require(new, new_key, new_report)
        old_value = require(old, old_key, fallback_report)
        limit = old_value + tolerance
        checks.append((name, new_value, old_value, f"<= {limit:.6f}", new_value <= limit))

    at_least("aligned_si_sdr", "aligned_si_sdr", "aligned_si_sdr", args.max_si_sdr_drop)
    at_least(
        "aligned_correlation",
        "aligned_correlation",
        "aligned_correlation",
        args.max_correlation_drop,
    )
    at_least(
        "voiced_hf_ratio_lower_bound",
        "voiced_hf_energy_ratio_db",
        "voiced_hf_energy_ratio_db",
        args.max_voiced_hf_drop_db,
    )
    at_most(
        "voiced_hf_ratio_upper_bound",
        "voiced_hf_energy_ratio_db",
        "voiced_hf_energy_ratio_db",
        args.max_voiced_hf_rise_db,
    )
    at_most(
        "quiet_hf_excess_db",
        "quiet_hf_excess_db",
        "quiet_hf_excess_db",
        args.max_quiet_hf_rise_db,
    )
    at_most("ac_320_isolated", "ac_320_isolated", "ac_320_isolated", args.max_ac320_rise)
    # Click is an advisory comparison, not a single-metric veto. The raw score
    # depends strongly on the target utterance's transients; Stage 1 now gates
    # matched click excess against the target instead.
    click_new = require(new, "click_score", new_report)
    click_old = require(old, "click_score", fallback_report)
    click_limit = click_old + args.max_click_rise
    click_advisory_pass = click_new <= click_limit

    eligibility_keys = (
        "q00_validation_eligible",
        "q01_validation_eligible",
        "rvq_validation_eligible",
    )
    new_eligibility = {
        key: require(new, key, new_report) >= 0.5 for key in eligibility_keys
    }
    fallback_eligibility = {
        key: require(old, key, fallback_report) >= 0.5 for key in eligibility_keys
    }
    def hard_eligible(metrics: dict[str, float], report: Path) -> tuple[bool, dict[str, bool]]:
        hard = {
            "q00_active": require(metrics, "codebook_q00_active_ratio", report) >= args.min_q00_active_ratio,
            "q00_perplexity": require(metrics, "codebook_q00_perplexity", report) >= args.min_q00_perplexity,
            "q00_stage_flag": require(metrics, "q00_validation_eligible", report) >= 0.5,
            "q01_stage_flag": require(metrics, "q01_validation_eligible", report) >= 0.5,
            "rvq_stage_flag": require(metrics, "rvq_validation_eligible", report) >= 0.5,
            "no_collapsed_quantizers": require(metrics, "codebook_collapsed_quantizers", report) < 0.5,
            "clip": require(metrics, "recon_clip_fraction", report) <= args.max_recon_clip_fraction,
        }
        return all(hard.values()), hard

    new_hard_ok, new_hard = hard_eligible(new, new_report)
    fallback_hard_ok, fallback_hard = hard_eligible(old, fallback_report)
    new_ok = all(check[-1] for check in checks) and new_hard_ok
    fallback_ok = fallback_hard_ok

    if new_ok:
        selected = new_checkpoint
        decision = "new checkpoint accepted"
    elif fallback_ok:
        selected = fallback_checkpoint
        decision = "new checkpoint rejected; compatible fallback selected"
    else:
        selected = None
        decision = "both new and fallback checkpoints failed Stage-2 initialization eligibility"

    decision_report.parent.mkdir(parents=True, exist_ok=True)
    with decision_report.open("w", encoding="utf-8") as handle:
        handle.write("Stage-1 automatic fallback decision\n")
        handle.write(f"new_checkpoint\t{new_checkpoint}\n")
        handle.write(f"fallback_checkpoint\t{fallback_checkpoint}\n")
        handle.write(f"decision\t{decision}\n")
        handle.write(f"selected_checkpoint\t{selected or 'none'}\n\n")
        handle.write("check\tnew\tfallback\trequirement\tpass\n")
        for name, new_value, old_value, requirement, passed in checks:
            handle.write(
                f"{name}\t{new_value:.9g}\t{old_value:.9g}\t{requirement}\t{int(passed)}\n"
            )
        handle.write(
            f"click_score_advisory\t{click_new:.9g}\t{click_old:.9g}\t"
            f"<= {click_limit:.6f}\t{int(click_advisory_pass)} (non-veto)\n"
        )
        for key in eligibility_keys:
            handle.write(
                f"{key}\t{require(new, key, new_report):.9g}\t"
                f"{require(old, key, fallback_report):.9g}\tnew and fallback >= 0.5\t"
                f"{int(new_eligibility[key])}/{int(fallback_eligibility[key])}\n"
            )
        for key in new_hard:
            handle.write(
                f"hard_{key}\t{int(new_hard[key])}\t{int(fallback_hard[key])}\t"
                "required for Stage 2\t"
                f"{int(new_hard[key])}/{int(fallback_hard[key])}\n"
            )

    if selected is None:
        raise SystemExit(
            "Automatic fallback refused to start Stage 2 because neither checkpoint passes the "
            "absolute q00/RVQ/clip preflight. "
            f"See {decision_report}"
        )

    # The launcher captures stdout; keep it to exactly one machine-readable path.
    print(selected)


if __name__ == "__main__":
    main()
