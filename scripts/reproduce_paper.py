#!/usr/bin/env python3
"""Run the experiment configurations reported in the paper."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from check_results import compare_results


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "experiments" / "out" / "paper"
REFERENCE_ROOT = ROOT / "experiments" / "reference"
SEEDS = [str(seed) for seed in range(20)]

TARGETS = {
    "synthetic": [
        "--suite", "synthetic",
        "--T", "3000",
        "--seeds", *SEEDS,
        "--gain-oracle", "second_order",
        "--out", str(OUT_ROOT / "synthetic"),
        "--plots",
    ],
    "sweep": [
        "--suite", "synthetic",
        "--T", "3000",
        "--seeds", *SEEDS,
        "--stream-filter", "group_subset_heterogeneous",
        "--learner-sweep", "1", "5", "20", "100",
        "--gain-oracle", "second_order",
        "--out", str(OUT_ROOT / "sweep"),
        "--plots",
    ],
    "real": [
        "--suite", "real",
        "--real-dim", "128",
        "--seeds", "0",
        "--gain-oracle", "second_order",
        "--out", str(OUT_ROOT / "real"),
        "--plots",
    ],
    "adaptive-real": [
        "--suite", "real",
        "--real-dim", "128",
        "--seeds", "0",
        "--adaptive-defensive",
        "--algorithm-filter", "defensive", "adaptive_defensive",
        "--gain-oracle", "second_order",
        "--out", str(OUT_ROOT / "adaptive-real"),
        "--plots",
    ],
    "drift": [
        "--suite", "drift",
        "--seeds", "0",
        "--adaptive-defensive",
        "--algorithm-filter", "defensive", "adaptive_defensive",
        "--gain-oracle", "second_order",
        "--out", str(OUT_ROOT / "drift"),
        "--plots",
    ],
    "regression": [],
    "sensitivity": [],
}

REFERENCE_FILES = {
    target: REFERENCE_ROOT / f"{target}.json" for target in TARGETS
}

FIGURE_MAP = {
    "synthetic": {
        "planted_decoy_margin_d=200_gamma=0.12__randomized_error.png": "planted_decoy_randomized_error.png",
        "group_subset_heterogeneous_m=10_k=6_delta=0.6__classification_error.png": "group_subset_classification_error.png",
        "group_subset_heterogeneous_m=10_k=6_delta=0.6__randomized_error.png": "group_subset_randomized_error.png",
        "linear_span_fallback_d=40__brier_loss.png": "linear_span_brier_loss.png",
        "mixed_linear_random_label_mixture_p=0.35__brier_loss.png": "random_label_mixture_brier_loss.png",
        "mixed_linear_random_label_mixture_p=0.35__certificate.png": "random_label_mixture_certificate.png",
        "random_labels__brier_loss.png": "random_labels_brier_loss.png",
        "bbm_vote_diagnostic.png": "bbm_vote_diagnostic.png",
    },
    "sweep": {
        "group_subset_heterogeneous_m=10_k=6_delta=0.6__compute_sweep.png": "group_subset_compute_sweep.png",
    },
    "real": {
        "real_bank_marketing__brier_loss.png": "real_bank_marketing_brier_loss.png",
        "real_moa_electricity__brier_loss.png": "real_moa_electricity_brier_loss.png",
        "real_moa_airlines__brier_loss.png": "real_moa_airlines_brier_loss.png",
        "real_uci_occupancy__brier_loss.png": "real_uci_occupancy_brier_loss.png",
        "real_summary_brier.png": "real_summary_brier.png",
    },
    "adaptive-real": {
        "adaptive_real_brier.png": "adaptive_real_brier.png",
    },
    "drift": {
        "adaptive_insects_brier.png": "adaptive_insects_brier.png",
        "adaptive_insects_abrupt_hard_core_evolution.png": "adaptive_insects_abrupt_hard_core_evolution.png",
    },
    "regression": {
        "regression_mse.png": "regression_mse.png",
    },
    "sensitivity": {
        "sensitivity.png": "hyperparameter_sensitivity.png",
    },
}


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_target(target: str, *, dry_run: bool) -> None:
    if target == "sensitivity":
        command = [
            sys.executable,
            "scripts/run_sensitivity.py",
            "--out",
            str(OUT_ROOT / "sensitivity"),
        ]
    elif target == "regression":
        command = [
            sys.executable,
            "-m",
            "experiments.run_regression",
            "--ogb-learners",
            "100",
            "--out",
            str(OUT_ROOT / "regression"),
        ]
    else:
        command = [sys.executable, "-m", "experiments.run", *TARGETS[target]]
    run_command(command, dry_run=dry_run)


def verify_target(target: str) -> None:
    actual = OUT_ROOT / target / "aggregate.json"
    failures = compare_results(actual, REFERENCE_FILES[target])
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures[:20])
        raise RuntimeError(f"{target} differs from the reference results:\n{details}")
    print(f"Verified {target} against {REFERENCE_FILES[target].relative_to(ROOT)}")


def sync_figures(target: str) -> None:
    plot_dir = (
        OUT_ROOT / target
        if target == "sensitivity"
        else OUT_ROOT / target / "plots"
    )
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for generated_name, paper_name in FIGURE_MAP[target].items():
        source = plot_dir / generated_name
        destination = figure_dir / paper_name
        if not source.exists():
            raise FileNotFoundError(f"missing generated figure: {source}")
        shutil.copy2(source, destination)
        print(f"Updated {destination.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["all", *TARGETS],
        help="experiment family to reproduce",
    )
    parser.add_argument("--skip-tests", action="store_true", help="skip unit tests")
    parser.add_argument("--no-verify", action="store_true", help="skip reference-result checks")
    parser.add_argument("--no-sync", action="store_true", help="do not update the paper figures")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(TARGETS) if args.target == "all" else [args.target]
    if not args.skip_tests:
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "experiments.test_core",
                "experiments.test_adaptive",
                "experiments.test_regression",
            ],
            dry_run=args.dry_run,
        )
    for target in selected:
        run_target(target, dry_run=args.dry_run)
        if args.dry_run:
            continue
        if not args.no_verify:
            verify_target(target)
        if not args.no_sync:
            sync_figures(target)


if __name__ == "__main__":
    main()
