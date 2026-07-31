#!/usr/bin/env python3
"""Run one-at-a-time hyperparameter sensitivity checks for the paper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/defensive-boosting-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/defensive-boosting-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_configuration(
    out_dir: Path,
    *,
    seeds: list[int],
    algorithms: list[str],
    option: str | None = None,
    value: float | None = None,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "experiments.run",
        "--suite",
        "synthetic",
        "--T",
        "3000",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--stream-filter",
        "group_subset_heterogeneous",
        "--algorithm-filter",
        *algorithms,
        "--no-brier-aggregator",
        "--no-bbm-vote-diagnostic",
        "--no-plots",
        "--no-progress",
        "--out",
        str(out_dir),
    ]
    if option is not None and value is not None:
        command.extend([option, str(value)])
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    with (out_dir / "aggregate.json").open() as handle:
        return json.load(handle)["group_subset_heterogeneous_m=10_k=6_delta=0.6"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "out" / "paper" / "sensitivity")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    defensive = run_configuration(
        args.out / "defensive",
        seeds=args.seeds,
        algorithms=["defensive"],
    )["defensive"]

    multipliers = [0.5, 1.0, 1.5]
    results: dict[str, dict] = {"defensive": defensive}
    for multiplier in multipliers:
        key = f"{multiplier:g}"
        results[f"ogb_step_{key}"] = run_configuration(
            args.out / f"ogb_step_{key}",
            seeds=args.seeds,
            algorithms=["ogb_N=100"],
            option="--ogb-step-multiplier",
            value=multiplier,
        )["ogb_N=100"]
        eta_result = run_configuration(
            args.out / f"classifier_eta_{key}",
            seeds=args.seeds,
            algorithms=["bbm_N=100", "osboost_N=100"],
            option="--classifier-eta-multiplier",
            value=multiplier,
        )
        results[f"bbm_eta_{key}"] = eta_result["bbm_N=100"]
        results[f"osboost_eta_{key}"] = eta_result["osboost_N=100"]
        edge_result = run_configuration(
            args.out / f"edge_{key}",
            seeds=args.seeds,
            algorithms=["bbm_N=100", "osboost_N=100"],
            option="--edge-multiplier",
            value=multiplier,
        )
        results[f"bbm_edge_{key}"] = edge_result["bbm_N=100"]
        results[f"osboost_edge_{key}"] = edge_result["osboost_N=100"]

    with (args.out / "aggregate.json").open("w") as handle:
        json.dump({"group_subset_sensitivity": results}, handle, indent=2, sort_keys=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45))
    x = np.asarray(multipliers)
    panels = [
        ("step", ["ogb"], "Brier loss", "brier_loss"),
        (
            "eta",
            ["bbm", "osboost"],
            "randomized error",
            "randomized_error",
        ),
        (
            "edge",
            ["bbm", "osboost"],
            "randomized error",
            "randomized_error",
        ),
    ]
    labels = {"ogb": "OGB", "bbm": "Online BBM", "osboost": "OSBoost"}
    colors = {"ogb": "#d62728", "bbm": "#9467bd", "osboost": "#8c564b"}
    for ax, (section, algorithms, ylabel, metric) in zip(axes, panels):
        for algorithm in algorithms:
            means = []
            ses = []
            for multiplier in multipliers:
                values = results[f"{algorithm}_{section}_{multiplier:g}"]
                means.append(values[f"{metric}_mean"])
                ses.append(values[f"{metric}_stderr"])
            ax.errorbar(
                x,
                means,
                yerr=ses,
                marker="o",
                capsize=2.5,
                color=colors[algorithm],
                label=labels[algorithm],
            )
        defensive_metric = defensive[f"{metric}_mean"]
        ax.axhline(defensive_metric, color="#1f77b4", linestyle="--", linewidth=1.5, label="Defensive")
        ax.set_xticks(x)
        ax.set_xlabel("multiplier of reported setting")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title("OGB stage step")
    axes[1].set_title("Classifier learning rate")
    axes[2].set_title("Target edge")
    fig.tight_layout()
    fig.savefig(args.out / "sensitivity.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote sensitivity results to {args.out}")


if __name__ == "__main__":
    main()
