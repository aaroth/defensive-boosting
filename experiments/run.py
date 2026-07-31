"""Run the online boosting comparison suite.

Example:
    python3 -m experiments.run --quick
    python3 -m experiments.run --T 4000 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Dict, List

import numpy as np

from .algorithms import (
    DefensiveBooster,
    OnlineBBM,
    OnlineGradientBoosting,
    OnlineSmoothBoost,
    OnlineSquaredLossRegressor,
    OnlineUnboostedClassifier,
    StronglyAdaptiveDefensiveBooster,
)
from . import __version__
from .metrics import RunTrace, run_online_algorithm, summarize_trace
from .streams import StreamData, default_streams
from .weak_learners import (
    DyadicFiniteSecondOrderLinearOracle,
    DyadicLinearSecondOrderOracle,
    FiniteHedgeClassifier,
    FiniteHedgeLinearOracle,
    FiniteSecondOrderLinearOracle,
    LinearBallOracle,
    LinearPerceptronClassifier,
    LinearSecondOrderOracle,
)


def finite_hedge_eta(num_experts: int, horizon: int) -> float:
    """Single global finite-class Hedge rule used by all algorithms.

    The rule is the standard horizon-aware exponential-weights scale for
    losses in [0,1], capped only to avoid overly aggressive updates on very
    short smoke tests.
    """

    n = max(int(num_experts), 2)
    T = max(int(horizon), 1)
    return min(0.5, math.sqrt(8.0 * math.log(n) / T))


def build_algorithms(
    stream: StreamData,
    *,
    ogb_learners: int,
    bbm_learners: int,
    osb_learners: int,
    osb_gamma: float,
    bbm_gamma: float,
    gain_oracle: str,
    include_adaptive: bool = False,
    learner_sweep: List[int] | None = None,
) -> List:
    if include_adaptive and gain_oracle != "second_order":
        raise ValueError("the strongly adaptive variant requires the second-order oracle")
    if stream.weak_type == "finite":
        dim = stream.dim
        linear_eta = finite_hedge_eta(2 * dim, stream.T)
        classifier_eta = finite_hedge_eta(dim, stream.T)

        if gain_oracle == "second_order":
            def gain_factory() -> FiniteSecondOrderLinearOracle:
                return FiniteSecondOrderLinearOracle(base_dim=dim)
        elif gain_oracle == "first_order":
            def gain_factory() -> FiniteHedgeLinearOracle:
                return FiniteHedgeLinearOracle(base_dim=dim, eta=linear_eta)
        else:
            raise ValueError(f"unknown gain oracle: {gain_oracle}")

        def classifier_factory() -> FiniteHedgeClassifier:
            return FiniteHedgeClassifier(base_dim=dim, eta=classifier_eta)

    elif stream.weak_type == "linear":
        dim = stream.dim

        if gain_oracle == "second_order":
            def gain_factory() -> LinearSecondOrderOracle:
                return LinearSecondOrderOracle(dim=dim, radius=1.0)
        elif gain_oracle == "first_order":
            def gain_factory() -> LinearBallOracle:
                return LinearBallOracle(dim=dim, radius=1.0, eta=1.0)
        else:
            raise ValueError(f"unknown gain oracle: {gain_oracle}")

        def classifier_factory() -> LinearPerceptronClassifier:
            return LinearPerceptronClassifier(dim=dim, radius=1.0, eta=1.0)

    else:
        raise ValueError(f"unknown weak_type: {stream.weak_type}")

    # StreamData.gamma_hint is a correlation edge.  Chen--Lin--Lu denote half
    # that correlation by gamma, while Online BBM uses the correlation itself.
    osb_gamma_value = 0.5 * stream.gamma_hint if stream.gamma_hint is not None else osb_gamma
    bbm_gamma_value = stream.gamma_hint if stream.gamma_hint is not None else bbm_gamma
    algos = [
        DefensiveBooster(gain_factory, name="defensive"),
        OnlineSquaredLossRegressor(gain_factory, name="unboosted"),
        OnlineUnboostedClassifier(classifier_factory, name="unboosted_cls"),
    ]
    if include_adaptive:
        if stream.weak_type == "finite":
            adaptive_weak = DyadicFiniteSecondOrderLinearOracle(
                base_dim=stream.dim,
                horizon=stream.T,
            )
        else:
            adaptive_weak = DyadicLinearSecondOrderOracle(
                dim=stream.dim,
                horizon=stream.T,
                radius=1.0,
            )
        algos.insert(
            1,
            StronglyAdaptiveDefensiveBooster(
                weak=adaptive_weak,
                horizon=stream.T,
                name="adaptive_defensive",
            ),
        )
    if learner_sweep:
        ogb_counts = sorted(set(int(n) for n in learner_sweep))
        bbm_counts = ogb_counts
        osb_counts = ogb_counts
    else:
        ogb_counts = [ogb_learners]
        bbm_counts = [bbm_learners]
        osb_counts = [osb_learners]
    for n in ogb_counts:
        algos.append(OnlineGradientBoosting(gain_factory, n_learners=n, name=f"ogb_N={n}"))
    for n in bbm_counts:
        algos.append(
            OnlineBBM(
                classifier_factory,
                n_learners=n,
                gamma=bbm_gamma_value,
                name=f"bbm_N={n}",
            )
        )
    for n in osb_counts:
        algos.append(
            OnlineSmoothBoost(
                classifier_factory,
                n_learners=n,
                gamma=osb_gamma_value,
                combiner="ocp",
                name=f"osboost_N={n}",
            )
        )
    return algos


def run_suite(args: argparse.Namespace) -> tuple[List[Dict], List[RunTrace]]:
    streams = []
    if args.suite in {"synthetic", "all"}:
        for seed in args.seeds:
            streams.extend((seed, stream) for stream in default_streams(T=args.T, seed=seed))
    if args.suite in {"real", "all"}:
        from .real_streams import real_streams

        if args.progress:
            print(
                f"Loading real datasets (up to {args.real_max_rows} rows each, dimension {args.real_dim})",
                flush=True,
            )

        real = real_streams(max_rows=args.real_max_rows, dim=args.real_dim, progress=args.progress)
        streams.extend((seed, stream) for seed in args.seeds for stream in real)
    if args.suite in {"drift", "all"}:
        from .real_streams import insects_drift_streams

        if args.progress:
            limit = "all" if args.drift_max_rows is None else str(args.drift_max_rows)
            print(f"Loading INSECTS drift datasets (up to {limit} rows each)", flush=True)
        drift = insects_drift_streams(max_rows=args.drift_max_rows, progress=args.progress)
        streams.extend((seed, stream) for seed in args.seeds for stream in drift)
    if args.stream_filter:
        filters = tuple(args.stream_filter)
        streams = [(seed, stream) for seed, stream in streams if any(f in stream.name for f in filters)]

    sweep_size = len(set(args.learner_sweep)) if args.learner_sweep else 1
    per_stream_runs = 3 + int(args.adaptive_defensive) + 3 * sweep_size
    if args.algorithm_filter:
        per_stream_runs = len(set(args.algorithm_filter))
    total_runs = len(streams) * per_stream_runs
    progress_width = len(str(max(total_runs, 1)))
    suite_start = time.perf_counter()
    if args.progress:
        print(f"Starting {total_runs} runs across {len(streams)} dataset/seed pairs", flush=True)

    summaries: List[Dict] = []
    traces: List[RunTrace] = []
    completed = 0
    for seed, stream in streams:
        algorithms = build_algorithms(
            stream,
            ogb_learners=args.ogb_learners,
            bbm_learners=args.bbm_learners,
            osb_learners=args.osb_learners,
            osb_gamma=args.osb_gamma,
            bbm_gamma=args.bbm_gamma,
            gain_oracle=args.gain_oracle,
            include_adaptive=args.adaptive_defensive,
            learner_sweep=args.learner_sweep,
        )
        if args.algorithm_filter:
            requested = set(args.algorithm_filter)
            algorithms = [algorithm for algorithm in algorithms if algorithm.name in requested]
            missing = requested - {algorithm.name for algorithm in algorithms}
            if missing:
                raise ValueError(f"unknown or unavailable algorithms: {sorted(missing)}")
        for algo in algorithms:
            start = time.perf_counter()
            trace = run_online_algorithm(algo, stream, seed=seed)
            elapsed = time.perf_counter() - start
            traces.append(trace)
            summary = summarize_trace(trace)
            summary["gain_oracle"] = args.gain_oracle
            summary["elapsed_seconds"] = float(elapsed)
            summary["microseconds_per_round"] = float(1_000_000.0 * elapsed / stream.T)
            summaries.append(summary)
            completed += 1
            if args.progress:
                suite_elapsed = time.perf_counter() - suite_start
                print(
                    f"[{completed:{progress_width}d}/{total_runs}] "
                    f"{stream.name} seed={seed} {algo.name}: {elapsed:.2f}s "
                    f"(suite {suite_elapsed / 60.0:.1f}m)",
                    flush=True,
                )
    return summaries, traces


def write_outputs(args: argparse.Namespace, summaries: List[Dict], traces: List[RunTrace]) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    (out_dir / ".mplconfig").mkdir(exist_ok=True)
    (out_dir / ".cache").mkdir(exist_ok=True)

    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    with (out_dir / "environment.json").open("w") as f:
        json.dump(environment_metadata(), f, indent=2, sort_keys=True)

    summary_csv = out_dir / "summary.csv"
    fields = list(summaries[0].keys()) if summaries else []
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    aggregate = aggregate_summaries(summaries)
    with (out_dir / "aggregate.json").open("w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    npz_payload = {}
    for idx, trace in enumerate(traces):
        prefix = f"{idx:03d}__{trace.stream.name}__{trace.algorithm}__seed={trace.seed}"
        npz_payload[prefix + "__scores"] = trace.scores
        npz_payload[prefix + "__mistakes"] = trace.mistakes
        npz_payload[prefix + "__brier_losses"] = trace.brier_losses
        npz_payload[prefix + "__randomized_errors"] = trace.randomized_errors
        npz_payload[prefix + "__weak_residual_corrs"] = trace.weak_residual_corrs
        npz_payload[prefix + "__self_residual_corrs"] = trace.self_residual_corrs
        npz_payload[prefix + "__hard_core_edges"] = trace.hard_core_edges
        npz_payload[prefix + "__hard_core_densities"] = trace.hard_core_densities
    np.savez_compressed(out_dir / "traces.npz", **npz_payload)

    if args.plots:
        from .plots import (
            plot_adaptive_drift_comparison,
            plot_adaptive_real_comparison,
            plot_local_hard_core_evolution,
            plot_real_summary,
            plot_traces,
        )

        plot_paths = plot_traces(traces, out_dir / "plots")
        real_summary = plot_real_summary(aggregate, out_dir / "plots")
        if real_summary is not None:
            plot_paths.append(real_summary)
        adaptive_real = plot_adaptive_real_comparison(traces, out_dir / "plots")
        if adaptive_real is not None:
            plot_paths.append(adaptive_real)
        adaptive_drift = plot_adaptive_drift_comparison(traces, out_dir / "plots")
        if adaptive_drift is not None:
            plot_paths.append(adaptive_drift)
        for stream_name in ["real_moa_electricity", "real_insects_abrupt"]:
            hard_core_evolution = plot_local_hard_core_evolution(
                traces,
                out_dir / "plots",
                stream_name=stream_name,
            )
            if hard_core_evolution is not None:
                plot_paths.append(hard_core_evolution)
        if args.learner_sweep:
            from .plots import plot_compute_sweeps

            plot_paths.extend(plot_compute_sweeps(aggregate, out_dir / "plots"))
    else:
        plot_paths = []
    write_markdown_summary(
        out_dir / "README.md",
        summaries,
        aggregate,
        plot_paths,
        gain_oracle=args.gain_oracle,
        include_adaptive=args.adaptive_defensive,
    )


def aggregate_summaries(summaries: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    for row in summaries:
        grouped.setdefault(row["stream"], {}).setdefault(row["algorithm"], []).append(row)
    metrics = [
        "classification_error",
        "brier_loss",
        "constant_brier_loss",
        "randomized_error",
        "hard_core_density",
        "hard_core_edge",
        "weak_residual_corr",
        "self_residual_corr",
        "offline_span_squared_loss",
        "offline_span_representation_norm",
        "offline_span_classification_error",
        "max_single_edge",
        "positive_rate",
        "elapsed_seconds",
        "microseconds_per_round",
    ]
    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stream, by_algo in grouped.items():
        aggregate[stream] = {}
        for algo, rows in by_algo.items():
            aggregate[stream][algo] = {}
            for metric in metrics:
                vals = np.array([float(row[metric]) for row in rows], dtype=float)
                aggregate[stream][algo][metric + "_mean"] = float(np.mean(vals))
                aggregate[stream][algo][metric + "_stderr"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return aggregate


def environment_metadata() -> Dict[str, object]:
    """Record enough environment information to diagnose reproduction drift."""

    packages = {}
    for package in ["numpy", "pandas", "scipy", "matplotlib", "requests"]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "defensive_boosting_version": __version__,
        "git_commit": commit,
        "packages": packages,
        "platform": platform.platform(),
        "python": sys.version,
    }


def write_markdown_summary(
    path: Path,
    summaries: List[Dict],
    aggregate: Dict,
    plot_paths: List[Path],
    *,
    gain_oracle: str,
    include_adaptive: bool,
) -> None:
    adaptive_sentence = (
        " The strongly adaptive variant uses one live copy at each dyadic scale "
        "and exact Adapt-ML-Prod specialist aggregation."
        if include_adaptive
        else ""
    )
    lines = [
        "# Online boosting experiment output",
        "",
        "This directory is generated by `python3 -m experiments.run`.",
        "",
        "The compared algorithms are the Defensive Booster, single-learner unboosted regression and classification controls, online gradient boosting for squared loss, Beygelzimer-Kale-Luo Online BBM, and Chen-Lin-Lu online SmoothBoost with the OCP combiner." + adaptive_sentence + " Labels are encoded in {-1,1}, and prediction scores in [-1,1] are converted to probabilities before scoring. The unboosted classifier and BBM make hard binary predictions, so their Brier score is the Brier score of the induced 0/1 probability forecast.",
        "",
        f"All runs use one global tuning regime rather than per-dataset hyperparameter selection. Linear-loss methods use the `{gain_oracle}` oracle variant. The Defensive Booster uses the parameter-free scalar adaptive-OGD updates specified in the paper. OGB, BBM, and OSBoost use the requested number of weak learners or learner-sweep values; OGB uses stage step `(log N)/N` for `N > 1` and step `1` for `N = 1`. BBM and OSBoost use `gamma=0.1` on real data. On synthetic streams with known correlation edge `delta`, BBM receives `delta` and OSBoost receives `delta/2`, matching the two papers' parameter conventions.",
        "",
        "## Aggregate metrics",
        "",
    ]
    for stream, by_algo in aggregate.items():
        lines.append(f"### {stream}")
        lines.append("")
        lines.append("| algorithm | 0/1 error | Brier | const. Brier | randomized error | hard-core edge | weak corr. | self corr. | LS span loss | LS norm | us/round |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for algo, vals in sorted(by_algo.items(), key=lambda item: _algorithm_sort_key(item[0])):
            lines.append(
                "| {algo} | {err} | {sq} | {base} | {rand} | {edge} | {weak} | {self_corr} | {span} | {span_norm} | {us} |".format(
                    algo=algo,
                    err=_fmt_mean_se(vals, "classification_error"),
                    sq=_fmt_mean_se(vals, "brier_loss"),
                    base=_fmt_mean_se(vals, "constant_brier_loss"),
                    rand=_fmt_mean_se(vals, "randomized_error"),
                    edge=_fmt_mean_se(vals, "hard_core_edge"),
                    weak=_fmt_mean_se(vals, "weak_residual_corr"),
                    self_corr=_fmt_mean_se(vals, "self_residual_corr"),
                    span=_fmt_mean_se(vals, "offline_span_squared_loss"),
                    span_norm=_fmt_mean_se(vals, "offline_span_representation_norm", digits=3),
                    us=_fmt_mean_se(vals, "microseconds_per_round", digits=1),
                )
            )
        lines.append("")
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for p in plot_paths:
            lines.append(f"- `{p}`")
    path.write_text("\n".join(lines) + "\n")


def _fmt_mean_se(vals: Dict[str, float], metric: str, digits: int = 4) -> str:
    mean = vals[f"{metric}_mean"]
    se = vals[f"{metric}_stderr"]
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"


def _algorithm_sort_key(name: str) -> tuple[int, int, str]:
    if name == "defensive":
        return (0, 0, name)
    if name == "adaptive_defensive":
        return (1, 0, name)
    if name == "unboosted":
        return (2, 0, name)
    if name == "unboosted_cls":
        return (3, 0, name)
    match = re.search(r"_N=(\d+)", name)
    n = int(match.group(1)) if match else 0
    if name.startswith("ogb"):
        return (4, n, name)
    if name.startswith("bbm"):
        return (5, n, name)
    if name.startswith("osboost"):
        return (6, n, name)
    return (7, n, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--suite", choices=["synthetic", "real", "drift", "all"], default="synthetic")
    parser.add_argument("--out", default="experiments/out/latest", help="output directory")
    parser.add_argument("--T", type=int, default=3000, help="rounds per stream")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="random seeds")
    parser.add_argument("--real-max-rows", type=int, default=50000)
    parser.add_argument("--real-dim", type=int, default=128)
    parser.add_argument(
        "--drift-max-rows",
        type=int,
        default=None,
        help="optional prefix length for each INSECTS drift variant",
    )
    parser.add_argument("--ogb-learners", type=int, default=100)
    parser.add_argument("--bbm-learners", type=int, default=100)
    parser.add_argument("--osb-learners", type=int, default=100)
    parser.add_argument("--learner-sweep", type=int, nargs="*", default=None, help="run OGB/BBM/OSBoost at these N values")
    parser.add_argument("--stream-filter", nargs="*", default=None, help="substring filters for stream names")
    parser.add_argument(
        "--algorithm-filter",
        nargs="*",
        default=None,
        help="exact algorithm names to run after constructing the requested suite",
    )
    parser.add_argument("--bbm-gamma", type=float, default=0.1)
    parser.add_argument("--osb-gamma", type=float, default=0.1)
    parser.add_argument(
        "--adaptive-defensive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include the strongly adaptive dyadic Defensive Booster",
    )
    parser.add_argument(
        "--gain-oracle",
        choices=["second_order", "first_order"],
        default="second_order",
        help="linear-loss oracle shared by Defensive, Unboosted, and OGB",
    )
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    args = parser.parse_args()
    if args.quick:
        args.T = 600
        args.seeds = [0]
        args.real_max_rows = min(args.real_max_rows, 1500)
        args.real_dim = min(args.real_dim, 64)
        args.drift_max_rows = 1500
        args.ogb_learners = 10
        args.bbm_learners = 20
        args.osb_learners = 20
        if args.out == "experiments/out/latest":
            args.out = "experiments/out/quick"
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.out) / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(args.out) / ".cache"))
    summaries, traces = run_suite(args)
    write_outputs(args, summaries, traces)
    print(f"Wrote {len(summaries)} runs to {args.out}")


if __name__ == "__main__":
    main()
