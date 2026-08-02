"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from .metrics import RunTrace, cumulative


def _algorithm_color(name: str) -> str:
    if name == "defensive":
        return "#1f77b4"
    if name == "adaptive_defensive":
        return "#00897b"
    if name == "unboosted":
        return "#ff7f0e"
    if name == "unboosted_cls":
        return "#2ca02c"
    if name.startswith("brier_aggregator"):
        return "#222222"
    if name.startswith("bbm_vote"):
        return "#c084fc"
    if name.startswith("ogb"):
        return "#d62728"
    if name.startswith("bbm"):
        return "#9467bd"
    if name.startswith("osboost"):
        return "#8c564b"
    return "#7f7f7f"


def _pretty_algorithm(name: str) -> str:
    if name == "defensive":
        return "Defensive"
    if name == "adaptive_defensive":
        return "Adaptive Defensive"
    if name == "unboosted":
        return "Unboosted reg."
    if name == "unboosted_cls":
        return "Unboosted cls."
    if name.startswith("brier_aggregator"):
        return "Brier aggregator"
    if name.startswith("bbm_vote"):
        return "Online BBM vote"
    if name.startswith("ogb"):
        return "OGB"
    if name.startswith("bbm"):
        return "Online BBM"
    if name.startswith("osboost"):
        return "OSBoost"
    return name


def _pretty_stream(name: str) -> str:
    if name.startswith("planted_decoy_margin"):
        return "Planted-decoy stream"
    if name.startswith("mixed_linear_random_label_mixture"):
        return "Random-label mixture"
    if name.startswith("group_subset_positive") or name.startswith("group_subset_heterogeneous"):
        return "Group-subset stream"
    if name.startswith("linear_span_fallback"):
        return "Linear-span stream"
    if name.startswith("random_labels"):
        return "Random labels"
    if name.startswith("heterogeneous_margin"):
        return "Heterogeneous-margin weak rule"
    if name == "real_bank_marketing":
        return "Bank Marketing"
    if name == "real_moa_electricity":
        return "Electricity"
    if name == "real_moa_airlines":
        return "Airlines"
    if name == "real_uci_occupancy":
        return "Occupancy"
    if name == "real_insects_abrupt":
        return "INSECTS abrupt"
    if name == "real_insects_incremental_gradual":
        return "INSECTS incremental-gradual"
    if name == "real_insects_incremental_abrupt":
        return "INSECTS incremental-abrupt"
    if name == "real_insects_incremental_recurring":
        return "INSECTS incremental-recurring"
    if name == "real_insects_incremental":
        return "INSECTS incremental"
    return name.replace("_", " ")


def _algorithm_sort_key(name: str) -> Tuple[int, int, str]:
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
    if name.startswith("brier_aggregator"):
        return (7, n, name)
    if name.startswith("bbm_vote"):
        return (8, n, name)
    return (9, n, name)


def plot_traces(traces: Iterable[RunTrace], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    traces_by_stream = {}
    for trace in traces:
        traces_by_stream.setdefault(trace.stream.name, []).append(trace)

    for stream_name, stream_traces in traces_by_stream.items():
        for metric_name, attr, ylabel in [
            ("classification_error", "mistakes", "cumulative 0/1 error"),
            ("brier_loss", "brier_losses", "cumulative Brier loss"),
            ("randomized_error", "randomized_errors", "cumulative randomized error"),
        ]:
            plt.figure(figsize=(7.2, 4.5))
            by_algo = {}
            for trace in stream_traces:
                if trace.algorithm.startswith("bbm_vote"):
                    continue
                if metric_name == "brier_loss" and trace.algorithm == "unboosted_cls":
                    continue
                by_algo.setdefault(trace.algorithm, []).append(cumulative(getattr(trace, attr)))
            for algo, curves in sorted(by_algo.items(), key=lambda item: _algorithm_sort_key(item[0])):
                min_len = min(curve.size for curve in curves)
                stacked = np.vstack([curve[:min_len] for curve in curves])
                xs = np.arange(1, min_len + 1)
                mean = np.mean(stacked, axis=0)
                stderr = np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros(min_len)
                line = plt.plot(
                    xs,
                    mean,
                    label=_pretty_algorithm(algo),
                    color=_algorithm_color(algo),
                    linewidth=1.8,
                )[0]
                if stacked.shape[0] > 1:
                    plt.fill_between(xs, mean - stderr, mean + stderr, color=line.get_color(), alpha=0.16, linewidth=0)
            plt.xlabel("round")
            plt.ylabel(ylabel)
            plt.title(_pretty_stream(stream_name))
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            path = out_dir / f"{stream_name}__{metric_name}.png"
            plt.savefig(path, dpi=180, facecolor="white", transparent=False)
            plt.close()
            paths.append(path)
        defensive_traces = [trace for trace in stream_traces if trace.algorithm == "defensive"]
        if defensive_traces:
            paths.append(_plot_certificate_stream(stream_name, defensive_traces, out_dir))
    return paths


def _plot_certificate_stream(stream_name: str, traces: List[RunTrace], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.45), sharex=True)
    series = [
        ("weak_residual_corrs", r"$\sup_h |t^{-1}\sum h r|$"),
        ("self_residual_corrs", r"$|t^{-1}\sum \mu r|$"),
        ("hard_core_densities", r"mistake density $\rho_w(t)$"),
        ("hard_core_edges", r"weak-class edge under $w$"),
    ]
    for ax, (attr, ylabel) in zip(axes, series):
        curves = [getattr(trace, attr) for trace in traces]
        min_len = min(curve.size for curve in curves)
        stacked = np.vstack([curve[:min_len] for curve in curves])
        xs = np.arange(1, min_len + 1)
        mean = np.mean(stacked, axis=0)
        stderr = np.std(stacked, axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros(min_len)
        line = ax.plot(xs, mean, color="#1f77b4", linewidth=1.8)[0]
        if stacked.shape[0] > 1:
            ax.fill_between(xs, mean - stderr, mean + stderr, color=line.get_color(), alpha=0.16, linewidth=0)
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Defensive Booster diagnostics: {_pretty_stream(stream_name)}", y=1.03)
    fig.tight_layout()
    path = out_dir / f"{stream_name}__certificate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_compute_sweeps(aggregate: Dict[str, Dict[str, Dict[str, float]]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for stream_name, by_algo in aggregate.items():
        ogb = _sweep_points(by_algo, "ogb")
        bbm = _sweep_points(by_algo, "bbm")
        osboost = _sweep_points(by_algo, "osboost")
        aggregator = _sweep_points(by_algo, "brier_aggregator")
        if not ogb and not bbm and not osboost and not aggregator:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
        for ax, metric, ylabel in [
            (axes[0], "brier_loss", "final Brier loss"),
            (axes[1], "randomized_error", "final randomized error"),
        ]:
            for label, points, color in [
                ("OGB", ogb, "#d62728"),
                ("Online BBM", bbm, "#9467bd"),
                ("OSBoost", osboost, "#8c564b"),
                ("Brier aggregator", aggregator, "#222222"),
            ]:
                if points:
                    multiplier = 3.0 if label == "Brier aggregator" else 1.0
                    ns = multiplier * np.array([p[0] for p in points], dtype=float)
                    means = np.array([p[1][f"{metric}_mean"] for p in points], dtype=float)
                    ses = np.array([p[1][f"{metric}_stderr"] for p in points], dtype=float)
                    ax.errorbar(ns, means, yerr=ses, label=label, color=color, marker="o", capsize=2.5)
            for baseline, color, linestyle in [
                ("defensive", "#1f77b4", "--"),
                ("unboosted", "#ff7f0e", ":"),
                ("unboosted_cls", "#2ca02c", "-."),
            ]:
                if metric == "brier_loss" and baseline == "unboosted_cls":
                    continue
                if baseline in by_algo:
                    vals = by_algo[baseline]
                    ax.axhline(vals[f"{metric}_mean"], color=color, linestyle=linestyle, linewidth=1.6, label=_pretty_algorithm(baseline))
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 20, 100, 300])
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.set_xlabel("total maintained weak learners")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(_pretty_stream(stream_name), y=0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=4,
            fontsize=8,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
        path = out_dir / f"{stream_name}__compute_sweep.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", transparent=False)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_real_summary(
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
    regression_aggregate: Dict[str, Dict[str, Dict[str, float]]] | None = None,
) -> Path | None:
    """Plot the real binary results, optionally with the regression comparison."""

    stream_order = [
        ("real_bank_marketing", "Bank"),
        ("real_moa_electricity", "Electricity"),
        ("real_moa_airlines", "Airlines"),
        ("real_uci_occupancy", "Occupancy"),
    ]
    if not all(stream in aggregate for stream, _ in stream_order):
        return None

    algorithm_order = [
        ("defensive", "Defensive (1)", "#1f77b4"),
        ("unboosted", "Unboosted reg. (1)", "#ff7f0e"),
        ("ogb_N=100", "OGB (100)", "#d62728"),
        ("bbm_N=100", "Online BBM (100)", "#9467bd"),
        ("osboost_N=100", "OSBoost (100)", "#8c564b"),
        ("brier_aggregator_N=100", "Brier aggregator (300)", "#222222"),
    ]
    if not all(
        algorithm in aggregate[stream]
        for stream, _ in stream_order
        for algorithm, _, _ in algorithm_order
    ):
        return None

    regression_order = [
        ("regression_appliances_energy", "Appliance energy"),
        ("regression_bike_demand", "Bike demand"),
        ("regression_interstate_traffic", "Interstate traffic"),
    ]
    regression_algorithms = [
        ("defensive", "Defensive (1)", "#1f77b4"),
        ("ogb_N=100", "OGB (100)", "#d62728"),
    ]
    show_regression = regression_aggregate is not None
    if show_regression and not all(
        algorithm in regression_aggregate[stream]
        for stream, _ in regression_order
        for algorithm, _, _ in regression_algorithms
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if show_regression:
        fig, (ax, regression_ax) = plt.subplots(
            2,
            1,
            figsize=(10.2, 7.2),
            gridspec_kw={"height_ratios": [1.8, 1.0], "hspace": 0.34},
        )
    else:
        fig, ax = plt.subplots(figsize=(10.2, 4.7))
        regression_ax = None

    x = np.arange(len(stream_order), dtype=float)
    width = 0.135
    for index, (algorithm, label, color) in enumerate(algorithm_order):
        ratios = []
        errors = []
        for stream, _ in stream_order:
            by_algo = aggregate[stream]
            best = min(
                by_algo[name]["brier_loss_mean"]
                for name, _, _ in algorithm_order
            )
            values = by_algo[algorithm]
            ratios.append(values["brier_loss_mean"] / best)
            errors.append(values["brier_loss_stderr"] / best)
        offset = (index - (len(algorithm_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            ratios,
            width,
            yerr=errors if any(error > 0.0 for error in errors) else None,
            capsize=2 if any(error > 0.0 for error in errors) else 0,
            label=label,
            color=color,
        )

    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in stream_order])
    ax.set_ylabel("Brier loss / best plotted method")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8.5, title="Algorithm (weak learners)")
    if show_regression:
        ax.set_title("Binary probability forecasts", loc="left", fontweight="medium")

        regression_x = np.arange(len(regression_order), dtype=float)
        regression_width = 0.28
        for index, (algorithm, _, color) in enumerate(regression_algorithms):
            ratios = []
            for stream, _ in regression_order:
                by_algo = regression_aggregate[stream]
                best = min(
                    by_algo[name]["scaled_mse"]
                    for name, _, _ in regression_algorithms
                )
                ratios.append(by_algo[algorithm]["scaled_mse"] / best)
            offset = (index - 0.5) * regression_width
            bars = regression_ax.bar(
                regression_x + offset,
                ratios,
                regression_width,
                color=color,
            )
            if algorithm == "ogb_N=100":
                for bar, ratio in zip(bars, ratios):
                    regression_ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        ratio + 0.025,
                        f"{ratio:.2f}x",
                        ha="center",
                        va="bottom",
                        color="#a32121",
                        fontsize=9,
                        fontweight="medium",
                    )
        regression_ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
        regression_ax.set_xticks(regression_x)
        regression_ax.set_xticklabels([label for _, label in regression_order])
        regression_ax.set_ylabel("MSE / best plotted method")
        regression_ax.set_ylim(0.0, 1.55)
        regression_ax.set_yticks([0.0, 0.5, 1.0, 1.5])
        regression_ax.set_title("Bounded regression", loc="left", fontweight="medium")
        regression_ax.grid(axis="y", alpha=0.25)
        fig.subplots_adjust(left=0.105, right=0.985, top=0.975, bottom=0.075)
    else:
        fig.tight_layout()
    path = out_dir / "real_summary_brier.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_bbm_vote_diagnostic(
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> Path | None:
    """Compare Online BBM's specified hard output with its raw vote score."""

    streams = [
        ("planted_decoy_margin_d=200_gamma=0.12", "Planted decoy"),
        ("group_subset_heterogeneous_m=10_k=6_delta=0.6", "Group subset"),
    ]
    algorithms = [
        ("bbm_N=100", "Hard output", "#9467bd"),
        ("bbm_vote_N=100", "Normalized vote", "#c084fc"),
    ]
    if not all(
        stream in aggregate and all(name in aggregate[stream] for name, _, _ in algorithms)
        for stream, _ in streams
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(streams), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55))
    for ax, metric, ylabel in [
        (axes[0], "brier_loss", "Brier loss"),
        (axes[1], "randomized_error", "randomized error"),
    ]:
        for index, (algorithm, label, color) in enumerate(algorithms):
            means = [aggregate[stream][algorithm][f"{metric}_mean"] for stream, _ in streams]
            ses = [aggregate[stream][algorithm][f"{metric}_stderr"] for stream, _ in streams]
            offset = (index - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=ses, capsize=2.5, label=label, color=color)
            for bar, mean in zip(bars, means):
                if mean < 1e-4:
                    ax.annotate(
                        r"$<10^{-4}$",
                        (bar.get_x() + bar.get_width() / 2.0, 0.0),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in streams])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    path = out_dir / "bbm_vote_diagnostic.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_adaptive_real_comparison(
    traces: Iterable[RunTrace],
    out_dir: Path,
) -> Path | None:
    """Compare the basic and strongly adaptive forecasters on all real streams."""

    stream_order = [
        ("real_bank_marketing", "Bank"),
        ("real_moa_electricity", "Electricity"),
        ("real_moa_airlines", "Airlines"),
        ("real_uci_occupancy", "Occupancy"),
    ]
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for trace in traces:
        if trace.algorithm not in {"defensive", "adaptive_defensive"}:
            continue
        grouped.setdefault(trace.stream.name, {}).setdefault(trace.algorithm, []).append(
            trace.brier_losses
        )
    if not all(
        stream in grouped and {"defensive", "adaptive_defensive"} <= set(grouped[stream])
        for stream, _ in stream_order
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    window = 1000
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    for ax, (stream, title) in zip(axes.flat, stream_order):
        for algorithm in ["defensive", "adaptive_defensive"]:
            curves = grouped[stream][algorithm]
            length = min(curve.size for curve in curves)
            kernel = np.ones(window, dtype=float) / window
            rolling = np.vstack(
                [np.convolve(curve[:length], kernel, mode="valid") for curve in curves]
            )
            mean = np.mean(rolling, axis=0)
            ax.plot(
                np.arange(window, length + 1),
                mean,
                color=_algorithm_color(algorithm),
                linewidth=1.8,
                label=_pretty_algorithm(algorithm),
            )
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_ylabel("trailing 1,000-round Brier loss")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path = out_dir / "adaptive_real_brier.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_adaptive_drift_comparison(
    traces: Iterable[RunTrace],
    out_dir: Path,
) -> Path | None:
    """Compare local loss on the abrupt and incremental-gradual INSECTS streams."""

    stream_order = [
        ("real_insects_abrupt", "Abrupt drift"),
        ("real_insects_incremental_gradual", "Incremental-gradual drift"),
    ]
    change_points = {
        "real_insects_abrupt": [14352, 19500, 33240, 38682, 39510],
        "real_insects_incremental_gradual": [14028],
    }
    grouped: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for trace in traces:
        if trace.algorithm not in {"defensive", "adaptive_defensive"}:
            continue
        grouped.setdefault(trace.stream.name, {}).setdefault(trace.algorithm, []).append(
            trace.brier_losses
        )
    if not all(
        stream in grouped and {"defensive", "adaptive_defensive"} <= set(grouped[stream])
        for stream, _ in stream_order
    ):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    window = 1000
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), sharey=True)
    for ax, (stream, title) in zip(axes, stream_order):
        for algorithm in ["defensive", "adaptive_defensive"]:
            curves = grouped[stream][algorithm]
            length = min(curve.size for curve in curves)
            kernel = np.ones(window, dtype=float) / window
            rolling = np.vstack(
                [np.convolve(curve[:length], kernel, mode="valid") for curve in curves]
            )
            ax.plot(
                np.arange(window, length + 1),
                np.mean(rolling, axis=0),
                color=_algorithm_color(algorithm),
                linewidth=1.8,
                label=_pretty_algorithm(algorithm),
            )
        for point in change_points[stream]:
            ax.axvline(point, color="#666666", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("trailing 1,000-round Brier loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    path = out_dir / "adaptive_insects_brier.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def plot_local_hard_core_evolution(
    traces: Iterable[RunTrace],
    out_dir: Path,
    stream_name: str = "real_moa_electricity",
) -> Path | None:
    """Plot density and edge of local mistake weightings across time and scale."""

    candidates = [
        trace
        for trace in traces
        if trace.stream.name == stream_name and trace.algorithm == "adaptive_defensive"
    ]
    if not candidates:
        return None
    trace = candidates[0]
    stream = trace.stream
    y = stream.y.astype(float)
    weights = (1.0 - y * trace.scores) / 2.0
    weighted_features = (weights * y)[:, None] * stream.X
    mass_prefix = np.concatenate([[0.0], np.cumsum(weights)])
    vector_prefix = np.vstack(
        [np.zeros((1, stream.dim), dtype=float), np.cumsum(weighted_features, axis=0)]
    )

    window_lengths = np.asarray([256, 512, 1024, 2048, 4096, 8192], dtype=int)
    stride = 64
    endpoints = np.arange(stride, stream.T + 1, stride, dtype=int)
    density = np.full((window_lengths.size, endpoints.size), np.nan, dtype=float)
    edge = np.full_like(density, np.nan)
    for row, length in enumerate(window_lengths):
        valid = endpoints >= length
        ends = endpoints[valid]
        starts = ends - length
        masses = mass_prefix[ends] - mass_prefix[starts]
        vectors = vector_prefix[ends] - vector_prefix[starts]
        density[row, valid] = masses / length
        if stream.weak_type == "linear":
            numerators = np.linalg.norm(vectors, axis=1)
        else:
            numerators = np.max(np.abs(vectors), axis=1)
        edge[row, valid] = np.divide(
            numerators,
            masses,
            out=np.zeros_like(numerators),
            where=masses > 1e-12,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    change_points = (
        [14352, 19500, 33240, 38682, 39510]
        if stream_name == "real_insects_abrupt"
        else []
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65), sharey=True)
    for ax, matrix, title, colorbar_label, cmap_name in [
        (axes[0], density, "Mistake-weight density", r"density $|I|^{-1}\sum_{s\in I}w_s$", "viridis"),
        (axes[1], edge, "Normalized weak-class edge", r"edge $\mathrm{edge}_{\mathcal{H}}(w)$", "magma"),
    ]:
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(endpoints[0], endpoints[-1], -0.5, window_lengths.size - 0.5),
            cmap=cmap,
            vmin=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("interval endpoint")
        ax.set_yticks(np.arange(window_lengths.size))
        ax.set_yticklabels([f"{length:,}" for length in window_lengths])
        for point in change_points:
            ax.axvline(point, color="white", linewidth=0.7, linestyle=":", alpha=0.9)
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(colorbar_label)
    axes[0].set_ylabel("trailing interval length")
    fig.tight_layout()
    suffix = {
        "real_moa_electricity": "electricity",
        "real_insects_abrupt": "insects_abrupt",
    }.get(stream_name, re.sub(r"^real_", "", stream_name))
    path = out_dir / f"adaptive_{suffix}_hard_core_evolution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    return path


def _sweep_points(by_algo: Dict[str, Dict[str, float]], prefix: str) -> List[Tuple[int, Dict[str, float]]]:
    points = []
    for algo, vals in by_algo.items():
        if not algo.startswith(prefix + "_N="):
            continue
        match = re.search(r"_N=(\d+)", algo)
        if match:
            points.append((int(match.group(1)), vals))
    return sorted(points, key=lambda item: item[0])
