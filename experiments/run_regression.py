"""Run chronological bounded-regression comparisons for the paper appendix."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .algorithms import (
    DefensiveBooster,
    OnlineGradientBoosting,
    OnlineSquaredLossRegressor,
)
from .regression_streams import RegressionStreamData, regression_streams
from .run import environment_metadata
from .weak_learners import LinearSecondOrderOracle


DISPLAY_NAMES = {
    "defensive": "Defensive Booster",
    "unboosted": "Unboosted regression",
    "ogb_N=100": "Online gradient boosting",
}
COLORS = {
    "defensive": "#1f77b4",
    "unboosted": "#2ca02c",
    "ogb_N=100": "#d62728",
}


def _algorithm_suite(
    stream: RegressionStreamData,
    n_learners: int,
    ogb_step_multiplier: float,
) -> list[Any]:
    def weak_factory() -> LinearSecondOrderOracle:
        return LinearSecondOrderOracle(dim=stream.dim, radius=1.0)

    return [
        DefensiveBooster(weak_factory, name="defensive"),
        OnlineSquaredLossRegressor(weak_factory, name="unboosted"),
        OnlineGradientBoosting(
            weak_factory,
            n_learners=n_learners,
            eta=ogb_step_multiplier
            * (1.0 if n_learners == 1 else np.log(n_learners) / n_learners),
            name=f"ogb_N={n_learners}",
        ),
    ]


def _run_algorithm(
    algorithm: Any,
    stream: RegressionStreamData,
) -> tuple[dict[str, float | str | int], dict[str, np.ndarray]]:
    start = time.perf_counter()
    for x_t, y_t in zip(stream.X_init, stream.y_init):
        algorithm.step(x_t, 2.0 * float(y_t) - 1.0)

    predictions = np.empty(stream.T, dtype=float)
    for t, (x_t, y_t) in enumerate(zip(stream.X, stream.y)):
        result = algorithm.step(x_t, 2.0 * float(y_t) - 1.0)
        predictions[t] = 0.5 * (float(result["score"]) + 1.0)
    elapsed = time.perf_counter() - start

    squared_losses = (stream.y - predictions) ** 2
    raw_predictions = stream.to_raw_scale(predictions)
    raw_squared_losses = (stream.raw_y - raw_predictions) ** 2
    row: dict[str, float | str | int] = {
        "stream": stream.name,
        "algorithm": algorithm.name,
        "initialization_rounds": stream.initialization_rounds,
        "evaluation_rounds": stream.T,
        "dimension": stream.dim,
        "target_min": stream.target_min,
        "target_max": stream.target_max,
        "target_clipped_fraction": stream.clipped_fraction,
        "scaled_mse": float(np.mean(squared_losses)),
        "scaled_rmse": float(np.sqrt(np.mean(squared_losses))),
        "raw_rmse": float(np.sqrt(np.mean(raw_squared_losses))),
        "raw_mae": float(np.mean(np.abs(stream.raw_y - raw_predictions))),
        "elapsed_seconds": float(elapsed),
        "microseconds_per_round": float(
            1_000_000.0 * elapsed / (stream.initialization_rounds + stream.T)
        ),
    }
    trace = {
        "predictions": predictions,
        "squared_losses": squared_losses,
        "raw_predictions": raw_predictions,
        "raw_squared_losses": raw_squared_losses,
    }
    return row, trace


def _causal_mean(
    stream: RegressionStreamData,
) -> tuple[dict[str, float | str | int], dict[str, np.ndarray]]:
    total = float(np.sum(stream.y_init))
    count = stream.initialization_rounds
    predictions = np.empty(stream.T, dtype=float)
    for t, y_t in enumerate(stream.y):
        predictions[t] = total / max(count, 1)
        total += float(y_t)
        count += 1
    predictions = np.clip(predictions, 0.0, 1.0)
    squared_losses = (stream.y - predictions) ** 2
    raw_predictions = stream.to_raw_scale(predictions)
    raw_squared_losses = (stream.raw_y - raw_predictions) ** 2
    row: dict[str, float | str | int] = {
        "stream": stream.name,
        "algorithm": "causal_mean",
        "initialization_rounds": stream.initialization_rounds,
        "evaluation_rounds": stream.T,
        "dimension": stream.dim,
        "target_min": stream.target_min,
        "target_max": stream.target_max,
        "target_clipped_fraction": stream.clipped_fraction,
        "scaled_mse": float(np.mean(squared_losses)),
        "scaled_rmse": float(np.sqrt(np.mean(squared_losses))),
        "raw_rmse": float(np.sqrt(np.mean(raw_squared_losses))),
        "raw_mae": float(np.mean(np.abs(stream.raw_y - raw_predictions))),
        "elapsed_seconds": 0.0,
        "microseconds_per_round": 0.0,
    }
    return row, {
        "predictions": predictions,
        "squared_losses": squared_losses,
        "raw_predictions": raw_predictions,
        "raw_squared_losses": raw_squared_losses,
    }


def _write_plot(
    streams: list[RegressionStreamData],
    traces: dict[tuple[str, str], dict[str, np.ndarray]],
    output: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.parent / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_names = {
        **DISPLAY_NAMES,
        "causal_mean": "Causal mean",
    }
    colors = {**COLORS, "causal_mean": "#7f7f7f"}
    fig, axes = plt.subplots(1, len(streams), figsize=(12.2, 3.45), sharey=False)
    if len(streams) == 1:
        axes = [axes]
    titles = {
        "regression_appliances_energy": "Appliance energy",
        "regression_bike_demand": "Bike demand",
        "regression_interstate_traffic": "Interstate traffic",
    }
    available = {algorithm for _, algorithm in traces}
    ogb_names = sorted(name for name in available if name.startswith("ogb_N="))
    algorithm_order = ["defensive", *ogb_names, "unboosted", "causal_mean"]
    for ax, stream in zip(axes, streams):
        for algorithm in algorithm_order:
            key = (stream.name, algorithm)
            if key not in traces:
                continue
            losses = traces[key]["squared_losses"]
            rounds = np.arange(1, losses.size + 1, dtype=float)
            running = np.cumsum(losses) / rounds
            ax.plot(
                rounds,
                running,
                color=colors.get(algorithm, "#d62728"),
                linewidth=1.7,
                label=display_names.get(algorithm, "Online gradient boosting"),
            )
        ax.set_title(titles.get(stream.name, stream.name))
        ax.set_xlabel("Evaluation rounds")
        ax.set_ylabel("Running mean squared error")
        ax.grid(alpha=0.22)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_markdown(
    path: Path,
    streams: list[RegressionStreamData],
    rows: list[dict[str, float | str | int]],
    initialization_fraction: float,
) -> None:
    by_stream = {stream.name: stream for stream in streams}
    lines = [
        "# Chronological regression experiment output",
        "",
        "Each dataset remains in timestamp order. Fixed, round-number target "
        f"bounds are declared before the run. The first "
        f"{100 * initialization_fraction:g}% "
        "of observations "
        "initialize every learner; metrics use only the remaining observations. "
        "Numeric contexts are standardized from earlier contexts only.",
        "",
    ]
    for stream_name, stream in by_stream.items():
        lines.extend(
            [
                f"## {stream_name}",
                "",
                stream.description,
                "",
                f"Target interval: `{stream.target_min:g}` to `{stream.target_max:g}` "
                f"{stream.target_unit}; later clipping fraction: `{stream.clipped_fraction:.4f}`.",
                "",
                "| algorithm | scaled MSE | raw RMSE | raw MAE | us/round |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            if row["stream"] != stream_name:
                continue
            lines.append(
                "| {algorithm} | {mse:.6f} | {rmse:.3f} | {mae:.3f} | {runtime:.1f} |".format(
                    algorithm=row["algorithm"],
                    mse=float(row["scaled_mse"]),
                    rmse=float(row["raw_rmse"]),
                    mae=float(row["raw_mae"]),
                    runtime=float(row["microseconds_per_round"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/out/regression"))
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--ogb-learners", type=int, default=100)
    parser.add_argument("--ogb-step-multiplier", type=float, default=1.0)
    parser.add_argument("--initialization-fraction", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--dataset",
        nargs="*",
        default=None,
        help="optional substrings selecting datasets",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    streams = regression_streams(
        max_rows=args.max_rows,
        dim=args.dim,
        initialization_fraction=args.initialization_fraction,
        progress=not args.no_progress,
    )
    if args.dataset:
        streams = [
            stream
            for stream in streams
            if any(fragment in stream.name for fragment in args.dataset)
        ]
    if not streams:
        raise ValueError("no regression datasets matched the requested filters")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []
    traces: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for stream in streams:
        mean_row, mean_trace = _causal_mean(stream)
        rows.append(mean_row)
        traces[(stream.name, "causal_mean")] = mean_trace
        for algorithm in _algorithm_suite(
            stream,
            args.ogb_learners,
            args.ogb_step_multiplier,
        ):
            if not args.no_progress:
                print(f"Running {stream.name}: {algorithm.name}", flush=True)
            row, trace = _run_algorithm(algorithm, stream)
            rows.append(row)
            traces[(stream.name, algorithm.name)] = trace
            if not args.no_progress:
                print(
                    f"  MSE={float(row['scaled_mse']):.6f}, "
                    f"raw RMSE={float(row['raw_rmse']):.3f}, "
                    f"time={float(row['elapsed_seconds']):.2f}s",
                    flush=True,
                )

    with (args.out / "run_config.json").open("w") as handle:
        config = vars(args).copy()
        config["out"] = str(config["out"])
        json.dump(config, handle, indent=2, sort_keys=True)
    with (args.out / "environment.json").open("w") as handle:
        json.dump(environment_metadata(), handle, indent=2, sort_keys=True)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        stream.name: {
            str(row["algorithm"]): {
                key: value
                for key, value in row.items()
                if key not in {"stream", "algorithm"}
            }
            for row in rows
            if row["stream"] == stream.name
        }
        for stream in streams
    }
    with (args.out / "aggregate.json").open("w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
    payload = {}
    for (stream_name, algorithm), trace in traces.items():
        for key, values in trace.items():
            payload[f"{stream_name}__{algorithm}__{key}"] = values
    np.savez_compressed(args.out / "traces.npz", **payload)
    if not args.no_plots:
        _write_plot(streams, traces, args.out / "plots" / "regression_mse.png")
    _write_markdown(
        args.out / "README.md",
        streams,
        rows,
        args.initialization_fraction,
    )
    print(f"Wrote regression results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
