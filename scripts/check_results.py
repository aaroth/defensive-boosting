#!/usr/bin/env python3
"""Compare a reproduced aggregate.json with a tracked reference result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


IGNORED_METRIC_PREFIXES = ("elapsed_seconds_", "microseconds_per_round_")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def compare_results(
    actual_path: Path,
    reference_path: Path,
    *,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-9,
) -> list[str]:
    actual = load_json(actual_path)
    reference = load_json(reference_path)
    failures: list[str] = []

    if set(actual) != set(reference):
        failures.append(
            f"stream keys differ: actual={sorted(actual)}, reference={sorted(reference)}"
        )
        return failures

    for stream in sorted(reference):
        if set(actual[stream]) != set(reference[stream]):
            failures.append(
                f"{stream}: algorithm keys differ: actual={sorted(actual[stream])}, "
                f"reference={sorted(reference[stream])}"
            )
            continue
        for algorithm in sorted(reference[stream]):
            actual_metrics = actual[stream][algorithm]
            reference_metrics = reference[stream][algorithm]
            for metric, expected in sorted(reference_metrics.items()):
                if metric.startswith(IGNORED_METRIC_PREFIXES):
                    continue
                if metric not in actual_metrics:
                    failures.append(f"{stream}/{algorithm}: missing metric {metric}")
                    continue
                observed = actual_metrics[metric]
                if not math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                ):
                    failures.append(
                        f"{stream}/{algorithm}/{metric}: observed {observed}, "
                        f"expected {expected}"
                    )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="aggregate.json produced by a rerun")
    parser.add_argument("reference", type=Path, help="tracked reference aggregate.json")
    parser.add_argument("--rtol", type=float, default=1e-7, help="relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = compare_results(
        args.actual,
        args.reference,
        relative_tolerance=args.rtol,
        absolute_tolerance=args.atol,
    )
    if failures:
        print(f"Reproduction check failed with {len(failures)} difference(s):")
        for failure in failures[:50]:
            print(f"  - {failure}")
        if len(failures) > 50:
            print(f"  - ... and {len(failures) - 50} more")
        raise SystemExit(1)
    print(f"Reproduction check passed: {args.actual}")


if __name__ == "__main__":
    main()
