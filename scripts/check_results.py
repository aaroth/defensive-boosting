#!/usr/bin/env python3
"""Compare a reproduced aggregate.json with a tracked reference result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


IGNORED_METRIC_PREFIXES = (
    "elapsed_seconds",
    "microseconds_per_round",
)


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

    def compare_node(observed: Any, expected: Any, path: tuple[str, ...]) -> None:
        location = "/".join(path) or "<root>"
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                failures.append(f"{location}: expected a mapping, observed {type(observed).__name__}")
                return
            observed_keys = set(observed)
            expected_keys = set(expected)
            if observed_keys != expected_keys:
                missing = sorted(expected_keys - observed_keys)
                extra = sorted(observed_keys - expected_keys)
                failures.append(f"{location}: key mismatch; missing={missing}, extra={extra}")
                return
            for key in sorted(expected):
                if key.startswith(IGNORED_METRIC_PREFIXES):
                    continue
                compare_node(observed[key], expected[key], (*path, key))
            return

        try:
            close = math.isclose(
                float(observed),
                float(expected),
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
        except (TypeError, ValueError):
            close = observed == expected
        if not close:
            failures.append(f"{location}: observed {observed}, expected {expected}")

    compare_node(actual, reference, ())
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
