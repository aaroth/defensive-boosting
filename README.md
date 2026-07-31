# Defensive Boosting for Online Probabilistic Forecasting

This repository contains the experiment code for **Defensive
Boosting for Online Probabilistic Forecasting** by Georgy Noarov and Aaron
Roth.  The implementation includes the Defensive Booster, its strongly
adaptive variant, the comparison algorithms used in the paper, all synthetic
generators, and loaders for the public real-data streams.

The experiment pipeline does not redistribute datasets.  It downloads each
public dataset from its original source, preserves the released order, and
caches the raw and processed files locally under `experiments/data/`.

## Quick start

The code supports Python 3.9 or later.  From a fresh clone:

```bash
git clone https://github.com/aaroth/defensive-boosting.git
cd defensive-boosting
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest experiments.test_core experiments.test_adaptive
python -m experiments.run --quick
```

The final command runs a small synthetic smoke test and writes its outputs to
`experiments/out/quick/`.  It does not download real data.

`requirements.txt` pins the package versions used for the reported results.
`pyproject.toml` gives looser compatible ranges for development installations.

## Reproduce the paper

Run every reported experiment, verify the numerical outputs against the
tracked reference aggregates, and update the paper figures with:

```bash
python scripts/reproduce_paper.py all
```

The full run evaluates 100-learner ensemble baselines over all streams and 20
synthetic seeds.  Its runtime is hardware- and network-dependent; the ensemble
baselines dominate the computation.  Results are written below
`experiments/out/paper/`, while downloaded datasets are cached below
`experiments/data/`.

Each experiment family can also be run independently:

```bash
python scripts/reproduce_paper.py synthetic
python scripts/reproduce_paper.py sweep
python scripts/reproduce_paper.py real
python scripts/reproduce_paper.py adaptive-real
python scripts/reproduce_paper.py drift
```

The targets correspond to the paper as follows:

| Target | Paper results |
| --- | --- |
| `synthetic` | Main complementary-regime and hard-core diagnostics; complete synthetic appendix |
| `sweep` | Ensemble-size comparison on the group-subset stream |
| `real` | Four naturally ordered real streams and the introductory summary figure |
| `adaptive-real` | Strongly adaptive comparison on the four real streams |
| `drift` | INSECTS controlled-drift results and local hard-core diagnostics |

By default, the script runs the unit tests, compares each `aggregate.json`
with the corresponding file in `experiments/reference/`, and copies the
regenerated paper plots into `figures/`.  Use `--no-verify`
to skip the numerical check, `--no-sync` to leave the checked-in figures
unchanged, or `--dry-run` to print the exact commands without running them.

For a direct invocation of the underlying runner, use:

```bash
python -m experiments.run --help
```

The full algorithm definitions, hyperparameter conventions, stream
constructions, and individual commands are documented in
[`experiments/README.md`](experiments/README.md).

## Outputs and provenance

Every run directory contains:

- `run_config.json`: all command-line parameters;
- `environment.json`: the git commit, Python version, platform, and dependency versions;
- `summary.csv`: one row per dataset, algorithm, and seed;
- `aggregate.json`: means and standard errors;
- `traces.npz`: per-round predictions and diagnostics;
- `plots/`: cumulative performance and certificate plots;
- `README.md`: a generated numerical summary.

Runtime values depend on hardware and are excluded from reference-result
comparisons.  All predictive metrics are checked with a small floating-point
tolerance.  The real streams are never shuffled, and feature standardization
uses only contexts observed before the current round.

## Repository layout

- `experiments/algorithms.py`: Defensive Booster and comparison algorithms.
- `experiments/weak_learners.py`: online weak-class oracles.
- `experiments/streams.py`: controlled synthetic streams.
- `experiments/real_streams.py`: public dataset downloaders and online preprocessing.
- `experiments/run.py`: experiment runner and output generation.
- `scripts/reproduce_paper.py`: exact paper-level orchestration.
- `scripts/check_results.py`: reference-result comparison.
- `experiments/reference/`: aggregate metrics from the reported runs.

## License and citation

The code is released under the MIT License.  Citation metadata is provided in
[`CITATION.cff`](CITATION.cff).
