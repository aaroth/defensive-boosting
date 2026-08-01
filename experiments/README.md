# Online Boosting Experiments

For environment setup and the one-command paper reproduction workflow, see
the [repository README](../README.md).  The commands below expose the
underlying runner directly.

This package compares seven main online binary-prediction procedures on shared datasets and includes the strongly adaptive extension studied in the appendix:

- `defensive`: the Defensive Booster in `boosting.tex`.
- `adaptive_defensive`: the strongly adaptive Defensive Booster, enabled with
  `--adaptive-defensive`.
- `unboosted`: a single online squared-loss regressor over the same weak class.
- `unboosted_cls`: a single binary classification learner of the kind used by the classification boosters.
- `ogb`: online gradient boosting for squared loss, specialized from Beygelzimer, Hazan, Kale, and Luo.
- `bbm`: Beygelzimer, Kale, and Luo Online BBM, the optimal-rate binary online boosting baseline.
- `osboost`: Chen, Lin, and Lu online SmoothBoost with the online convex programming combiner and importance-weighted weak-learner updates.
- `brier_aggregator`: an exponential-weights aggregator over OGB, Online BBM,
  and OSBoost under Brier loss. It combines their forecasts before observing
  the current label and updates its weights afterward. It runs all three
  ensembles.

The output also includes `bbm_vote`, an appendix diagnostic that interprets
Online BBM's normalized raw vote as a probability. The primary Online BBM
baseline remains the hard prediction specified by the algorithm.

Run a smoke test:

```bash
python3 -m experiments.run --quick
```

Run the default suite:

```bash
python3 -m experiments.run --T 3000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
```

Run an ensemble-size sweep:

```bash
python3 -m experiments.run --T 3000 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --stream-filter group_subset \
  --learner-sweep 1 5 20 100 --out experiments/out/sweep
```

Run the one-at-a-time hyperparameter sensitivity check:

```bash
python3 scripts/run_sensitivity.py
```

Run the real-data suite used in the paper:

```bash
python3 -m experiments.run --suite real --real-dim 128 \
  --seeds 0 --out experiments/out/real_full
```

Run the appendix comparison between the basic and strongly adaptive variants:

```bash
python3 -m experiments.run --suite real --real-dim 128 \
  --seeds 0 --adaptive-defensive \
  --algorithm-filter defensive adaptive_defensive \
  --out experiments/out/adaptive_real
```

Run the controlled-drift appendix benchmark:

```bash
python3 -m experiments.run --suite drift --seeds 0 \
  --adaptive-defensive \
  --algorithm-filter defensive adaptive_defensive \
  --out experiments/out/adaptive_drift
```

Run the bounded-regression appendix benchmark:

```bash
python3 -m experiments.run_regression \
  --ogb-learners 100 --out experiments/out/regression
```

This comparison uses Appliances Energy Prediction, hourly Capital Bikeshare
demand, and Metro Interstate Traffic Volume. Each stream remains in timestamp
order. Fixed target intervals are declared before the run, the first 10% of
observations initialize every method, and losses are measured on the remaining
90%. The loaders add calendar and lagged-target features, standardize numeric
contexts using only earlier observations, and never redistribute the data.
The regression runner compares the Defensive Booster with 100-stage OGB over
the same Euclidean linear weak class; unboosted regression and the mean of
previously observed outcomes are included as controls.

The real-data loaders require `numpy`, `pandas`, `scipy`, and `requests`.
They download Bank Marketing, Electricity, Airlines, and Occupancy Detection,
then evaluate each stream in its released chronological order without
shuffling.  The three Occupancy Detection files are merged by timestamp.
Numeric features are standardized using only means and variances from earlier
contexts; the current value is added to those statistics after it is encoded.
The `drift` suite downloads the five balanced INSECTS optical-sensor streams
through the public URLs maintained by River.  It preserves each released
order and uses the same fixed binary target on every variant: identify
*Aedes albopictus* (either sex) versus *Aedes aegypti* or *Culex
quinquefasciatus*.  No drift markers or future labels are supplied to either
algorithm.

The runner reports loading and per-run progress by default, including elapsed
time. Pass `--no-progress` for quiet batch runs.

Raw and processed datasets are cached under `experiments/data/`, which is
ignored by git.

Outputs are written under `experiments/out/`.  The runner produces:

- `summary.csv`: one row per `(stream, algorithm, seed)`.
- `aggregate.json`: means and standard errors by dataset and algorithm.
- `run_config.json`: the complete command-line configuration.
- `environment.json`: git, Python, platform, and package-version provenance.
- `traces.npz`: per-round scores and losses.
- `plots/*.png`: cumulative error/loss curves with standard-error bands,
  defensive-certificate diagnostics, and compute-sweep plots when requested.
  Runs containing both defensive variants also produce the fixed-window
  `adaptive_real_brier.png` comparison.  The drift suite additionally produces
  `adaptive_insects_brier.png`, whose vertical lines mark the change points
  published with the INSECTS benchmark.
  The hard-label unboosted classifier is omitted from Brier plots for scale,
  but remains in classification/randomized-error plots and all tables.

The synthetic summary also reports the unrestricted offline least-squares
span score and a representation norm that places that score in the comparator
class of the paper's span theorem.  Finite coordinate classes use the
coefficient `l1` norm; Euclidean linear classes use the `l2` norm.  The score
is not clipped before its normalized squared loss is evaluated.

The algorithms encode labels as `{-1,1}` and represent prediction scores in
`[-1,1]`.  Reported loss is Brier loss for the associated probabilities
`p=(score+1)/2`, i.e. one quarter of signed squared loss.

All plotted experiments use one global tuning regime rather than per-dataset
hyperparameter selection.  The Defensive Booster uses the two parameter-free
scalar adaptive-OGD updates specified in `boosting.tex`.  Defensive Booster,
unboosted regression, and OGB share a second-order linear-loss oracle: adaptive
entropy-FTRL for finite classes and projected adaptive gradient ascent for
Euclidean linear classes.  OGB, Online BBM, and OSBoost all use 100 weak
learners in the main runs, or the requested values in a learner-count sweep;
OGB uses the theory-suggested stage step `(log N)/N` for `N > 1` and step `1`
for `N = 1`.  Online BBM and OSBoost both use target classification advantage
`gamma=0.1`, where each paper defines gamma as the improvement over error
`1/2`.  This is stored separately from the real-valued correlation edge used
by the paper's theorem.  The finite classification learners used by unboosted
classification, Online BBM, and OSBoost share the horizon-aware Hedge rate
`min(0.5, sqrt(8 log(n_experts) / T))`; their linear counterparts share the
same weighted projected-perceptron update.  No hyperparameter is selected
separately for an individual dataset.

The synthetic datasets are designed to separate the guarantees:

- `group_subset_heterogeneous`: a latent generator uses a group, a sign equal
  to the signed label, and a magnitude in `{0.6, 1}`. Algorithms receive only
  the resulting vector of 210 weak-rule values, not those latent variables as
  separate features. The real-valued class has edge at least `0.12` under every
  reweighting, but the changing magnitude prevents any one affine span score
  from fitting all rounds exactly. The useful subset rule also changes with
  the reweighting.
- `planted_decoy_margin`: a large finite class contains one sign-perfect heterogeneous-margin weak rule among many decoys, giving a weak-to-strong favorable regime for OSBoost.
- `heterogeneous_margin`: a one-rule sanity check where smooth weak learning holds, but the one-dimensional span still has constant squared loss.
- `linear_span_fallback`: the weak class is the infinite Euclidean linear ball; the smooth weak-learning condition fails on near-margin subsets, but a scaled linear span predictor is accurate.
- `mixed_linear_random_label_mixture`: labels are independently randomized on 35% of rounds. The weak class is the Euclidean unit ball of linear predictors; the randomized rounds give a smooth, low-edge weighting, while span prediction remains useful on the structured component.
- `random_labels`: neither condition helps; this is a negative control.

## Algorithm provenance

- `defensive` implements the Defensive Booster from `boosting.tex`.
- `adaptive_defensive` implements Section 5 using the canonical dyadic interval
  family.  At each scale it runs a fresh copy of the same second-order oracle,
  and it aggregates the active copies using Adapt-ML-Prod (Algorithm 2 of
  Gaillard, Stoltz, and van Erven) with their sleeping-expert confidence
  reduction.  Linear and finite-class states are vectorized across scales;
  this changes runtime but not the per-scale updates.
- `unboosted` runs one online squared-loss learner over the same weak class.
- `unboosted_cls` runs one copy of the same online binary classifier used as a weak learner by Online BBM and OSBoost.
- `ogb` implements Algorithm 1 of Beygelzimer, Hazan, Kale, and Luo, "Online Gradient Boosting", specialized to one-dimensional squared loss with predictions projected to `[-1,1]`.
- `bbm` implements the Online BBM algorithm of Beygelzimer, Kale, and Luo, "Optimal and Adaptive Algorithms for Online Boosting", using the importance-weighted variant.  BBM outputs hard binary predictions; its reported Brier score is the Brier score of the induced `0/1` probability forecast.
- `osboost` implements Algorithm 1 of Chen, Lin, and Lu, "An Online Boosting Algorithm with Theoretical Justifications", using the SmoothBoost-style weights and the OCP simplex combiner.  As in their experiments, weak learners receive importance-weighted updates rather than sampled updates.
- `brier_aggregator` runs OGB, Online BBM, and OSBoost in parallel. Before the
  current label is revealed, it averages their probability forecasts using
  exponential weights computed from earlier Brier losses; after observing the
  label, it updates those weights. With three `N`-learner constituents it
  maintains `3N` weak learners.
- `bbm_vote` is a diagnostic, not a separate trained algorithm: it scores
  Online BBM's normalized raw ensemble vote as a probability. Online BBM's
  specified hard prediction remains the primary baseline.

The baselines are therefore intentionally theory-facing rather than tuned production implementations.

The tracked files in `experiments/reference/` contain the aggregate metrics
reported in the paper.  `scripts/check_results.py` compares a new
`aggregate.json` against one of these references while ignoring machine-specific
runtime fields.
