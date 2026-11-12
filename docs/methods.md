# Methods: how the synthetic data are generated

Related: [README](../README.md) · [Commands](commands.md) · [Scenarios](scenarios.md) · [Working with the data](working-with-data.md) · [Rebuilding](rebuilding.md)

This page explains the generation mechanism in more detail than the
[README](../README.md). It is written for users who want to know what the
generated numbers mean before using them.

## What this repository ships

The repository contains a **fitted bundle** of about 27 MB under
`bundles/can2014/`. It holds three kinds of object and nothing else:

| Stored object | Count | What it is |
| --- | --- | --- |
| Distribution parameters for age | 1 | Mean, standard deviation, minimum, and maximum of the 2014 cohort age distribution |
| Trained conditional predictors | 32 | One gradient-boosted classifier per clinical or utilization covariate |
| Fixed configurations | 7 | The age cut points, the two CAN coefficient sets, the probability transforms, and the outcome draw settings |

These are **parameters and model weights**, not rows. There is no table of
patients in the bundle, and no individual record can be read out of it directly.

## The three generation stages

### 1. Age

Age is drawn from a truncated normal distribution fitted to the cohort:

```text
age ~ Normal(mean + age_shift, sd),  bounded to [age_min, age_max]
```

The bounds are the observed cohort range (21 to 99 years). The draw is rounded to
whole years. A scenario's `age_shift` moves the mean before truncation, which is
how the age-drift scenarios simulate an older population.

The age group `agegrp` is then derived deterministically from `age_cont` using
fixed cut points at 54, 64, 74, and 84 years. It is not modelled; it is a
deterministic re-coding.

### 2. Covariates

The 32 remaining covariates are drawn **sequentially**. For the `j`-th
covariate, a gradient-boosted classifier predicts its conditional distribution
from `agegrp` and the covariates already drawn:

```text
X_j ~ P(X_j | agegrp, X_1, ..., X_{j-1})
```

A two-level covariate is drawn as a Bernoulli outcome. A multi-level covariate is
drawn from the predicted categorical distribution. The feature order and the
fitted models are fixed once and reused for every scenario.

This is what distinguishes the generator from independent resampling: the
covariates keep their joint structure rather than only their marginal
frequencies. Measured against the historical cohort data, the pairwise
correlation structure across all 561 covariate pairs reproduces at a
correlation-of-correlations of about 0.998.

### 3. Outcomes

A fixed CAN linear predictor combines the drawn covariates into log-odds:

```text
eta_hosp(X), eta_mort(X)
```

This uses the published CAN hospitalization and mortality coefficients. The
predictions are combined into log-odds and then transformed with a piecewise
adjustment controlled by the scenario:

```text
logit P(Y = 1 | X) = alpha + beta * (eta(X) - mu) + mu     if eta(X) >= mu
                     alpha + eta(X)                          if eta(X) < mu
```

where `mu` is the logit of the mean predicted risk. At the baseline,
`alpha = 0` and `beta = 1`, so the probability reduces to the ordinary logistic
transform of `eta(X)`.

Finally a single Bernoulli draw per row produces `Hosp_1y` and `Mort_1y`. Both
outcomes come from the **same synthetic row**, so a synthetic patient who is
hospitalized also carries that patient's mortality risk.

## What the scenario parameters mean

| Parameter | Effect | Interpretation |
| --- | --- | --- |
| `age_shift` | Added to the mean age before truncation | Simulates an older or younger population. Case mix changes; the score-outcome relationship does not. |
| `alpha` | Added to the outcome log-odds | Shifts calibration uniformly. Negative values make outcomes less likely at every level of predicted risk. |
| `beta` | Multiplies the centred log-odds | Compresses or expands risk spread. Below 1 makes patients look more alike in risk; above 1 exaggerates differences. |

`beta` must stay strictly positive. The published grid satisfies this.

## Reproducibility

Each scenario has a base seed in `configs/scenarios.yaml`. A replicate's stream is
derived by hashing `(scenario_seed, replicate_index)`, so:

- replicate `0` reproduces the published draw exactly
- every higher replicate index is independent
- no two scenarios share a stream at any index

The same scenario, row count, replicate index, and pinned environment reproduce
the same frame. Output is not promised to be byte-identical across different
`tabdat-synth` or Polars versions.

## A note on privacy

The bundle stores fitted parameters rather than patient rows, and this repository
distributes no VA extract, no patient identifiers, and no protected health
information. Nothing here is subject to HIPAA or VA DUA restrictions on
redistribution.

That said, **fitted parameters are not a formal privacy guarantee.** Gradient-
boosted trees and summary statistics can retain information about the cohort they
were fitted on, especially for rare categories. The generator implements no
differential privacy. Treat it as a tool for methods research, teaching, and
workflow prototyping, not as a mechanism for releasing sensitive data.

Do not present generated data as a de-identified copy of real patients, and do not
use it to make claims about real individuals.
