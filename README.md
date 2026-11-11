# Synthetic VA CAN Dataset Generator

This repository generates semi-synthetic **Veterans Affairs Care Assessment
Needs (CAN) cohort** data. You can produce a complete analysis-ready dataset
with the same structure as the original 2014 CAN extract, at whatever sample
size you need, on your own machine, without access to the original data.

The generator ships with a **fitted bundle**: a small set of learned
parameters captured once from a 2014 VA CAN cohort. Generating data only reads
that bundle. It never reads the original extract, so you do not need a data use
agreement, a private path, or a network connection to produce data.

---

## What you get

Each generated table contains one row per synthetic patient and the following
columns:

| Column | Meaning |
| --- | --- |
| `age_cont` | Age in completed years, as an integer |
| `agegrp` | Age group: `1` through `5`, cut at 54, 64, 74, and 84 years |
| 32 clinical and utilization covariates | The CAN risk-factor columns, e.g. `Albumin`, `BUN`, `Prior1245`, `deyo`, `HCC_HXCHF` |
| `pHosp_1y` | One-year hospitalization risk, a probability between 0 and 1 |
| `pMort_1y` | One-year mortality risk, a probability between 0 and 1 |
| `Hosp_1y` | One-year hospitalization, `0` or `1` |
| `Mort_1y` | One-year mortality, `0` or `1` |

Values are coded exactly as in the original extract: categorical covariates use
the same integer levels, with `99` meaning "not recorded". Two internal
log-odds columns used while generating the outcomes are dropped before the file
is written.

## How the data are produced

Generation happens in three stages.

1. **Age.** An age is drawn from a truncated normal distribution fitted to the
   cohort, clipped to the observed age range and rounded to whole years. Scenarios
   can shift the mean age. `agegrp` is then derived from `age_cont` with the
   cohort cut points.
2. **Covariates.** The remaining covariates are drawn one at a time. Each has a
   trained gradient-boosting model that predicts it from `agegrp` and the
   covariates drawn before it. This keeps the joint distribution realistic
   instead of making every column independent.
3. **Outcomes.** A fixed risk model combines the drawn covariates into
   hospitalization and mortality log-odds. A scenario can rescale those
   log-odds (`alpha`, `beta`) to simulate a sicker or healthier population. Each
   probability column is the logistic transform of its log-odds, and each binary
   outcome is drawn once from that probability. Hospitalization and mortality
   therefore come from the same synthetic patient, as they would in a real cohort.

## What you need

- A Mac, Linux, or Windows machine.
- [uv](https://docs.astral.sh/uv/), the Python package manager. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

  On Windows, use the PowerShell installer from the same page instead. You do
  **not** need to install Python yourself; `uv` fetches the right version.

## Step 1: get the repository and set up

```bash
git clone https://github.com/SaehwanPark/tosw-synthetic-va-data-generator.git
cd tosw-synthetic-va-data-generator
uv sync
```

`uv sync` creates a local environment and installs everything the generator
needs, including a pinned release of the `tabdat-synth` library.

## Step 2: generate a dataset

Generate one scenario at the published size of 250,000 rows:

```bash
uv run python -m tosw_synth generate \
  --scenario synth_t1_ageshift \
  --n-samples 250000 \
  --replicate-index 0 \
  --output data/synth_t1_ageshift.parquet
```

This writes `data/synth_t1_ageshift.parquet`. That directory is ignored by git,
so your generated data stays local.

A real run prints a short summary:

```text
wrote data/synth_t1_ageshift.parquet (250,000 rows x 38 columns)
  scenario: synth_t1_ageshift
  age shift: +0.0 years
  outcome alpha: +0.00, beta: 1.00
  seed: 343
  mean age: 62.71
  hospitalization rate: 0.0864
  mortality rate: 0.0308
```

## Step 3: open the data

The file is [Parquet](https://parquet.apache.org/), a compact columnar format
that Python, R, Stata 18+, and DuckDB all read directly. It is faster and
smaller than CSV and keeps column types intact.

**Python**

```python
import polars as pl

df = pl.read_parquet("data/synth_t1_ageshift.parquet")
print(df.shape)
print(df.select(["age_cont", "Hosp_1y", "Mort_1y"]).describe())
```

**R**

```r
library(arrow)
df <- read_parquet("data/synth_t1_ageshift.parquet")
summary(df$age_cont)
```

**Stata** (18 or newer reads Parquet natively)

```stata
import parquet using "data/synth_t1_ageshift.parquet", clear
```

Stata versions before 18 cannot read Parquet. Convert to CSV or `.dta` first, as
shown below.

If your tool cannot read Parquet, convert once with Python or DuckDB:

```python
import polars as pl

pl.read_parquet("data/synth_t1_ageshift.parquet").write_csv(
  "data/synth_t1_ageshift.csv"
)
```

```bash
duckdb -c "COPY (SELECT * FROM 'data/synth_t1_ageshift.parquet') TO 'data/synth_t1_ageshift.csv' (HEADER)"
```

## Choosing a scenario

A **scenario** sets one age shift, one outcome intercept `alpha`, and one
outcome slope `beta`. The same `alpha` and `beta` apply to hospitalization and
mortality. The baseline scenario has an age shift of `0`, `alpha = 0`, and
`beta = 1`, which is the cohort as fitted.

- **`age_shift`** adds years to the mean age before the age distribution is
  truncated. `synth_t2_ageshift4` shifts the mean age up by 4 years.
- **`alpha`** moves both outcome log-odds up or down by a fixed amount. Negative
  values lower the outcome rates throughout the population.
  `synth_t2_intercept-50` uses `alpha = -0.5`.
- **`beta`** multiplies the spread of the outcome log-odds. Values below `1`
  flatten risk differences between patients; `synth_t2_slope80` uses
  `beta = 0.8`. `beta` must stay strictly positive.

List every scenario in the manifest, including its parameters and seed:

```bash
uv run python -m tosw_synth scenarios
```

Add `--all` to include the retained but disabled historical records:

```bash
uv run python -m tosw_synth scenarios --all
```

### The published scenario grid

| Family | Scenario names | Count |
| --- | --- | --- |
| Baseline | `synth_t1_ageshift` | 1 |
| Age shift | `synth_t2_ageshift0`, `2`, `4`, `6`, `8`, `10` | 6 |
| Intercept | `synth_t2_intercept0`, `-5`, `-10`, …, `-100` | 21 |
| Slope | `synth_t2_slope100`, `98`, `96`, …, `80` | 11 |

`synth_t2_ageshift0` is a separate seeded draw from the baseline, so the two
share parameters but not random draws.

### Generating the whole grid

```bash
uv run python -m tosw_synth generate --enabled --output-dir data/
```

This writes one parquet file per enabled scenario, named after the scenario.
Expect roughly 40 minutes on a laptop for all 39 files at 250,000 rows each,
or about a minute per 250,000-row file.

## Reproducibility: `--n-samples` and `--replicate-index`

**`--n-samples`** sets the row count for that run. Omit it to use the manifest
value, which is 250,000. Any positive integer works, so a quick test is cheap:

```bash
uv run python -m tosw_synth generate \
  --scenario synth_t2_ageshift4 \
  --n-samples 1000 \
  --output data/quick_test.parquet
```

**`--replicate-index`** selects which repeat draw to produce. A scenario's seed
is `scenario_seed + replicate_index`, so index `0` is the published draw and
index `1` is an independent draw with the same parameters. The default is `0`.
This is how you build repeated samples for simulation studies without changing
the scenario definition:

```bash
uv run python -m tosw_synth generate \
  --scenario synth_t1_ageshift \
  --n-samples 250000 \
  --replicate-index 17 \
  --output data/synth_t1_ageshift_r17.parquet
```

The same scenario, row count, and replicate index always reproduce the same
frame on the same pinned environment. Pin `uv.lock` and the `tabdat-synth`
release if you need that guarantee across machines. Byte-identical output is
not promised across different library versions.

## Drawing a smaller analysis sample

Subsampling reads a generated file and draws rows **without replacement**. It
does not refit any model, so it is fast and deterministic:

```bash
uv run python -m tosw_synth subsample \
  --input data/synth_t2_ageshift4.parquet \
  --n-samples 100000 \
  --seed 343 \
  --output data/synth_t2_ageshift4_n100000.parquet
```

Change `--seed` to draw a different subsample from the same generated file.
Every column is carried through unchanged.

## Privacy and intended use

The bundle stores **fitted parameters from a 2014 VA CAN cohort**, not patient
rows. Categorical models are gradient-boosted trees over aggregate splits, and
the numeric steps store means, standard deviations, and bounds. No patient
identifier, no source row, and no original extract is included in this
repository.

Those fitted parameters are **not a formal privacy guarantee.** Gradient-boosted
models and empirical summaries can leak information about the cohort they were
fit on, especially for rare categories, and this generator implements no
differential privacy. Treat it as a tool for methods research, teaching, and
workflow prototyping rather than as a public-release mechanism for sensitive
data. Do not present generated data as a de-identified copy of real patients,
and do not use it to make claims about real individuals.

## Author-only: rebuilding the bundle

Refitting requires the original 2014 VA CAN extract, which is not distributed.
If you have authorized access to it, point `VA_DIR` at its directory:

```bash
export VA_DIR=/path/to/can/extract
uv run --extra fit python -c "from tosw_synth.fit import rebuild; print(rebuild())"
```

A rebuild reads `all_can_2014_v2_ordinalver.parquet` from that directory, draws
the historical 50 percent fitting sample with seed `343`, fits 32 covariate
models plus the age summary, and rewrites `bundles/can2014/`. It takes about 10
minutes and roughly 4 GB of memory. Rebuilding is only needed if the cohort or
the model specification changes; normal use never runs it.
## Troubleshooting

**`uv: command not found`.** Install `uv` with the command in
[What you need](#what-you-need), then open a new terminal so your shell picks up
the new path.

**`unknown scenario`.** The name must match the manifest exactly. Run
`uv run python -m tosw_synth scenarios --all` to see every valid name.

**`fitted bundle package version mismatch`.** The bundle was written by a
different `tabdat-synth` release. Run `uv sync` to install the pinned version.

**`cannot draw N rows without replacement`.** The subsample is larger than the
input file. Generate a bigger base file first, or lower `--n-samples`.

**Generation is slow or runs out of memory.** A 250,000-row scenario needs about
2.5 GB of memory and roughly a minute. Lower `--n-samples`, or generate one
scenario at a time instead of using `--enabled`.

## Citation

If you use this generator, cite the repository and the `tabdat-synth` release it
pins. See `pyproject.toml` for the exact version.

## License

Apache-2.0. See [LICENSE](LICENSE).
