# Working with the generated files

Related: [README](../README.md) · [Methods](methods.md) · [Commands](commands.md) · [Scenarios](scenarios.md) · [Rebuilding](rebuilding.md)

Practical notes on opening, checking, and pairing the generated tables.

## Reading the files

Output is [Parquet](https://parquet.apache.org/), a compact columnar format. It is
smaller and faster than CSV and keeps column types intact.

**Python**

```python
import polars as pl

df = pl.read_parquet("data/drift/synth_t1_ageshift.parquet")
print(df.shape)
print(df.select(["age_cont", "Hosp_1y", "Mort_1y"]).describe())
```

`pandas` works too: `pd.read_parquet(...)`.

**R**

```r
library(arrow)
df <- read_parquet("data/drift/synth_t1_ageshift.parquet")
summary(df$age_cont)
```

**Stata 18 or newer**

```stata
import parquet using "data/drift/synth_t1_ageshift.parquet", clear
```

Stata before 18 cannot read Parquet; convert first.

**Anything else**

```python
import polars as pl

pl.read_parquet("data/drift/synth_t1_ageshift.parquet").write_csv(
  "data/drift/synth_t1_ageshift.csv"
)
```

```bash
duckdb -c "COPY (SELECT * FROM 'data/drift/synth_t1_ageshift.parquet') TO 'out.csv' (HEADER)"
```

## Columns

Each table has one row per synthetic patient and 38 columns.

| Column | Type | Meaning |
| --- | --- | --- |
| `age_cont` | integer | Age in completed years |
| `agegrp` | categorical | Age group `1`–`5`, cut at 54, 64, 74, 84 |
| 32 covariates | categorical | CAN clinical and utilization factors, coded as integers |
| `pHosp_1y` | float | One-year hospitalization risk, in (0, 1) |
| `pMort_1y` | float | One-year mortality risk, in (0, 1) |
| `Hosp_1y` | integer | One-year hospitalization, 0 or 1 |
| `Mort_1y` | integer | One-year mortality, 0 or 1 |

Covariates use the original coding, where `99` means "not recorded". Two internal
log-odds columns are used during generation and dropped before writing, so they
never appear in output.

## Pairing T1 with T2

The repeated-sampling analysis compares a baseline sample (T1) against a drifted
sample (T2) within a replicate. Read `pair_index.csv` to get the linkage:

```python
import polars as pl

root = "data/pairs"
index = pl.read_csv(f"{root}/pair_index.csv")

row = index.filter(
  (pl.col("t2_scenario") == "synth_t2_intercept-50") & (pl.col("replicate_index") == 0)
).row(0, named=True)

t1 = pl.read_parquet(f"{root}/{row['t1_path']}")
t2 = pl.read_parquet(f"{root}/{row['t2_path']}")
```

The index has one row per scenario-replicate and these columns:

| Column | Meaning |
| --- | --- |
| `replicate_index` | Replicate number |
| `label` | Directory name, e.g. `replicate_r0000` |
| `t1_scenario`, `t2_scenario` | Scenario names for the two time points |
| `t1_path`, `t2_path` | Paths relative to the pairs root |
| `t1_seed`, `t2_seed` | Resolved seeds for the two draws |
| `n_samples` | Rows per time point |

Every drift scenario in a replicate shares the same T1 file, and T1 differs
across replicates. That is what makes within-replicate contrasts comparable while
keeping replicates independent.

## Quick sanity checks

Confirm a file looks right:

```python
import polars as pl

df = pl.read_parquet("data/drift/synth_t1_ageshift.parquet")

assert df.height == 250_000
assert df.null_count().sum_horizontal().item() == 0
assert set(df["agegrp"].cast(pl.String).unique().to_list()) <= {"1", "2", "3", "4", "5"}
assert set(df["Hosp_1y"].unique().to_list()) <= {0, 1}
assert df["pHosp_1y"].is_between(0, 1).all()
```

Expected rough magnitudes at the baseline: mean age near 62.7, hospitalization
rate near 0.085, mortality rate near 0.031. Small deviations are normal sampling
variation.

## Reproducing a file

The same scenario, row count, replicate index, and pinned environment reproduce
the same frame:

```bash
uv run python -m tosw_synth generate --scenario synth_t1_ageshift --n-samples 250000
```

`uv.lock` pins the library versions. Byte-identical output is not promised across
different `tabdat-synth` or Polars versions, so record the pinned versions in any
study that depends on exact reproduction.
