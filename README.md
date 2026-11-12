# Semi-Synthetic VA CAN Dataset Generator

Generate semi-synthetic **Veterans Affairs Care Assessment Needs (CAN)** datasets
from a fitted bundle. No access to the original VA data is required.

**This repository distributes no VA data.** It contains no source extract, no
patient records, no patient identifiers, and no protected health information.
Nothing here falls under HIPAA or VA data use agreement restrictions on
redistribution. What is distributed instead is described under
[what is distributed](#what-is-distributed).

## Quick start

You need [uv](https://docs.astral.sh/uv/). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and set up:

```bash
git clone https://github.com/SaehwanPark/tosw-synthetic-va-data-generator.git
cd tosw-synthetic-va-data-generator
uv sync
```

Confirm the installation works with a small trial run, which takes under a minute:

```bash
uv run python -m tosw_synth suite --replicates 2 --n-samples 500 --output-dir data_trial
```

Then run the full grid:

```bash
uv run python -m tosw_synth suite --output-dir data/
```

That produces every dataset a simulation study needs. It is a large job — 39,079
files, roughly 172 GiB, several hours on a laptop — so run the trial first and make
sure you have the disk space. `data/` is git-ignored, so generated files stay
local.

## What you generate

`suite` writes three groups of Parquet tables, all with the same 38 columns:

| Output | Contents |
| --- | --- |
| `data/drift/*.parquet` | 39 scenario tables, 250,000 rows each |
| `data/reduced/*_n100000.parquet` | 39 tables, 100,000 rows drawn without replacement |
| `data/pairs/` | 38 drift scenarios × 1,000 replicate T1–T2 pairs, plus `pair_index.csv` |

Each row is one synthetic patient. A few rows from the baseline table:

```text
age_cont agegrp BLACK Albumin Prior1245 sex pHosp_1y pMort_1y Hosp_1y Mort_1y
      40      1     0       0         0  99   0.0612   0.0019       0       0
      69      3     0       0         1  99   0.1225   0.0137       0       0
      69      3     0       0         2  99   0.0255   0.0112       0       0
```

Columns: `age_cont`, `agegrp`, the 32 CAN clinical and utilization covariates
(`Albumin`, `BUN`, `HCC_HXCHF`, `Prior1245`, and others, all coded as integers),
then `pHosp_1y`, `pMort_1y`, `Hosp_1y`, and `Mort_1y`.

Covariates keep the original coding, where `99` means "not recorded". That code is
common: in the example rows above, `sex` is `99` and about a third of `Albumin`
values are `99`. Treat `99` as missing rather than as a real level when you
analyze the data.

Open them with Python, R, Stata 18+, or DuckDB — see
[working with the data](docs/working-with-data.md).

## What is distributed

The repository ships a 27 MB fitted bundle under `bundles/can2014/`. It contains
exactly three things:

- **Distribution parameters for age** — the mean, standard deviation, minimum, and
  maximum of the 2014 cohort age distribution.
- **32 trained conditional predictors** — one gradient-boosted classifier per
  covariate, each predicting that covariate from `agegrp` and the covariates
  drawn before it.
- **Fixed outcome configurations** — the CAN hospitalization and mortality
  coefficient sets, the probability transform, and the Bernoulli draw settings.

These are parameters, coefficient values, and model weights. There is no patient
table in the bundle, and no individual record can be read out of it.

## How generation works

Three stages, in order:

1. **Age.** Draw from a truncated normal fitted to the cohort, rounded to whole
   years, then cut into age group `1`–`5` at 54, 64, 74, and 84 years.
2. **Covariates.** Draw each of the 32 covariates in sequence, conditional on
   `agegrp` and everything drawn before it. This preserves the joint structure
   rather than only the marginal frequencies.
3. **Outcomes.** Combine covariates through the fixed CAN linear predictors, apply
   the scenario's `alpha` and `beta`, transform to probabilities, and draw one
   Bernoulli outcome per row for hospitalization and mortality.

Full detail, including the exact equations and what each scenario parameter means,
is in [methods](docs/methods.md).

## Scenarios

A scenario sets an age shift, an outcome intercept `alpha`, and an outcome slope
`beta`. The published grid has 39 enabled scenarios: 1 baseline, 6 age-shift,
11 slope, and 21 intercept scenarios. `uv run python -m tosw_synth scenarios`
lists them with their seeds.

See the [scenario grid](docs/scenarios.md) for the full table and seed formulas.

## Documentation

| Page | What it covers |
| --- | --- |
| [Methods](docs/methods.md) | Generation mechanism, equations, what the bundle contains |
| [Commands](docs/commands.md) | Every command and option, including partial runs |
| [Scenarios](docs/scenarios.md) | The scenario grid, parameters, and seeds |
| [Working with the data](docs/working-with-data.md) | Opening files, pairing T1/T2, sanity checks |
| [Rebuilding](docs/rebuilding.md) | Refitting the bundle with authorized VA data access |

## Requirements and cost

Mac, Linux, or Windows, plus `uv`. You do not need to install Python; `uv` fetches
the pinned version.

One 250,000-row table takes about a minute and 2.5 GB of memory and produces a
~4.5 MiB file.

## Privacy

The bundle stores fitted parameters rather than patient rows, and no VA data is
redistributed here. Fitted parameters are **not** a formal privacy guarantee,
however: gradient-boosted trees and summary statistics can retain information
about the cohort they were fit on. The generator implements no differential
privacy.

Use it for methods research, teaching, and workflow prototyping. Do not present
generated data as a de-identified copy of real patients, and do not use it to make
claims about real individuals. See [methods](docs/methods.md#a-note-on-privacy).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `uv: command not found` | Install `uv`, then open a new terminal |
| `unknown scenario` | Run `uv run python -m tosw_synth scenarios` for valid names |
| `fitted bundle package version mismatch` | Run `uv sync` to install the pinned version |
| `cannot draw N rows without replacement` | The subsample exceeds the input; lower `--n-samples` |
| Slow or out of memory | Lower `--n-samples`, or generate one scenario at a time |

## Citation

Built on **`tabdat-synth`** 0.2.0, the spec-driven synthesis engine that fits and
applies the bundle.

If you use this generator, please cite it and `tabdat-synth`:

```bibtex
@software{tosw_synthetic_va_data_generator,
  author  = {Park, Sae-Hwan},
  title   = {tosw-synthetic-va-data-generator: semi-synthetic VA Care
             Assessment Needs dataset generator},
  version = {0.1.0},
  url     = {https://github.com/SaehwanPark/tosw-synthetic-va-data-generator}
}

@software{tabdat_synth,
  author  = {Park, Sae-Hwan},
  title   = {tabdat-synth: spec-driven synthetic tabular data generation
             with reusable fitted bundles},
  version = {0.2.0},
  url     = {https://github.com/SaehwanPark/tabdat-synth},
  note    = {PyPI: https://pypi.org/project/tabdat-synth/0.2.0/}
}
```

The generator also depends on the CAN risk model of Wang et al. (2013, *Medical
Care* 51:368–373, doi:10.1097/MLR.0b013e31827da95a) and on LightGBM
(Ke et al., NeurIPS 2017, pp. 3146–3154). Cite both if your work relies on them.

## License

Apache-2.0. See [LICENSE](LICENSE).
