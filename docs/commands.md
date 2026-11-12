# Command reference

Related: [README](../README.md) · [Methods](methods.md) · [Scenarios](scenarios.md) · [Working with the data](working-with-data.md) · [Rebuilding](rebuilding.md)

Full option reference for the four commands. For the quick path, see the
[README](../README.md).

All commands are run from the repository root after `uv sync`:

```bash
uv run python -m tosw_synth <command> [options]
```

Two global options apply to every command:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bundle` | `bundles/can2014` | Fitted bundle directory to generate from |
| `--manifest` | `configs/scenarios.yaml` | Scenario manifest to read |

---

## `generate`

Write scenario tables, one file per scenario.

| Option | Default | Meaning |
| --- | --- | --- |
| `--scenario NAME` | — | Generate this one scenario. Mutually exclusive with `--enabled`. |
| `--enabled` | off | Generate every enabled scenario in the manifest. |
| `--n-samples N` | manifest value (250,000) | Rows for this run. Overrides the manifest. |
| `--replicate-index N` | `0` | Which repeat draw to produce. Index `0` is the published draw. |
| `--output PATH` | derived | Output parquet path. Only valid with `--scenario`. |
| `--output-dir DIR` | `data/` | Output directory for `--enabled`, or the base directory when `--output` is omitted. |

Examples:

```bash
# One scenario at the published size
uv run python -m tosw_synth generate --scenario synth_t2_ageshift4

# A fast trial
uv run python -m tosw_synth generate --scenario synth_t1_ageshift --n-samples 1000

# The 17th repeat draw of the baseline
uv run python -m tosw_synth generate \
  --scenario synth_t1_ageshift \
  --replicate-index 17 \
  --output data/synth_t1_ageshift_r17.parquet

# The whole enabled grid
uv run python -m tosw_synth generate --enabled --output-dir data/
```

When `--output` is omitted with `--scenario`, the file is named
`<scenario>.parquet` for replicate `0` and `<scenario>_rXXXX.parquet` otherwise.

Cost: about one minute and 2.5 GB of memory per 250,000-row table. Each file is
about 4.5 MiB compressed.

---

## `suite`

Generate every dataset a monitoring study needs: the drift grid, the
reduced-sample copies, and the repeated pair grid.

| Option | Default | Meaning |
| --- | --- | --- |
| `--output-dir DIR` | `data/` | Root output directory. |
| `--replicates N` | `1000` | Replicate pairs per drift scenario. |
| `--n-samples N` | manifest value | Rows per table. Use a small value for a trial. |
| `--skip-pairs` | off | Skip the repeated pair grid. |
| `--skip-reduced` | off | Skip the 100,000-row reduced copies. |
| `--dry-run` | off | Print the plan without generating. |

Output layout:

```text
data/
  drift/<scenario>.parquet                       39 tables, 250,000 rows
  reduced/<scenario>_n100000.parquet             39 tables, 100,000 rows
  pairs/pair_index.csv                           the T1-to-T2 linkage
  pairs/synth_t1_ageshift/replicate_rXXXX/t1.parquet
  pairs/<drift_scenario>/replicate_rXXXX/t2.parquet
```

Cost: a full 1,000-replicate run writes 39,079 files and needs roughly
172 GiB of free disk, taking several hours on a laptop. Start small.

```bash
# Inspect the plan
uv run python -m tosw_synth suite --dry-run

# Cheap trial
uv run python -m tosw_synth suite --replicates 5 --n-samples 1000 --output-dir data_trial

# Full run
uv run python -m tosw_synth suite --output-dir data/
```

---

## `pairs`

Generate only the repeated T1–T2 pairs, optionally for selected drift scenarios.

| Option | Default | Meaning |
| --- | --- | --- |
| `--drift NAME` | all enabled non-baseline scenarios | Drift scenario to pair against T1. Repeatable. |
| `--replicates N` | `1000` | Number of replicate pairs per scenario. |
| `--replicate-start N` | `0` | First replicate index to generate. |
| `--n-samples N` | manifest value | Rows per time point. |
| `--output-dir DIR` | `data/pairs` | Root directory for pair output. |
| `--skip-existing` | off | Skip draws whose file already exists. |

```bash
uv run python -m tosw_synth pairs \
  --drift synth_t2_intercept-50 \
  --drift synth_t2_slope82 \
  --replicates 1000

# Resume an interrupted run
uv run python -m tosw_synth pairs --replicates 1000 --skip-existing
```

A baseline draw depends only on the replicate index, so `t1.parquet` is written
once per replicate and shared by every drift scenario in that replicate.
`pair_index.csv` records which T1 file pairs with which T2 file. Read it rather
than inferring the linkage from directory names.

---

## `subsample`

Draw rows without replacement from an already generated file. No model is refit,
so this is fast and deterministic.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input PATH` | required | Input parquet. |
| `--n-samples N` | required | Rows to draw. Must not exceed the input row count. |
| `--seed N` | `343` | Draw seed. Change it for a different subsample of the same file. |
| `--output PATH` | required | Output parquet. |

```bash
uv run python -m tosw_synth subsample \
  --input data/drift/synth_t2_ageshift4.parquet \
  --n-samples 100000 \
  --seed 343 \
  --output data/reduced/synth_t2_ageshift4_n100000.parquet
```

Every column is carried through unchanged.

---

## `scenarios`

List manifest scenarios, their parameters, and their seeds.

| Option | Default | Meaning |
| --- | --- | --- |
| `--all` | off | Include disabled reference records as well as enabled ones. |

```bash
uv run python -m tosw_synth scenarios
```

---

## Editing the manifest

`configs/scenarios.yaml` is plain YAML. Each record has:

| Field | Meaning |
| --- | --- |
| `name` | Scenario name used on the command line |
| `enabled` | Whether `--enabled` and `suite` include it |
| `family` | Free-text grouping (`baseline`, `age_shift`, `slope`, `intercept`) |
| `age_shift` | Years added to the mean age |
| `alpha` | Outcome log-odds intercept adjustment |
| `beta` | Outcome log-odds slope multiplier; must stay positive |
| `n_samples` | Default row count |
| `seed` | Base seed; a replicate hashes this with the replicate index |

To add a scenario, append a record. To change the default grid, edit the
`enabled` flags. Run `--dry-run` before a long run to confirm the plan.
