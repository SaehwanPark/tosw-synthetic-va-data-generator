# Scenario grid

Related: [README](../README.md) · [Methods](methods.md) · [Commands](commands.md) · [Working with the data](working-with-data.md) · [Rebuilding](rebuilding.md)

A **scenario** fixes one age shift, one outcome intercept `alpha`, and one outcome
slope `beta`. The same `alpha` and `beta` apply to hospitalization and mortality.

The full list, with parameters and seeds, is always available from the manifest:

```bash
uv run python -m tosw_synth scenarios
```

## Published grid

39 enabled scenarios.

| Family | Scenario names | Count | Parameters |
| --- | --- | --- | --- |
| Baseline | `synth_t1_ageshift` | 1 | shift 0, alpha 0, beta 1 |
| Age shift | `synth_t2_ageshift0`, `2`, `4`, `6`, `8`, `10` | 6 | shift as named, alpha 0, beta 1 |
| Intercept | `synth_t2_intercept0`, `-5`, `-10`, …, `-100` | 21 | shift 0, alpha = value/100, beta 1 |
| Slope | `synth_t2_slope100`, `98`, `96`, …, `80` | 11 | shift 0, alpha 0, beta = value/100 |

`synth_t2_ageshift0` shares parameters with the baseline but uses a different
seed, so it is a separate draw rather than a duplicate.

Interpretation:

- **`age_shift`** adds years to the mean age before truncation. Case mix changes;
  the score-outcome relationship does not.
- **`alpha`** shifts the outcome log-odds uniformly. Negative values lower outcome
  rates at every level of predicted risk.
- **`beta`** multiplies the centred log-odds. Below 1 compresses risk spread;
  above 1 exaggerates it. Must stay positive.

## Seeds

Base seeds follow the historical insertion order with base seed 343:

| Family | Seed formula |
| --- | --- |
| `synth_t1_ageshift` | 343 |
| `synth_t2_ageshift{shift}` | 345 + shift/2 |
| `synth_t2_slope{value}` | 357 + (120 − value)/2 |
| `synth_t2_intercept{value}` | 388 + (−value)/5 |

A replicate hashes `(base_seed, replicate_index)`, so replicate 0 is the
published draw and every higher index is an independent stream. No two scenarios
share a stream.

## Disabled reference records

The manifest also retains historical scenario families from earlier development,
marked `enabled: false`, so the record of what was explored stays intact. These
are not part of the published grid and are not produced by `--enabled` or `suite`:

- priority-group scenarios (`synth_t1_prior1245`, `synth_t2_priorityshift*`)
- slope values outside 80–100
- disease-subset families recorded at 500,000 rows with no single base seed

Run `uv run python -m tosw_synth scenarios --all` to see them.
