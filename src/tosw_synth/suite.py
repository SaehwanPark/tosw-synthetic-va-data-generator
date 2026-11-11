"""One-command generation of every dataset the paper's simulations need.

The paper uses two kinds of dataset, and this module produces both:

``drift``
  One 250,000-row table per drift scenario. These feed the calibration curves,
  threshold selection, and the minimum equivalence-breaking drift analysis.

``pairs``
  Repeated T1-T2 sample pairs. Replicate ``r`` is the baseline draw at ``r``
  together with one drift scenario at ``r``, written as
  ``pairs/<scenario>/replicate_r0000/{t1,t2}.parquet``. These feed the
  loss-of-equivalence-support rates.

``reduced``
  A 100,000-row draw without replacement from each drift table, for the
  reduced-sample analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tabdat_synth import FittedSynthesizer

from tosw_synth.generate import (
  BASELINE_SCENARIO,
  apply_scenario,
  build_pair_plan,
  generate_pair,
  subsample,
  write_pair,
  write_parquet,
)
from tosw_synth.scenario import ScenarioManifest

REDUCED_N_SAMPLES = 100_000
REDUCED_SEED = 343
DRIFT_DIRNAME = "drift"
PAIRS_DIRNAME = "pairs"


@dataclass(frozen=True)
class SuiteOptions:
  """Configuration for a full suite run."""

  output_dir: Path
  replicates: int = 1000
  include_reduced: bool = True
  include_pairs: bool = True
  n_samples: int | None = None


def drift_output_path(output_dir: Path, scenario_name: str) -> Path:
  return output_dir / DRIFT_DIRNAME / f"{scenario_name}.parquet"


def reduced_output_path(output_dir: Path, scenario_name: str) -> Path:
  return output_dir / "reduced" / f"{scenario_name}_n{REDUCED_N_SAMPLES}.parquet"


def build_suite_steps(
  manifest: ScenarioManifest,
  options: SuiteOptions,
) -> list[tuple[str, str]]:
  """Describe the planned work without touching the bundle."""
  drift = [scenario for scenario in manifest.enabled()]
  drift_path = options.output_dir / DRIFT_DIRNAME
  reduced_path = options.output_dir / "reduced"
  pairs_path = options.output_dir / PAIRS_DIRNAME

  steps: list[tuple[str, str]] = []

  drift_description = (
    f"{len(drift)} scenario table(s) at the manifest row count under {drift_path}"
  )
  steps.append(("drift", drift_description))

  if options.include_reduced:
    reduced_description = (
      f"{len(drift)} table(s) of {REDUCED_N_SAMPLES:,} rows drawn without "
      f"replacement under {reduced_path}"
    )
    steps.append(("reduced", reduced_description))

  if options.include_pairs:
    drift_pairs = [scenario for scenario in drift if scenario.name != BASELINE_SCENARIO]
    pairs_description = (
      f"{len(drift_pairs)} drift scenario(s) x {options.replicates} replicate "
      f"pair(s) under {pairs_path}"
    )
    steps.append(("pairs", pairs_description))

  return steps


def run_suite(
  fitted: FittedSynthesizer,
  manifest: ScenarioManifest,
  options: SuiteOptions,
  steps: list[tuple[str, str]] | None = None,
) -> None:
  """Generate every dataset in the suite, in dependency order."""
  for label, description in steps or build_suite_steps(manifest, options):
    print(f"[{label}] {description}", flush=True)
    if label == "drift":
      _run_drift(fitted, manifest, options)
    elif label == "reduced":
      _run_reduced(fitted, manifest, options)
    elif label == "pairs":
      _run_pairs(fitted, manifest, options)
  print(f"suite complete: {options.output_dir}", flush=True)


def _run_drift(
  fitted: FittedSynthesizer,
  manifest: ScenarioManifest,
  options: SuiteOptions,
) -> None:
  for scenario in manifest.enabled():
    n_samples = options.n_samples or scenario.n_samples
    frame = apply_scenario(
      fitted, scenario, n_samples=n_samples, seed=scenario.resolve_seed(0)
    )
    path = write_parquet(frame, drift_output_path(options.output_dir, scenario.name))
    print(f"  wrote {path} ({frame.height:,} rows)", flush=True)


def _run_reduced(
  fitted: FittedSynthesizer,
  manifest: ScenarioManifest,
  options: SuiteOptions,
) -> None:
  for scenario in manifest.enabled():
    source = drift_output_path(options.output_dir, scenario.name)
    n_samples = options.n_samples or scenario.n_samples
    if source.exists():
      frame = pl.read_parquet(source)
    else:
      frame = apply_scenario(
        fitted, scenario, n_samples=n_samples, seed=scenario.resolve_seed(0)
      )
    target = min(REDUCED_N_SAMPLES, frame.height)
    drawn = subsample(frame, target, REDUCED_SEED)
    path = write_parquet(drawn, reduced_output_path(options.output_dir, scenario.name))
    print(f"  wrote {path} ({drawn.height:,} rows)", flush=True)


def _run_pairs(
  fitted: FittedSynthesizer,
  manifest: ScenarioManifest,
  options: SuiteOptions,
) -> None:
  drift_names = [
    scenario.name
    for scenario in manifest.enabled()
    if scenario.name != BASELINE_SCENARIO
  ]
  reference = manifest.require(drift_names[0])
  n_samples = options.n_samples or reference.n_samples
  pairs_root = options.output_dir / PAIRS_DIRNAME

  for drift_name in drift_names:
    scenario_dir = pairs_root / drift_name
    written = 0
    for replicate_index in range(options.replicates):
      plan = build_pair_plan(manifest, drift_name, replicate_index, n_samples)
      t1, t2 = generate_pair(fitted, plan, manifest)
      write_pair(scenario_dir / plan.label, plan, t1, t2)
      written += 1
      if written % 25 == 0:
        print(
          f"  {drift_name}: {written}/{options.replicates} pair(s) written",
          flush=True,
        )
    print(f"  {drift_name}: done ({written} pair(s))", flush=True)
