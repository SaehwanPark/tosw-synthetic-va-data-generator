"""Scenario application, generation, and subsampling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
from tabdat_synth import (
  FittedSynthesizer,
  GenerationStep,
  generate_from_fitted,
)

from tosw_synth.scenario import Scenario

# Internal log-odds columns stay in the library frame but never reach disk.
INTERNAL_COLUMNS = ("hosp_log_odds", "mort_log_odds")
# The historical scripts appended the outcome columns in this order.
OUTCOME_COLUMNS = ("pHosp_1y", "pMort_1y", "Hosp_1y", "Mort_1y")


def apply_scenario(
  fitted: FittedSynthesizer,
  scenario: Scenario,
  n_samples: int | None = None,
  seed: int | None = None,
) -> pl.DataFrame:
  """Generate one scenario frame from a fitted bundle.

  The age shift goes on the ``truncated_normal`` step, and ``alpha``/``beta``
  go on both ``bernoulli_response`` steps. Both the spec steps and the matching
  fitted artifacts are replaced so the two copies stay in agreement; the library
  rejects a bundle whose spec and artifact steps diverge.
  """
  resolved_n = scenario.n_samples if n_samples is None else n_samples
  resolved_seed = scenario.seed if seed is None else seed
  if resolved_n <= 0:
    raise ValueError("n_samples must be positive")
  if resolved_seed is None:
    raise ValueError(f"scenario '{scenario.name}' has no seed")

  new_steps = tuple(_patched_step(step, scenario) for step in fitted.spec.steps)
  steps_by_output = {step.output_column: step for step in new_steps}
  new_artifacts = tuple(
    replace(artifact, step=steps_by_output[artifact.step.output_column])
    for artifact in fitted.step_artifacts
  )

  patched = replace(
    fitted,
    spec=replace(
      fitted.spec,
      steps=new_steps,
      n_samples=resolved_n,
      random_seed=resolved_seed,
    ),
    step_artifacts=new_artifacts,
  )

  frame = generate_from_fitted(patched)
  frame = frame.drop([c for c in INTERNAL_COLUMNS if c in frame.columns])
  ordered = [
    name
    for name in _declared_output_order(fitted)
    if name in frame.columns and name not in INTERNAL_COLUMNS
  ]
  if len(ordered) != frame.width:
    raise ValueError(
      "generated columns do not match the declared step order: "
      f"missing {sorted(set(frame.columns) - set(ordered))}"
    )
  return frame.select(ordered)


def _declared_output_order(fitted: FittedSynthesizer) -> list[str]:
  """Return output columns in the order the spec declares them.

  The library executes steps in dependency order, which can interleave terminal
  steps. Writing columns in declared order keeps the file layout stable and
  matching the source convention.
  """
  return [step.output_column for step in fitted.spec.steps]


def _patched_step(step: GenerationStep, scenario: Scenario) -> GenerationStep:
  if step.kind == "truncated_normal":
    return replace(step, mean_shift=scenario.age_shift)
  if step.kind == "bernoulli_response":
    assert step.response is not None
    return replace(
      step,
      response=replace(
        step.response,
        intercept=scenario.alpha,
        slope=scenario.beta,
      ),
    )
  return step


def subsample(
  frame: pl.DataFrame,
  n_samples: int,
  seed: int,
) -> pl.DataFrame:
  """Draw ``n_samples`` rows without replacement from a generated frame."""
  if n_samples <= 0:
    raise ValueError("n_samples must be positive")
  if n_samples > frame.height:
    raise ValueError(
      f"cannot draw {n_samples} rows without replacement from {frame.height} rows"
    )
  return frame.sample(n=n_samples, with_replacement=False, seed=seed)


def write_parquet(frame: pl.DataFrame, output_path: Path | str) -> Path:
  """Write a generated frame to parquet, creating parent directories."""
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.write_parquet(path)
  return path
