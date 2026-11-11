"""End-to-end checks against the committed bundle.

These tests only read the committed fitted bundle. They never touch the
original VA extract and do not require ``VA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest
from tabdat_synth import load_fitted_synthesizer

from tosw_synth.generate import (
  apply_scenario,
  build_pair_plan,
  generate_pair,
  subsample,
)
from tosw_synth.scenario import load_manifest
from tosw_synth.suite import (
  SuiteOptions,
  build_suite_steps,
  drift_output_path,
  reduced_output_path,
  run_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO_ROOT / "bundles" / "can2014"
MANIFEST_PATH = REPO_ROOT / "configs" / "scenarios.yaml"

REQUIRED_COLUMNS = ("age_cont", "agegrp", "pHosp_1y", "Hosp_1y", "pMort_1y", "Mort_1y")
TRAILING_OUTCOME_COLUMNS = ("pHosp_1y", "pMort_1y", "Hosp_1y", "Mort_1y")


@pytest.fixture(scope="module")
def fitted():
  return load_fitted_synthesizer(BUNDLE_PATH)


@pytest.fixture(scope="module")
def manifest():
  return load_manifest(MANIFEST_PATH)


def test_bundle_generates_without_source_extract(fitted, monkeypatch):
  monkeypatch.delenv("VA_DIR", raising=False)
  placeholder = fitted.spec.source_path
  assert not placeholder.exists()

  frame = apply_scenario(fitted, load_manifest(MANIFEST_PATH).require("synth_t1_ageshift"), n_samples=16)
  assert frame.height == 16


def test_same_scenario_and_replicate_is_repeatable(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  seed = scenario.resolve_seed(0)

  first = apply_scenario(fitted, scenario, n_samples=64, seed=seed)
  second = apply_scenario(fitted, scenario, n_samples=64, seed=seed)

  assert first.equals(second)


def test_different_replicate_index_changes_values(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")

  first = apply_scenario(fitted, scenario, n_samples=64, seed=scenario.resolve_seed(0))
  second = apply_scenario(fitted, scenario, n_samples=64, seed=scenario.resolve_seed(1))

  assert not first.equals(second)


def test_output_columns_and_row_count(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  frame = apply_scenario(fitted, scenario, n_samples=64, seed=scenario.resolve_seed(0))

  assert frame.height == 64
  for column in REQUIRED_COLUMNS:
    assert column in frame.columns
  assert "hosp_log_odds" not in frame.columns
  assert "mort_log_odds" not in frame.columns
  assert frame.columns[-4:] == list(TRAILING_OUTCOME_COLUMNS)


def test_age_group_values_are_can_labels(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  frame = apply_scenario(fitted, scenario, n_samples=512, seed=scenario.resolve_seed(0))

  labels = set(frame["agegrp"].cast(pl.String).to_list())
  assert labels <= {"1", "2", "3", "4", "5"}


def test_age_group_is_a_deterministic_function_of_age(fitted, manifest):
  scenario = manifest.require("synth_t2_ageshift4")
  frame = apply_scenario(fitted, scenario, n_samples=1024, seed=scenario.resolve_seed(0))

  ages = frame["age_cont"].to_list()
  expected = []
  for age in ages:
    if age <= 54:
      expected.append("1")
    elif age <= 64:
      expected.append("2")
    elif age <= 74:
      expected.append("3")
    elif age <= 84:
      expected.append("4")
    else:
      expected.append("5")
  assert frame["agegrp"].cast(pl.String).to_list() == expected


def test_age_shift_raises_mean_age(fitted, manifest):
  low = apply_scenario(
    fitted,
    manifest.require("synth_t2_ageshift0"),
    n_samples=4000,
    seed=manifest.require("synth_t2_ageshift0").resolve_seed(0),
  )
  high = apply_scenario(
    fitted,
    manifest.require("synth_t2_ageshift10"),
    n_samples=4000,
    seed=manifest.require("synth_t2_ageshift10").resolve_seed(0),
  )

  assert high["age_cont"].mean() - low["age_cont"].mean() > 4.0


def test_lower_intercept_reduces_outcome_rate(fitted, manifest):
  high = apply_scenario(
    fitted,
    manifest.require("synth_t2_intercept0"),
    n_samples=4000,
    seed=manifest.require("synth_t2_intercept0").resolve_seed(0),
  )
  low = apply_scenario(
    fitted,
    manifest.require("synth_t2_intercept-100"),
    n_samples=4000,
    seed=manifest.require("synth_t2_intercept-100").resolve_seed(0),
  )

  assert low["Hosp_1y"].mean() < high["Hosp_1y"].mean()
  assert low["Mort_1y"].mean() < high["Mort_1y"].mean()


def test_outcomes_are_binary_and_probabilities_bounded(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  frame = apply_scenario(fitted, scenario, n_samples=512, seed=scenario.resolve_seed(0))

  assert set(frame["Hosp_1y"].unique().to_list()) <= {0, 1}
  assert set(frame["Mort_1y"].unique().to_list()) <= {0, 1}
  assert frame["pHosp_1y"].is_between(0.0, 1.0).all()
  assert frame["pMort_1y"].is_between(0.0, 1.0).all()
  assert frame.null_count().sum_horizontal().item() == 0


def test_subsample_draws_without_replacement(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  base = apply_scenario(fitted, scenario, n_samples=250, seed=scenario.resolve_seed(0))

  drawn = subsample(base, 100, seed=343)

  assert drawn.height == 100
  assert drawn.columns == base.columns

  repeat = subsample(base, 100, seed=343)
  assert drawn.equals(repeat)

  other = subsample(base, 100, seed=999)
  assert not drawn.equals(other)


def test_subsample_rejects_oversized_draw(fitted, manifest):
  scenario = manifest.require("synth_t1_ageshift")
  base = apply_scenario(fitted, scenario, n_samples=32, seed=scenario.resolve_seed(0))

  with pytest.raises(ValueError, match="without replacement"):
    subsample(base, 33, seed=343)


def test_bundle_spec_has_no_private_path():
  stored = (BUNDLE_PATH / "spec.yaml").read_text()

  assert "/Volumes/" not in stored
  assert "/Users/" not in stored
  assert "va_algo" not in stored
  assert "source/not_distributed.parquet" in stored


def test_bundle_tree_has_no_private_path():
  for path in BUNDLE_PATH.rglob("*"):
    if not path.is_file():
      continue
    if path.suffix == ".pkl":
      continue  # pickled models are binary; covered by the spec check above
    text = path.read_text(errors="ignore")
    for needle in ("/Volumes/", "/Users/", "va_algo", "patienticn"):
      assert needle not in text, f"{needle} leaked into {path.relative_to(BUNDLE_PATH)}"


def test_manifest_enables_only_the_published_grid(manifest):
  enabled = {scenario.name for scenario in manifest.enabled()}

  expected = {"synth_t1_ageshift", "synth_t2_intercept0"}
  expected |= {f"synth_t2_ageshift{shift}" for shift in range(0, 11, 2)}
  expected |= {f"synth_t2_slope{v}" for v in range(80, 101, 2)}
  expected |= {f"synth_t2_intercept{v}" for v in range(0, -101, -5)}

  assert enabled == expected
  assert len(enabled) == 39


def test_manifest_records_disabled_families(manifest):
  names = {scenario.name for scenario in manifest.scenarios}

  assert "synth_t1_prior1245" in names
  for shift in range(0, 30, 5):
    assert f"synth_t2_priorityshift{shift}" in names
  for value in range(60, 121, 2):
    if 80 <= value <= 100:
      continue
    assert f"synth_t2_slope{value}" in names
  for disease in ("hxchf", "metasca", "dementia"):
    assert any(name.startswith(f"{disease}/") for name in names)


def test_manifest_enabled_scenarios_have_seeds(manifest):
  for scenario in manifest.enabled():
    assert scenario.seed is not None
    assert scenario.n_samples == 250_000
    assert scenario.beta > 0


def test_manifest_seeds_follow_the_published_formula(manifest):
  assert manifest.require("synth_t1_ageshift").seed == 343
  for shift in range(0, 11, 2):
    assert manifest.require(f"synth_t2_ageshift{shift}").seed == 345 + shift // 2
  for value in range(80, 101, 2):
    assert manifest.require(f"synth_t2_slope{value}").seed == 357 + (120 - value) // 2
  for value in range(0, -101, -5):
    assert (
      manifest.require(f"synth_t2_intercept{value}").seed == 388 + (-value) // 5
    )


def test_replicate_index_shifts_the_seed(manifest):
  from tosw_synth.scenario import derive_replicate_seed

  scenario = manifest.require("synth_t2_ageshift2")
  assert scenario.resolve_seed(0) == scenario.seed
  assert scenario.resolve_seed(7) == derive_replicate_seed(scenario.seed, 7)
  assert scenario.resolve_seed(7) != scenario.seed + 7
  assert scenario.resolve_seed(7) != scenario.resolve_seed(8)


def test_replicate_zero_is_unchanged_from_the_published_draw(fitted, manifest):
  """Replicate 0 must keep the exact stream the current release publishes."""
  import hashlib

  scenario = manifest.require("synth_t1_ageshift")
  assert scenario.resolve_seed(0) == 343

  frame = apply_scenario(fitted, scenario, n_samples=500, seed=scenario.resolve_seed(0))
  digest = hashlib.sha256(frame.write_csv().encode()).hexdigest()

  # Pinned from the released generator. If this changes, replicate 0 moved and
  # every published table would need regenerating.
  assert digest == (
    "cd1472ea73007d391a6a02572dc5dba86e7e1b6154c23f68146d0bdd12f63eab"
  )


def test_all_scenario_replicate_streams_are_distinct(manifest):
  seen: dict[int, tuple[str, int]] = {}
  for scenario in manifest.enabled():
    for replicate in range(1000):
      seed = scenario.resolve_seed(replicate)
      assert seed not in seen, (
        f"seed {seed} shared by {scenario.name}[r={replicate}] "
        f"and {seen[seed]}"
      )
      seen[seed] = (scenario.name, replicate)

  assert len(seen) == len(manifest.enabled()) * 1000


def test_first_time_point_is_independent_of_the_second(manifest):
  baseline = manifest.require("synth_t1_ageshift")
  drift = manifest.require("synth_t2_ageshift0")

  # Before the fix, baseline[r+2] collided with drift[r].
  assert baseline.resolve_seed(2) != drift.resolve_seed(0)
  assert baseline.resolve_seed(4) != drift.resolve_seed(2)


def test_replicate_pairs_generate_independent_frames(fitted, manifest):
  plan = build_pair_plan(
    manifest, "synth_t2_intercept-50", replicate_index=0, n_samples=500
  )
  t1, t2 = generate_pair(fitted, plan, manifest)
  assert not t1.equals(t2)
  assert t1.columns == t2.columns

  other = build_pair_plan(
    manifest, "synth_t2_intercept-50", replicate_index=1, n_samples=500
  )
  other_t1, other_t2 = generate_pair(fitted, other, manifest)
  assert not t1.equals(other_t1)
  assert not t2.equals(other_t2)


def test_pair_plan_matches_manifest_seeds(manifest):
  plan = build_pair_plan(
    manifest, "synth_t2_slope82", replicate_index=5, n_samples=123
  )

  assert plan.t1_scenario == "synth_t1_ageshift"
  assert plan.t2_scenario == "synth_t2_slope82"
  assert plan.t1_seed == manifest.require("synth_t1_ageshift").resolve_seed(5)
  assert plan.t2_seed == manifest.require("synth_t2_slope82").resolve_seed(5)
  assert plan.n_samples == 123
  assert plan.label == "replicate_r0005"


def test_suite_dry_run_lists_the_three_dataset_kinds(manifest):
  options = SuiteOptions(output_dir=Path("/tmp/suite_plan"))
  steps = build_suite_steps(manifest, options)
  labels = [label for label, _ in steps]

  assert labels == ["drift", "reduced", "pairs"]


def test_suite_skips_pairs_and_reduced_on_request(manifest):
  options = SuiteOptions(
    output_dir=Path("/tmp/suite_plan"),
    include_reduced=False,
    include_pairs=False,
  )
  labels = [label for label, _ in build_suite_steps(manifest, options)]

  assert labels == ["drift"]


def test_suite_writes_the_expected_tree(tmp_path, fitted, manifest):
  options = SuiteOptions(
    output_dir=tmp_path,
    replicates=1,
    n_samples=64,
  )
  run_suite(fitted, manifest, options)

  drift = sorted((tmp_path / "drift").glob("*.parquet"))
  reduced = sorted((tmp_path / "reduced").glob("*.parquet"))
  pair_dirs = sorted((tmp_path / "pairs").glob("*/replicate_r0000"))

  assert len(drift) == len(manifest.enabled())
  assert len(reduced) == len(manifest.enabled())
  assert len(pair_dirs) == len(manifest.enabled()) - 1

  first_pair = pair_dirs[0]
  assert (first_pair / "t1.parquet").exists()
  assert (first_pair / "t2.parquet").exists()

  t1 = pl.read_parquet(first_pair / "t1.parquet")
  t2 = pl.read_parquet(first_pair / "t2.parquet")
  assert t1.height == 64
  assert t2.height == 64
  assert not t1.equals(t2)


def test_suite_output_paths_are_stable(tmp_path):
  path = drift_output_path(tmp_path, "synth_t2_slope82")
  reduced = reduced_output_path(tmp_path, "synth_t2_slope82")

  assert path == tmp_path / "drift" / "synth_t2_slope82.parquet"
  assert reduced == tmp_path / "reduced" / "synth_t2_slope82_n100000.parquet"


def test_disabled_scenario_without_seed_cannot_be_generated(manifest):
  scenario = manifest.require("hxchf/synth_t1_ageshift")
  assert scenario.seed is None

  with pytest.raises(ValueError, match="has no seed"):
    scenario.resolve_seed(0)


def test_generation_works_with_va_dir_unset(fitted, monkeypatch):
  monkeypatch.delenv("VA_DIR", raising=False)
  assert "VA_DIR" not in os.environ

  frame = apply_scenario(
    fitted,
    load_manifest(MANIFEST_PATH).require("synth_t1_ageshift"),
    n_samples=8,
    seed=343,
  )
  assert frame.height == 8
