"""Scenario manifest loading and seed derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_MANIFEST_PATH = (
  Path(__file__).resolve().parents[2] / "configs" / "scenarios.yaml"
)


@dataclass(frozen=True)
class Scenario:
  """One row of the scenario manifest."""

  name: str
  enabled: bool
  age_shift: float
  alpha: float
  beta: float
  n_samples: int
  seed: int | None
  family: str = ""

  def resolve_seed(self, replicate_index: int) -> int:
    """Return the seed for one replicate of this scenario."""
    if self.seed is None:
      raise ValueError(
        f"scenario '{self.name}' has no seed and cannot be generated; "
        "it is recorded for reference only"
      )
    if replicate_index < 0:
      raise ValueError("replicate_index must be zero or greater")
    return self.seed + replicate_index


@dataclass(frozen=True)
class ScenarioManifest:
  """Loaded scenario manifest."""

  scenarios: tuple[Scenario, ...]

  def by_name(self) -> dict[str, Scenario]:
    """Index the manifest by scenario name."""
    return {scenario.name: scenario for scenario in self.scenarios}

  def enabled(self) -> tuple[Scenario, ...]:
    """Return the scenarios enabled for generation."""
    return tuple(scenario for scenario in self.scenarios if scenario.enabled)

  def require(self, name: str) -> Scenario:
    """Look up one scenario by name or raise a helpful error."""
    scenario = self.by_name().get(name)
    if scenario is None:
      known = ", ".join(sorted(self.by_name()))
      raise ValueError(f"unknown scenario '{name}'; known scenarios: {known}")
    return scenario


def load_manifest(path: Path | str | None = None) -> ScenarioManifest:
  """Load the scenario manifest from YAML."""
  manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
  raw = yaml.safe_load(manifest_path.read_text())
  if not isinstance(raw, dict):
    raise TypeError("scenario manifest must be a mapping")
  raw_scenarios = raw.get("scenarios", [])
  if not isinstance(raw_scenarios, list):
    raise TypeError("scenario manifest 'scenarios' must be a list")

  scenarios = []
  seen: set[str] = set()
  for entry in raw_scenarios:
    if not isinstance(entry, dict):
      raise TypeError("scenario manifest entries must be mappings")
    name = str(entry["name"])
    if name in seen:
      raise ValueError(f"duplicate scenario name '{name}'")
    seen.add(name)
    raw_seed = entry.get("seed")
    scenarios.append(
      Scenario(
        name=name,
        enabled=bool(entry.get("enabled", False)),
        age_shift=float(entry.get("age_shift", 0.0)),
        alpha=float(entry.get("alpha", 0.0)),
        beta=float(entry.get("beta", 1.0)),
        n_samples=int(entry.get("n_samples", 250_000)),
        seed=None if raw_seed is None else int(raw_seed),
        family=str(entry.get("family", "")),
      )
    )

  return ScenarioManifest(scenarios=tuple(scenarios))
