"""Command-line entry point: ``python -m tosw_synth``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from tabdat_synth import load_fitted_synthesizer

from tosw_synth.generate import apply_scenario, subsample, write_parquet
from tosw_synth.scenario import Scenario, load_manifest

DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "bundles" / "can2014"
DEFAULT_OUTPUT_DIR = Path("data")


def build_parser() -> argparse.ArgumentParser:
  """Build the command-line parser."""
  parser = argparse.ArgumentParser(
    prog="python -m tosw_synth",
    description=(
      "Generate semi-synthetic VA Care Assessment Needs (CAN) datasets "
      "from a fitted bundle."
    ),
  )
  parser.add_argument(
    "--bundle",
    type=Path,
    default=DEFAULT_BUNDLE,
    help="fitted bundle directory (default: bundles/can2014)",
  )
  parser.add_argument(
    "--manifest",
    type=Path,
    default=None,
    help="scenario manifest path (default: configs/scenarios.yaml)",
  )

  subparsers = parser.add_subparsers(dest="command", required=True)

  generate = subparsers.add_parser("generate", help="generate scenario tables")
  target = generate.add_mutually_exclusive_group(required=True)
  target.add_argument("--scenario", help="scenario name from the manifest")
  target.add_argument(
    "--enabled",
    action="store_true",
    help="generate every enabled scenario in the manifest",
  )
  generate.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help="row count for this run (default: the manifest value)",
  )
  generate.add_argument(
    "--replicate-index",
    type=int,
    default=0,
    help="replicate index added to the scenario seed (default: 0)",
  )
  generate.add_argument(
    "--output",
    type=Path,
    default=None,
    help="output parquet path; only valid with --scenario",
  )
  generate.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="directory for --enabled output (default: data/)",
  )

  sample = subparsers.add_parser(
    "subsample", help="draw rows without replacement from a generated file"
  )
  sample.add_argument("--input", type=Path, required=True, help="input parquet")
  sample.add_argument(
    "--n-samples", type=int, required=True, help="rows to draw"
  )
  sample.add_argument("--seed", type=int, default=343, help="draw seed")
  sample.add_argument("--output", type=Path, required=True, help="output parquet")

  describe = subparsers.add_parser(
    "scenarios", help="list manifest scenarios and their settings"
  )
  describe.add_argument(
    "--all",
    action="store_true",
    help="include records that are disabled for generation",
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  """Run the command-line interface."""
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.command == "scenarios":
    return _run_scenarios(args)
  if args.command == "subsample":
    return _run_subsample(args)
  if args.command == "generate":
    return _run_generate(args, parser)

  parser.error(f"unknown command '{args.command}'")
  return 2


def _run_scenarios(args: argparse.Namespace) -> int:
  manifest = load_manifest(args.manifest)
  scenarios = manifest.scenarios if args.all else manifest.enabled()
  print(f"{'scenario':28s} {'enabled':7s} {'shift':>6s} {'alpha':>7s} {'beta':>6s} {'seed':>6s}")
  for scenario in scenarios:
    seed = "-" if scenario.seed is None else str(scenario.seed)
    print(
      f"{scenario.name:28s} {scenario.enabled!s:7s} "
      f"{scenario.age_shift:6.1f} {scenario.alpha:7.2f} {scenario.beta:6.2f} {seed:>6s}"
    )
  print(f"\n{len(scenarios)} scenario(s) listed")
  return 0


def _run_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
  manifest = load_manifest(args.manifest)
  n_samples = args.n_samples

  if args.enabled:
    if args.output is not None:
      parser.error("--output cannot be combined with --enabled; use --output-dir")
    if n_samples is not None and n_samples <= 0:
      parser.error("--n-samples must be positive")

    selected = manifest.enabled()
    if not selected:
      parser.error("the manifest has no enabled scenarios")
    fitted = load_fitted_synthesizer(args.bundle)
    for scenario in selected:
      frame = apply_scenario(
        fitted,
        scenario,
        n_samples=n_samples,
        seed=scenario.resolve_seed(args.replicate_index),
      )
      path = write_parquet(frame, args.output_dir / f"{scenario.name}.parquet")
      print(f"wrote {path} ({frame.height:,} rows)")
    return 0

  assert args.scenario is not None
  scenario = manifest.require(args.scenario)
  if n_samples is not None and n_samples <= 0:
    parser.error("--n-samples must be positive")
  if n_samples is None and scenario.n_samples <= 0:
    parser.error(f"scenario '{scenario.name}' has no usable n_samples")

  seed = scenario.resolve_seed(args.replicate_index)
  fitted = load_fitted_synthesizer(args.bundle)
  frame = apply_scenario(fitted, scenario, n_samples=n_samples, seed=seed)

  output = args.output
  if output is None:
    suffix = "" if args.replicate_index == 0 else f"_r{args.replicate_index:04d}"
    output = args.output_dir / f"{scenario.name}{suffix}.parquet"

  path = write_parquet(frame, output)
  print(f"wrote {path} ({frame.height:,} rows x {len(frame.columns)} columns)")
  _print_summary(frame, scenario, seed)
  return 0


def _run_subsample(args: argparse.Namespace) -> int:
  if not args.input.exists():
    raise SystemExit(f"input not found: {args.input}")
  frame = pl.read_parquet(args.input)
  drawn = subsample(frame, args.n_samples, args.seed)
  path = write_parquet(drawn, args.output)
  print(
    f"wrote {path} ({drawn.height:,} rows drawn without replacement "
    f"from {frame.height:,})"
  )
  return 0


def _print_summary(frame: pl.DataFrame, scenario: Scenario, seed: int) -> None:
  print(f"  scenario: {scenario.name}")
  print(f"  age shift: {scenario.age_shift:+.1f} years")
  print(f"  outcome alpha: {scenario.alpha:+.2f}, beta: {scenario.beta:.2f}")
  print(f"  seed: {seed}")
  print(f"  mean age: {frame['age_cont'].mean():.2f}")
  print(f"  hospitalization rate: {frame['Hosp_1y'].mean():.4f}")
  print(f"  mortality rate: {frame['Mort_1y'].mean():.4f}")


if __name__ == "__main__":
  sys.exit(main())
