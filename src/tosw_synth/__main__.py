"""Command-line entry point: ``python -m tosw_synth``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from tabdat_synth import load_fitted_synthesizer

from tosw_synth.generate import (
  PairPlan,
  apply_scenario,
  build_pair_plan,
  subsample,
  write_pair_index,
  write_parquet,
)
from tosw_synth.scenario import Scenario, load_manifest
from tosw_synth.suite import SuiteOptions, build_suite_steps, run_suite

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
    help="replicate index selecting the random stream (default: 0)",
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

  pairs = subparsers.add_parser(
    "pairs",
    help="generate repeated T1-T2 sample pairs for the drift grid",
  )
  pairs.add_argument(
    "--replicates",
    type=int,
    default=1000,
    help="number of replicate pairs per drift scenario (default: 1000)",
  )
  pairs.add_argument(
    "--drift",
    action="append",
    default=None,
    help=(
      "drift scenario to pair against T1; repeatable "
      "(default: every enabled non-baseline scenario)"
    ),
  )
  pairs.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help="rows per time point (default: the manifest value)",
  )
  pairs.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="root directory for pair output (default: data/pairs)",
  )
  pairs.add_argument(
    "--replicate-start",
    type=int,
    default=0,
    help="first replicate index to generate (default: 0)",
  )
  pairs.add_argument(
    "--skip-existing",
    action="store_true",
    help="skip pairs whose two files already exist",
  )

  suite = subparsers.add_parser(
    "suite",
    help="generate every dataset a monitoring study needs in one run",
  )
  suite.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help="root output directory (default: data/)",
  )
  suite.add_argument(
    "--replicates",
    type=int,
    default=1000,
    help="replicate pairs per drift scenario (default: 1000)",
  )
  suite.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help=(
      "override rows per table for a quick trial (default: the manifest "
      "row count)"
    ),
  )
  suite.add_argument(
    "--skip-reduced",
    action="store_true",
    help="skip the reduced-sample 100,000-row copies",
  )
  suite.add_argument(
    "--skip-pairs",
    action="store_true",
    help="skip the repeated pair grid",
  )
  suite.add_argument(
    "--dry-run",
    action="store_true",
    help="print the plan without generating anything",
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
  if args.command == "pairs":
    return _run_pairs(args, parser)
  if args.command == "suite":
    return _run_suite(args, parser)

  parser.error(f"unknown command '{args.command}'")
  return 2


def _run_scenarios(args: argparse.Namespace) -> int:
  manifest = load_manifest(args.manifest)
  scenarios = manifest.scenarios if args.all else manifest.enabled()
  print(
    f"{'scenario':28s} {'enabled':7s} {'shift':>6s} "
    f"{'alpha':>7s} {'beta':>6s} {'seed':>6s}"
  )
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


def _run_pairs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
  manifest = load_manifest(args.manifest)
  if args.replicates <= 0:
    parser.error("--replicates must be positive")
  if args.replicate_start < 0:
    parser.error("--replicate-start must be zero or greater")
  if args.n_samples is not None and args.n_samples <= 0:
    parser.error("--n-samples must be positive")

  drift_names = args.drift or [
    scenario.name for scenario in manifest.enabled() if scenario.name != "synth_t1_ageshift"
  ]
  if not drift_names:
    parser.error("no drift scenarios selected")

  reference = manifest.require(drift_names[0])
  n_samples = args.n_samples or reference.n_samples
  output_root = args.output_dir or Path("data") / "pairs"

  fitted = load_fitted_synthesizer(args.bundle)
  baseline_dir = output_root / "synth_t1_ageshift"
  total_written = 0
  total_skipped = 0
  baseline_written = 0

  # T1 is a pure function of the replicate index, so write it once per
  # replicate and let every drift scenario reuse it.
  baseline_scenario = manifest.require("synth_t1_ageshift")
  for offset in range(args.replicates):
    replicate_index = args.replicate_start + offset
    target = baseline_dir / f"replicate_r{replicate_index:04d}" / "t1.parquet"
    if args.skip_existing and target.exists():
      total_skipped += 1
      continue
    frame = apply_scenario(
      fitted,
      baseline_scenario,
      n_samples=n_samples,
      seed=baseline_scenario.resolve_seed(replicate_index),
    )
    write_parquet(frame, target)
    baseline_written += 1
    total_written += 1
    if baseline_written % 25 == 0:
      print(
        f"  synth_t1_ageshift: {baseline_written}/{args.replicates} "
        f"t1 draw(s) written",
        flush=True,
      )
  print(f"  synth_t1_ageshift: done ({baseline_written} t1 draw(s))", flush=True)

  all_plans: list[PairPlan] = []
  for drift_name in drift_names:
    scenario_dir = output_root / drift_name
    scenario_written = 0
    for offset in range(args.replicates):
      replicate_index = args.replicate_start + offset
      plan = build_pair_plan(manifest, drift_name, replicate_index, n_samples)
      target = scenario_dir / plan.label / "t2.parquet"
      all_plans.append(plan)

      if args.skip_existing and target.exists():
        total_skipped += 1
        continue

      frame = apply_scenario(
        fitted,
        manifest.require(drift_name),
        n_samples=n_samples,
        seed=plan.t2_seed,
      )
      write_parquet(frame, target)
      scenario_written += 1
      total_written += 1
      if scenario_written % 25 == 0:
        print(
          f"  {drift_name}: {scenario_written}/{args.replicates} t2 draw(s) written",
          flush=True,
        )
    print(
      f"  {drift_name}: done ({scenario_written} written, "
      f"{args.replicates - scenario_written} skipped)",
      flush=True,
    )

  index_path = write_pair_index(output_root, all_plans)
  print(
    f"pairs: wrote {total_written} draw(s) under {output_root}, "
    f"skipped {total_skipped}"
  )
  print(f"pair index: {index_path} ({len(all_plans)} row(s))")
  return 0


def _run_suite(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
  if args.replicates <= 0:
    parser.error("--replicates must be positive")
  if args.n_samples is not None and args.n_samples <= 0:
    parser.error("--n-samples must be positive")

  manifest = load_manifest(args.manifest)
  options = SuiteOptions(
    output_dir=args.output_dir,
    replicates=args.replicates,
    include_reduced=not args.skip_reduced,
    include_pairs=not args.skip_pairs,
    n_samples=args.n_samples,
  )
  steps = build_suite_steps(manifest, options)

  if args.dry_run:
    print(f"suite plan for {args.output_dir} ({len(steps)} step(s)):")
    for label, description in steps:
      print(f"  [{label}] {description}")
    return 0

  fitted = load_fitted_synthesizer(args.bundle)
  run_suite(fitted, manifest, options, steps)
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
