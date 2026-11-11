"""Author-only rebuild of the fitted CAN bundle.

This module needs the original 2014 VA CAN extract, which is not distributed.
A normal user never runs it: the committed bundle under ``bundles/can2014``
already carries every fitted parameter needed to generate data.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import polars as pl
from tabdat_synth import (
  BernoulliResponseSpec,
  BinMapSpec,
  CoefficientSumSpec,
  CoefficientTerm,
  ColumnSpec,
  DatasetSchema,
  GenerationSpec,
  GenerationStep,
  ModelBackendSpec,
  fit_synthesizer,
  save_fitted_synthesizer,
)

AGE_COLUMN = "age_cont"
AGE_GROUP_COLUMN = "agegrp"
AGE_BREAKS = (54.0, 64.0, 74.0, 84.0)
AGE_LABELS = ("1", "2", "3", "4", "5")
SOURCE_FILENAME = "all_can_2014_v2_ordinalver.parquet"
SOURCE_PLACEHOLDER = "source/not_distributed.parquet"
FIT_FRACTION = 0.5
FIT_SEED = 343
N_ESTIMATORS = 100
REFERENCE_VALUE = "99"

DROP_COLUMNS = ("patienticn", "yr", "Death1y", "intercept")
RESPONSE_COLUMNS = ("Hosp_1y", "Mort_1y", "pHosp_1y", "pMort_1y")
AGE_COLUMNS = (AGE_COLUMN, AGE_GROUP_COLUMN)

LOG_ODDS_COLUMNS = ("hosp_log_odds", "mort_log_odds")
PROBABILITY_COLUMNS = ("pHosp_1y", "pMort_1y")
BINARY_OUTCOME_COLUMNS = ("Hosp_1y", "Mort_1y")

HOSPITALIZATION_COEFFICIENTS = {
  "agegrp_c_1": 0.11195135687796806,
  "agegrp_c_2": 0.3829908347336344,
  "agegrp_c_3": 0.1170440313539304,
  "agegrp_c_4": -0.0005318645488637682,
  "Albumin_c_0": -0.03809892539309278,
  "Albumin_c_1": 0.25407421773906447,
  "AntiPsychotic_c_0": -0.3298482906983808,
  "BB_c_0": -0.2895503337138029,
  "BUN_c_0": 0.2543661084368837,
  "BUN_c_1": 0.23868025203403645,
  "BUN_c_2": 0.2937981776666459,
  "covwtflag_c_0": -0.25025532925442856,
  "cpt21to30fon730_c_0": -0.5589833785819982,
  "cpt21to30fon730_c_1": -0.19968189621269836,
  "cptOffVisEst90_c_0": -0.9782689213227374,
  "cptOffVisEst90_c_1": -0.5684071312689593,
  "deyo_c_0": -0.6788404706458051,
  "deyo_c_1": -0.4986713060119847,
  "deyo_c_2": -0.35097959162233605,
  "deyo_c_3": -0.2059603623205303,
  "diast_c_1": -0.09811790600445333,
  "diast_c_2": -0.14037557605784087,
  "dxAlcohol730_c_0": -0.5715002256100432,
  "dxChrAirObstNEC730_c_0": -0.2520873149332106,
  "furosemide_c_0": -0.1945527443791755,
  "HCC_dementia_c_0": -0.3374570797072577,
  "HCC_HXCHF_c_0": -0.28190262100546126,
  "HCC_METASCA_c_0": -0.14062464739308836,
  "hmgcoa_c_0": -0.057526048032501505,
  "lbLymphoSBPL365_c_0": -0.3271382198608108,
  "lbLymphoSBPL365_c_1": -0.17376987786663692,
  "lbRBCSynFlL730_c_0": -0.8370898823640706,
  "lbRBCSynFlL730_c_1": -0.5402929297898932,
  "mentptsd_c_0": -0.2718803207364405,
  "mentptsd_c_1": -0.06493459708622953,
  "Prior1245_c_0": -0.5094980935687793,
  "Prior1245_c_1": -0.2867314522456794,
  "Prior1245_c_2": -0.5337381860875798,
  "Prior1245_c_4": -0.05731908915786009,
  "pulse_c_1": -0.32841864894363376,
  "pulse_c_2": -0.2311486046328702,
  "resp_c_1": -0.05557713260514261,
  "resp_c_2": -0.020200919076358015,
  "syst_c_1": -0.21048681926710555,
  "syst_c_2": -0.29121559010083686,
  "syst_c_3": -0.12949914240995997,
  "intercept": 3.319921815594095,
}

MORTALITY_COEFFICIENTS = {
  "agegrp_c_1": -3.041659007749423,
  "agegrp_c_2": -1.9445312535988006,
  "agegrp_c_3": -1.4758485834234012,
  "agegrp_c_4": -0.7912478216455152,
  "Albumin_c_0": -0.13557725055521463,
  "Albumin_c_1": 0.5377253900720375,
  "AntiPsychotic_c_0": -0.23653245527986028,
  "BB_c_0": -0.14583384345121053,
  "BenZoDiZp_c_0": -0.14692869865125324,
  "BetaAgn_c_0": -0.30459776092306867,
  "BUN_c_0": -0.43652617992858445,
  "BUN_c_1": -0.46645002467694674,
  "BUN_c_2": -0.11489077691928529,
  "covwtflag_c_0": -0.5167087346352488,
  "cpt21to30fon730_c_0": -0.1953284762154348,
  "cpt21to30fon730_c_1": -0.04370541178013787,
  "cptOffVisEst90_c_0": 0.25877002111797287,
  "cptOffVisEst90_c_1": -0.02102949673572263,
  "deyo_c_0": -0.6980769959221858,
  "deyo_c_1": -0.4394886246641741,
  "deyo_c_2": -0.3545158906773217,
  "deyo_c_3": -0.2569899760554582,
  "diast_c_1": 0.14361520931713792,
  "diast_c_2": -0.04095153522957887,
  "dxAlcohol730_c_0": -0.29849906027385303,
  "dxChrAirObstNEC730_c_0": -0.3027158545945407,
  "furosemide_c_0": -0.2992441998276141,
  "HCC_dementia_c_0": -0.45778099619275336,
  "HCC_HXCHF_c_0": -0.2537350371171132,
  "HCC_METASCA_c_0": -0.7005136766217787,
  "hmgcoa_c_0": 0.2832049627184717,
  "lbLymphoSBPL365_c_0": -0.2811993636895115,
  "lbLymphoSBPL365_c_1": 0.00647749636701922,
  "lbRBCSynFlL730_c_0": -0.2378986535995422,
  "lbRBCSynFlL730_c_1": 0.08751336104963463,
  "lbSodimSbpL730_c_0": -0.3812581362577485,
  "lbSodimSbpL730_c_1": -0.20222695564803453,
  "lbSodimSbpL730_c_2": -0.16016863702173315,
  "lbSodimSbpL730_c_3": -0.12515491303381632,
  "mentptsd_c_0": -0.12344917283690982,
  "mentptsd_c_1": -0.2091478278055237,
  "nsaid_c_0": 0.21989580566004946,
  "Prior1245_c_0": -0.08505239529071978,
  "Prior1245_c_1": -0.2681967844722317,
  "Prior1245_c_2": -0.23795377489564162,
  "Prior1245_c_4": 0.03569584899839834,
  "pulse_c_1": -0.6231317038721609,
  "pulse_c_2": -0.38048078381549544,
  "resp_c_1": -0.1418368172932178,
  "resp_c_2": -0.0805804862021008,
  "sex_c_1": -0.4436664896872696,
  "syst_c_1": 0.2540071756139976,
  "syst_c_2": -0.06358133697337809,
  "syst_c_3": -0.12984733509751428,
  "tiuConsent365_c_0": 0.21328914021877987,
  "tiuConsent365_c_1": 0.014395085165553323,
  "tiuConsent365_c_2": -0.06926461383922747,
  "tiuTeleph730_c_0": -0.19675493847912373,
  "tiuTeleph730_c_1": -0.11961445814938668,
  "tiuTeleph730_c_2": -0.06246268326328519,
  "WBCC_c_0": -0.2429443613064096,
  "WBCC_c_1": 1.0180812918827922,
  "WBCC_c_2": 0.02920437603394846,
  "WBCC_c_3": 0.19218189180658704,
  "intercept": 3.1630656359719795,
}


def resolve_va_dir() -> Path:
  """Return the source-data directory from the environment."""
  _load_dotenv_if_available()
  value = os.getenv("VA_DIR")
  if not value:
    raise ValueError(
      "VA_DIR is not set. Export it to the directory that holds "
      f"{SOURCE_FILENAME} before running the author-only fit."
    )
  return Path(value)


def _load_dotenv_if_available() -> None:
  try:
    from dotenv import load_dotenv
  except ModuleNotFoundError:
    return
  load_dotenv()


def covariate_columns(prepared: pl.DataFrame) -> list[str]:
  """Return the sequentially generated covariates in source column order."""
  return [
    column
    for column in prepared.schema.names()
    if column not in RESPONSE_COLUMNS + AGE_COLUMNS
  ]


def load_fitting_sample(va_dir: Path) -> tuple[pl.DataFrame, list[str]]:
  """Read the 2014 CAN extract and draw the historical fitting sample."""
  source_path = va_dir / SOURCE_FILENAME
  if not source_path.exists():
    raise FileNotFoundError(f"CAN extract not found: {source_path.name}")
  frame = (
    pl.read_parquet(source_path)
    .drop(*DROP_COLUMNS)
    .sample(fraction=FIT_FRACTION, seed=FIT_SEED)
  )
  return frame, covariate_columns(frame)


def build_spec(frame: pl.DataFrame, target_columns: list[str]) -> GenerationSpec:
  """Build the CAN generation spec from a fitting frame."""
  _require_no_nulls(frame, target_columns)

  schema_columns = [
    ColumnSpec(
      name=AGE_COLUMN,
      data_type="integer",
      role="feature",
      minimum=float(frame[AGE_COLUMN].min()),
      maximum=float(frame[AGE_COLUMN].max()),
    ),
    ColumnSpec(
      name=AGE_GROUP_COLUMN,
      data_type="categorical",
      role="feature",
      allowed_values=AGE_LABELS,
    ),
  ]
  for column in target_columns:
    schema_columns.append(
      ColumnSpec(
        name=column,
        data_type="categorical",
        role="feature",
        allowed_values=tuple(
          sorted(str(value) for value in frame[column].unique().to_list())
        ),
      )
    )
  for column in LOG_ODDS_COLUMNS:
    schema_columns.append(ColumnSpec(name=column, data_type="float", role="outcome"))
  for column in PROBABILITY_COLUMNS:
    schema_columns.append(ColumnSpec(name=column, data_type="float", role="outcome"))
  for column in BINARY_OUTCOME_COLUMNS:
    schema_columns.append(ColumnSpec(name=column, data_type="integer", role="outcome"))

  steps: list[GenerationStep] = [
    GenerationStep(
      name=AGE_COLUMN,
      kind="truncated_normal",
      output_column=AGE_COLUMN,
      source_column=AGE_COLUMN,
      mean_shift=0.0,
    ),
    GenerationStep(
      name=AGE_GROUP_COLUMN,
      kind="bin_map",
      output_column=AGE_GROUP_COLUMN,
      input_column=AGE_COLUMN,
      bin_map=BinMapSpec(breaks=AGE_BREAKS, labels=AGE_LABELS),
    ),
  ]

  planned: tuple[str, ...] = (AGE_GROUP_COLUMN,)
  for column in target_columns:
    steps.append(
      GenerationStep(
        name=f"{column}_model",
        kind="categorical_model",
        output_column=column,
        source_column=column,
        parent_columns=planned,
        feature_columns=planned,
      )
    )
    planned = planned + (column,)

  coefficient_parents = _coefficient_parent_columns(steps, target_columns)
  steps.append(
    GenerationStep(
      name="hosp_log_odds",
      kind="coefficient_sum",
      output_column="hosp_log_odds",
      parent_columns=coefficient_parents,
      coefficient_sum=_coefficient_sum_spec(
        HOSPITALIZATION_COEFFICIENTS, frame, coefficient_parents
      ),
    )
  )
  steps.append(
    GenerationStep(
      name="mort_log_odds",
      kind="coefficient_sum",
      output_column="mort_log_odds",
      parent_columns=coefficient_parents,
      coefficient_sum=_coefficient_sum_spec(
        MORTALITY_COEFFICIENTS, frame, coefficient_parents
      ),
    )
  )
  steps.append(
    GenerationStep(
      name="pHosp_1y",
      kind="sigmoid_probability",
      output_column="pHosp_1y",
      input_column="hosp_log_odds",
    )
  )
  steps.append(
    GenerationStep(
      name="pMort_1y",
      kind="sigmoid_probability",
      output_column="pMort_1y",
      input_column="mort_log_odds",
    )
  )
  steps.append(
    GenerationStep(
      name="Hosp_1y",
      kind="bernoulli_response",
      output_column="Hosp_1y",
      input_column="hosp_log_odds",
      response=BernoulliResponseSpec(intercept=0.0, slope=1.0),
    )
  )
  steps.append(
    GenerationStep(
      name="Mort_1y",
      kind="bernoulli_response",
      output_column="Mort_1y",
      input_column="mort_log_odds",
      response=BernoulliResponseSpec(intercept=0.0, slope=1.0),
    )
  )

  return GenerationSpec(
    name="can2014",
    source_path=Path(SOURCE_PLACEHOLDER),
    source_format="parquet",
    schema=DatasetSchema(columns=tuple(schema_columns)),
    steps=tuple(steps),
    model_backend=ModelBackendSpec(
      kind="lightgbm",
      params={"n_estimators": N_ESTIMATORS, "verbosity": -1},
    ),
    n_samples=250_000,
    random_seed=FIT_SEED,
  )


def _coefficient_parent_columns(
  steps: list[GenerationStep], target_columns: list[str]
) -> tuple[str, ...]:
  columns = [AGE_GROUP_COLUMN, *target_columns]
  return tuple(dict.fromkeys(columns))


def _coefficient_sum_spec(
  coefficients: dict[str, float],
  frame: pl.DataFrame,
  parent_columns: tuple[str, ...],
) -> CoefficientSumSpec:
  terms: list[CoefficientTerm] = []
  for key, coefficient in coefficients.items():
    if key == "intercept":
      continue
    column, value = key.rsplit("_c_", 1)
    if column not in parent_columns:
      raise ValueError(
        f"coefficient key '{key}' names column '{column}', which is not a "
        "generated covariate"
      )
    if value == REFERENCE_VALUE:
      raise ValueError(
        f"coefficient key '{key}' uses the reference value "
        f"'{REFERENCE_VALUE}'; choose a sentinel that is not a real level"
      )
    terms.append(
      CoefficientTerm(
        feature_name=column,
        coefficient=float(coefficient),
        value=value,
      )
    )

  return CoefficientSumSpec(
    intercept=float(coefficients["intercept"]),
    reference_value=REFERENCE_VALUE,
    terms=tuple(terms),
  )


def _require_no_nulls(frame: pl.DataFrame, columns: list[str]) -> None:
  for column in [AGE_COLUMN, *columns]:
    if frame[column].null_count() > 0:
      raise ValueError(
        f"column '{column}' contains nulls after the fitting sample; "
        "the bundle cannot be fit from it"
      )


def write_fit_info(
  output_dir: Path,
  frame: pl.DataFrame,
  target_columns: list[str],
  package_version: str,
) -> Path:
  """Write a short, path-free record of the fit."""
  path = output_dir / "FIT_INFO.txt"
  lines = [
    "Synthetic CAN bundle fit record",
    "",
    f"source_file: {SOURCE_FILENAME}",
    "source_year: 2014",
    f"source_row_count: {_source_row_count(frame)}",
    f"fitting_row_count: {frame.height}",
    f"sampling_fraction: {FIT_FRACTION}",
    f"sampling_seed: {FIT_SEED}",
    f"tabdat_synth_version: {package_version}",
    f"model_backend: lightgbm (n_estimators={N_ESTIMATORS}, verbosity=-1)",
    "",
    "column_order:",
  ]
  lines.extend(
    f"  {index + 1:2d}. {column}"
    for index, column in enumerate([AGE_COLUMN, AGE_GROUP_COLUMN, *target_columns])
  )
  path.write_text("\n".join(lines) + "\n")
  return path


def _source_row_count(frame: pl.DataFrame) -> int:
  # The fitting sample is a fixed fraction of the extract, so the source count
  # is recovered without keeping a second copy of the full table in memory.
  return round(frame.height / FIT_FRACTION)


def rebuild(va_dir: Path | None = None, output_dir: Path | None = None) -> Path:
  """Fit the CAN bundle from the source extract and save it to disk."""
  from tabdat_synth import __version__ as package_version

  resolved_va_dir = va_dir if va_dir is not None else resolve_va_dir()
  resolved_output = (
    output_dir
    if output_dir is not None
    else Path(__file__).resolve().parents[2] / "bundles" / "can2014"
  )

  frame, target_columns = load_fitting_sample(resolved_va_dir)
  spec = build_spec(frame, target_columns)
  fit_target = replace(
    spec,
    source_path=resolved_va_dir / SOURCE_FILENAME,
    source_format="parquet",
  )
  fitted = fit_synthesizer(fit_target)
  # Persist the placeholder path, never the author's private extract location.
  portable = replace(fitted, spec=replace(fitted.spec, source_path=Path(SOURCE_PLACEHOLDER)))
  save_fitted_synthesizer(portable, resolved_output)
  write_fit_info(resolved_output, frame, target_columns, package_version)
  return resolved_output
