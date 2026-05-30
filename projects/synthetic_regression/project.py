from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from projects.synthetic_regression.datasets import (
    DEFAULT_DATA_DIR,
    load_evaluation_data,
)
from projects.synthetic_regression.spec import DefaultEvaluationSpec
from toy_imodels.core.project import EvaluationData, Project
from toy_imodels.spec import EvaluationSpec

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"
PROTECTED_PATHS = (
    "toy_imodels/",
    "projects/synthetic_regression/spec.py",
    "projects/synthetic_regression/datasets.py",
    "projects/synthetic_regression/data/",
)
FORBIDDEN_CANDIDATE_SOURCE_FRAGMENTS = (
    "projects.synthetic_regression.datasets",
    "synthetic_regression.datasets",
    "DEFAULT_DATA_DIR",
    "train.csv",
    "valid.csv",
    "test.csv",
    "sample_submission.csv",
    "metadata.json",
)


def write_submission(
    submissions_dir: Path,
    run_id: str,
    data: EvaluationData,
    predictions: Any,
) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submissions_dir / f"{run_id}.csv"
    pd.DataFrame(
        {"id": data.x_test["id"].to_numpy(), "prediction": predictions}
    ).to_csv(submission_path, index=False)
    return submission_path


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    corr = left.corr(right)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _feature_bin_records(
    feature: pd.Series, residuals: pd.Series, *, n_bins: int = 4
) -> list[dict[str, object]]:
    ranked = feature.rank(method="first")
    bins = cast(pd.Series, pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop"))
    records: list[dict[str, object]] = []
    for bin_index in sorted(bins.dropna().unique()):
        mask = bins == bin_index
        feature_bin = cast(pd.Series, feature[mask])
        residual_bin = cast(pd.Series, residuals[mask])
        records.append(
            {
                "bin_index": int(bin_index),
                "row_count": int(mask.sum()),
                "feature_min": float(feature_bin.min()),
                "feature_max": float(feature_bin.max()),
                "residual_mean": float(residual_bin.mean()),
                "residual_abs_mean": float(residual_bin.abs().mean()),
            }
        )
    return records


def write_residual_diagnostics(
    run_dir: Path,
    run_id: str,
    project_id: str,
    data: EvaluationData,
    oof_predictions: pd.Series,
) -> Path:
    residuals = data.y_labeled - oof_predictions
    feature_records = []
    for feature_name in data.feature_columns:
        feature = cast(pd.Series, data.x_labeled[feature_name])
        bins = _feature_bin_records(feature, residuals)
        max_bin_shift = max(
            (abs(cast(float, item["residual_mean"])) for item in bins),
            default=0.0,
        )
        feature_records.append(
            {
                "feature": feature_name,
                "residual_correlation": _safe_corr(feature, residuals),
                "absolute_residual_correlation": _safe_corr(
                    feature.abs(), residuals.abs()
                ),
                "max_abs_bin_residual_mean": float(max_bin_shift),
                "bins": bins,
            }
        )

    ranked_features = sorted(
        feature_records,
        key=lambda item: (
            abs(cast(float, item["residual_correlation"])),
            cast(float, item["max_abs_bin_residual_mean"]),
        ),
        reverse=True,
    )
    diagnostics_path = run_dir / "residual_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "project_id": project_id,
                "diagnostic_note": (
                    "Diagnostics are computed from out-of-fold residuals only. "
                    "They expose observed error patterns, not hidden benchmark "
                    "oracle structure."
                ),
                "residual_summary": {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "abs_mean": float(residuals.abs().mean()),
                    "p10": float(residuals.quantile(0.10)),
                    "p50": float(residuals.quantile(0.50)),
                    "p90": float(residuals.quantile(0.90)),
                },
                "top_features_by_residual_pattern": ranked_features[:8],
                "feature_diagnostics": feature_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return diagnostics_path


def synthetic_regression_project(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    spec: EvaluationSpec | None = None,
) -> Project:
    return Project(
        project_id="synthetic_regression",
        package_name="projects.synthetic_regression",
        spec=spec or DefaultEvaluationSpec(),
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        load_data=load_evaluation_data,
        protected_paths=PROTECTED_PATHS,
        forbidden_candidate_source_fragments=FORBIDDEN_CANDIDATE_SOURCE_FRAGMENTS,
        write_submission=write_submission,
        write_diagnostics=write_residual_diagnostics,
    )
