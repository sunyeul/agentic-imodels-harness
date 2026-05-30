from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import traceback
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from toy_imodels.core.candidate import BaseCandidateModel
from toy_imodels.core.project import Project
from toy_imodels.interpretability import (
    audit_interpretability_judgment,
    evaluate_interpretability,
    validate_interpretability_judgment,
    write_interpretability_packet,
)
from toy_imodels.leaderboard import append_result, read_leaderboard, update_result
from toy_imodels.reports import (
    apply_interpretability_judgment_to_report,
    write_run_report,
)
from toy_imodels.spec import EvaluationSpec


class CandidateContractError(RuntimeError):
    """Raised when the editable candidate model does not match the harness API."""


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


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResult:
    run_id: str
    timestamp: str
    project_id: str
    candidate_module: str
    model_name: str
    notes: str
    spec_name: str
    primary_metric: str
    primary_metric_direction: str
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_mae_mean: float
    cv_mae_std: float
    cv_r2_mean: float
    cv_r2_std: float
    cv_strategy_name: str
    cv_n_splits: int
    cv_random_state: int | None
    interpretability_score: float
    status: Literal["success"]
    submission_path: str
    report_path: str
    fold_metrics_path: str
    run_metadata_path: str
    candidate_snapshot_path: str
    residual_diagnostics_path: str
    metrics: dict[str, float]
    valid_rmse: float | None = None
    valid_mae: float | None = None
    valid_r2: float | None = None
    error_traceback_path: str = ""
    error: str = ""

    def to_leaderboard_row(
        self,
        *,
        spec_metadata: dict[str, object],
        leaderboard_metrics: dict[str, float],
    ) -> dict[str, object]:
        row = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "metrics"
        }
        row.update(spec_metadata)
        row.update(leaderboard_metrics)
        return row


def _load_candidate_class(
    module_name: str, import_path: list[Path] | None = None
) -> type[BaseCandidateModel]:
    _reject_candidate_leakage_source(module_name, import_path=import_path)
    added_paths: list[str] = []
    if import_path:
        for path in import_path:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                added_paths.append(path_str)
    try:
        importlib.invalidate_caches()
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
    finally:
        for path_str in added_paths:
            if path_str in sys.path:
                sys.path.remove(path_str)

    if not hasattr(module, "CandidateModel"):
        raise CandidateContractError(f"{module_name} must define CandidateModel")
    candidate_class: Any = module.CandidateModel
    return candidate_class


def _candidate_source_before_import(
    module_name: str, import_path: list[Path] | None = None
) -> str:
    added_paths: list[str] = []
    if import_path:
        for path in import_path:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
                added_paths.append(path_str)
    try:
        spec = importlib.util.find_spec(module_name)
    finally:
        for path_str in added_paths:
            if path_str in sys.path:
                sys.path.remove(path_str)

    if spec is None or spec.origin is None:
        return ""
    try:
        return Path(spec.origin).read_text()
    except OSError:
        return ""


def _reject_candidate_leakage_source(
    module_name: str, import_path: list[Path] | None = None
) -> None:
    source = _candidate_source_before_import(module_name, import_path=import_path)
    forbidden_hits = [
        fragment
        for fragment in FORBIDDEN_CANDIDATE_SOURCE_FRAGMENTS
        if fragment in source
    ]
    if forbidden_hits:
        raise CandidateContractError(
            "CandidateModel must not reference competition data files, "
            "dataset loaders, or benchmark metadata. Forbidden fragments: "
            + ", ".join(forbidden_hits)
        )


def _make_candidate(candidate_class: type[BaseCandidateModel]) -> BaseCandidateModel:
    try:
        candidate = candidate_class()
    except TypeError as exc:
        raise CandidateContractError(f"CandidateModel contract error: {exc}") from exc
    if not isinstance(candidate, BaseCandidateModel):
        raise CandidateContractError(
            "CandidateModel must inherit from toy_imodels.core.candidate."
            "BaseCandidateModel"
        )
    for method_name in ("fit", "predict", "__str__"):
        if not hasattr(candidate, method_name):
            raise CandidateContractError(f"CandidateModel must define {method_name}")
    return candidate


def _cross_validate_candidate(
    candidate_class: type[BaseCandidateModel],
    x_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    *,
    evaluation_spec: EvaluationSpec,
) -> tuple[dict[str, float], list[dict[str, object]], pd.Series]:
    fold_scores: list[dict[str, float]] = []
    fold_records: list[dict[str, object]] = []
    oof_predictions = pd.Series(index=x_labeled.index, dtype=float)

    splitter = evaluation_spec.cv_strategy.splitter
    for fold_index, (train_index, valid_index) in enumerate(
        splitter.split(x_labeled, y_labeled)
    ):
        candidate = _make_candidate(candidate_class)
        x_train_fold = x_labeled.iloc[train_index]
        y_train_fold = y_labeled.iloc[train_index]
        x_valid_fold = x_labeled.iloc[valid_index]
        y_valid_fold = y_labeled.iloc[valid_index]

        candidate.fit(x_train_fold, y_train_fold)
        fold_pred = candidate.predict(x_valid_fold)
        oof_predictions.iloc[valid_index] = fold_pred
        fold_metrics = evaluation_spec.score_predictions(y_valid_fold, fold_pred)
        fold_scores.append(fold_metrics)
        fold_records.append(
            {
                "fold_index": fold_index,
                "train_rows": int(len(train_index)),
                "valid_rows": int(len(valid_index)),
                "metrics": fold_metrics,
            }
        )

    return (
        evaluation_spec.aggregate_fold_scores(fold_scores),
        fold_records,
        oof_predictions,
    )


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_dir(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "runs" / run_id


def _write_json_artifact(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _candidate_source(
    candidate_class: type[BaseCandidateModel], candidate_module: str
) -> tuple[str, str]:
    module = sys.modules.get(candidate_class.__module__) or sys.modules.get(
        candidate_module
    )
    source = ""
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if module_file:
            try:
                source = Path(module_file).read_text()
            except OSError:
                source = ""
        if not source:
            try:
                source = inspect.getsource(module)
            except (OSError, TypeError):
                source = ""

    if not source:
        return (
            f"# Candidate source unavailable for module {candidate_module}.\n",
            "",
        )
    return source, hashlib.sha256(source.encode()).hexdigest()


def _write_candidate_snapshot(
    run_dir: Path,
    *,
    candidate_class: type[BaseCandidateModel],
    candidate_module: str,
) -> tuple[Path, str]:
    source, source_hash = _candidate_source(candidate_class, candidate_module)
    snapshot_path = run_dir / "candidate_snapshot.py"
    snapshot_path.write_text(source)
    return snapshot_path, source_hash


def _write_fold_metrics(
    run_dir: Path,
    *,
    run_id: str,
    project_id: str,
    fold_records: list[dict[str, object]],
) -> Path:
    return _write_json_artifact(
        run_dir / "fold_metrics.json",
        {
            "run_id": run_id,
            "project_id": project_id,
            "folds": fold_records,
        },
    )


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


def _write_residual_diagnostics(
    run_dir: Path,
    *,
    run_id: str,
    project_id: str,
    x_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    oof_predictions: pd.Series,
) -> Path:
    residuals = y_labeled - oof_predictions
    feature_records = []
    for feature_name in x_labeled.columns:
        feature = cast(pd.Series, x_labeled[feature_name])
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
    return _write_json_artifact(
        run_dir / "residual_diagnostics.json",
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
    )


def _write_run_metadata(
    run_dir: Path,
    *,
    run_id: str,
    timestamp: str,
    project_id: str,
    candidate_module: str,
    candidate_source_sha256: str,
    spec_metadata: dict[str, object],
    report_path: Path,
    submission_path: Path,
    fold_metrics_path: Path,
    residual_diagnostics_path: Path,
    candidate_snapshot_path: Path,
) -> Path:
    return _write_json_artifact(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "project_id": project_id,
            "candidate_module": candidate_module,
            "candidate_source_sha256": candidate_source_sha256,
            "spec": {
                "spec_name": spec_metadata["spec_name"],
                "primary_metric": spec_metadata["primary_metric"],
                "primary_metric_direction": spec_metadata["primary_metric_direction"],
                "cv_strategy_name": spec_metadata["cv_strategy_name"],
                "cv_n_splits": spec_metadata["cv_n_splits"],
                "cv_random_state": spec_metadata["cv_random_state"],
                "cv_description": spec_metadata["cv_description"],
            },
            "artifacts": {
                "report_path": str(report_path),
                "submission_path": str(submission_path),
                "fold_metrics_path": str(fold_metrics_path),
                "residual_diagnostics_path": str(residual_diagnostics_path),
                "candidate_snapshot_path": str(candidate_snapshot_path),
            },
        },
    )


def _write_error_traceback(run_dir: Path) -> Path:
    traceback_path = run_dir / "error_traceback.txt"
    traceback_path.write_text(traceback.format_exc())
    return traceback_path


def run_experiment(
    *,
    project: Project,
    candidate_module: str | None = None,
    run_id: str | None = None,
    import_path: list[Path] | None = None,
) -> EvaluationResult:
    run_id = run_id or _make_run_id()
    candidate_module = candidate_module or project.default_candidate_module
    evaluation_spec = project.spec
    spec_metadata = evaluation_spec.report_metadata()
    cv_n_splits = cast(int, spec_metadata["cv_n_splits"])
    cv_random_state = cast(int | None, spec_metadata["cv_random_state"])
    timestamp = datetime.now(timezone.utc).isoformat()
    results_path = Path(project.results_dir)
    submissions_dir = results_path / "submissions"
    runs_dir = results_path / "runs" / run_id
    submissions_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    try:
        candidate_class = _load_candidate_class(
            candidate_module, import_path=import_path
        )
        competition_data = project.load_data(project.data_dir)
        x_labeled = pd.concat(
            [competition_data.x_train, competition_data.x_valid], ignore_index=True
        )
        y_labeled = pd.concat(
            [competition_data.y_train, competition_data.y_valid], ignore_index=True
        )
        cv_metrics, fold_records, oof_predictions = _cross_validate_candidate(
            candidate_class, x_labeled, y_labeled, evaluation_spec=evaluation_spec
        )
        result_metrics = evaluation_spec.result_metrics(cv_metrics)
        leaderboard_metrics = evaluation_spec.leaderboard_metrics(cv_metrics)
        candidate_snapshot_path, candidate_source_sha256 = _write_candidate_snapshot(
            runs_dir,
            candidate_class=candidate_class,
            candidate_module=candidate_module,
        )
        fold_metrics_path = _write_fold_metrics(
            runs_dir,
            run_id=run_id,
            project_id=project.project_id,
            fold_records=fold_records,
        )
        residual_diagnostics_path = _write_residual_diagnostics(
            runs_dir,
            run_id=run_id,
            project_id=project.project_id,
            x_labeled=x_labeled,
            y_labeled=y_labeled,
            oof_predictions=oof_predictions,
        )

        candidate = _make_candidate(candidate_class)
        candidate.fit(x_labeled, y_labeled)
        test_pred = candidate.predict(
            competition_data.x_test[competition_data.feature_columns]
        )
        model_string = str(candidate)

        interpretability = evaluate_interpretability(model_string)
        interp_score = interpretability.score
        write_interpretability_packet(
            runs_dir,
            run_id=run_id,
            model_string=model_string,
            project_id=project.project_id,
        )

        submission_path = submissions_dir / f"{run_id}.csv"
        pd.DataFrame(
            {"id": competition_data.x_test["id"].to_numpy(), "prediction": test_pred}
        ).to_csv(submission_path, index=False)
        planned_run_metadata_path = runs_dir / "run_metadata.json"
        report_path = write_run_report(
            runs_dir,
            run_id=run_id,
            project_id=project.project_id,
            candidate_module=candidate_module,
            model_name=getattr(candidate, "model_name", candidate.__class__.__name__),
            notes=getattr(candidate, "notes", ""),
            spec_name=str(spec_metadata["spec_name"]),
            primary_metric=str(spec_metadata["primary_metric"]),
            primary_metric_direction=str(spec_metadata["primary_metric_direction"]),
            cv_strategy_name=str(spec_metadata["cv_strategy_name"]),
            cv_n_splits=cv_n_splits,
            cv_random_state=cv_random_state,
            cv_description=str(spec_metadata["cv_description"]),
            metric_lines=evaluation_spec.report_metric_lines(cv_metrics),
            interpretability_score=interp_score,
            model_string=model_string,
            fold_metrics_path=fold_metrics_path,
            residual_diagnostics_path=residual_diagnostics_path,
            run_metadata_path=planned_run_metadata_path,
            candidate_snapshot_path=candidate_snapshot_path,
            next_candidate_path=f"{candidate_module.replace('.', '/')}.py",
        )
        run_metadata_path = _write_run_metadata(
            runs_dir,
            run_id=run_id,
            timestamp=timestamp,
            project_id=project.project_id,
            candidate_module=candidate_module,
            candidate_source_sha256=candidate_source_sha256,
            spec_metadata=spec_metadata,
            report_path=report_path,
            submission_path=submission_path,
            fold_metrics_path=fold_metrics_path,
            residual_diagnostics_path=residual_diagnostics_path,
            candidate_snapshot_path=candidate_snapshot_path,
        )

        result = EvaluationResult(
            run_id=run_id,
            timestamp=timestamp,
            project_id=project.project_id,
            candidate_module=candidate_module,
            model_name=getattr(candidate, "model_name", candidate.__class__.__name__),
            notes=getattr(candidate, "notes", ""),
            spec_name=str(spec_metadata["spec_name"]),
            primary_metric=str(spec_metadata["primary_metric"]),
            primary_metric_direction=str(spec_metadata["primary_metric_direction"]),
            cv_rmse_mean=result_metrics["cv_rmse_mean"],
            cv_rmse_std=result_metrics["cv_rmse_std"],
            cv_mae_mean=result_metrics["cv_mae_mean"],
            cv_mae_std=result_metrics["cv_mae_std"],
            cv_r2_mean=result_metrics["cv_r2_mean"],
            cv_r2_std=result_metrics["cv_r2_std"],
            cv_strategy_name=str(spec_metadata["cv_strategy_name"]),
            cv_n_splits=cv_n_splits,
            cv_random_state=cv_random_state,
            interpretability_score=interp_score,
            status="success",
            submission_path=str(submission_path),
            report_path=str(report_path),
            fold_metrics_path=str(fold_metrics_path),
            run_metadata_path=str(run_metadata_path),
            candidate_snapshot_path=str(candidate_snapshot_path),
            residual_diagnostics_path=str(residual_diagnostics_path),
            metrics=cv_metrics,
        )
        append_result(
            results_path,
            result.to_leaderboard_row(
                spec_metadata=spec_metadata,
                leaderboard_metrics=leaderboard_metrics,
            ),
        )
        return result
    except CandidateContractError as exc:
        error_traceback_path = _write_error_traceback(runs_dir)
        append_result(
            results_path,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "project_id": project.project_id,
                "candidate_module": candidate_module,
                "model_name": candidate_module,
                "status": "contract_error",
                "error_traceback_path": str(error_traceback_path),
                "error": str(exc),
            },
        )
        raise
    except Exception as exc:
        error_traceback_path = _write_error_traceback(runs_dir)
        append_result(
            results_path,
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "project_id": project.project_id,
                "candidate_module": candidate_module,
                "model_name": candidate_module,
                "status": "error",
                "error_traceback_path": str(error_traceback_path),
                "error": str(exc),
            },
        )
        raise


def apply_interpretability_judgment(
    *,
    results_dir: str | Path,
    run_id: str,
    judgment_path: str | Path,
    project_id: str | None = None,
) -> Path:
    """Validate a judge artifact and apply it through the fixed harness."""

    source_path = Path(judgment_path)
    judgment = json.loads(source_path.read_text())
    normalized = validate_interpretability_judgment(judgment, expected_run_id=run_id)

    leaderboard = read_leaderboard(results_dir)
    matches = leaderboard["run_id"] == run_id
    if project_id is not None:
        matches = matches & (leaderboard["project_id"] == project_id)
    if not matches.any():
        if project_id is None:
            raise ValueError(f"No leaderboard row found for run_id {run_id}")
        raise ValueError(
            f"No leaderboard row found for project_id {project_id} and run_id {run_id}"
        )
    if int(matches.sum()) != 1:
        if project_id is None:
            raise ValueError(
                f"Expected exactly one leaderboard row for run_id {run_id}"
            )
        raise ValueError(
            "Expected exactly one leaderboard row for "
            f"project_id {project_id} and run_id {run_id}"
        )

    report_path_value = leaderboard.loc[matches, "report_path"].iloc[0]
    if not isinstance(report_path_value, str) or not report_path_value:
        raise ValueError(f"Leaderboard row for run_id {run_id} has no report_path")

    target_run_dir = _run_dir(results_dir, run_id)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    packet_path = target_run_dir / "interpretability_packet.json"
    if not packet_path.exists():
        raise ValueError(f"No interpretability packet found for run_id {run_id}")
    packet = json.loads(packet_path.read_text())

    normalized_path = target_run_dir / "interpretability_judgment.json"
    normalized_path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    audit = audit_interpretability_judgment(packet=packet, judgment=normalized)
    audit_path = target_run_dir / "interpretability_judgment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    update_result(
        results_dir,
        run_id=run_id,
        project_id=project_id,
        updates={
            "interpretability_score": normalized["interpretability_score"],
            "interpretability_rubric_version": normalized["rubric_version"],
            "interpretability_judgment_path": str(normalized_path),
            "interpretability_audit_status": audit["audit_status"],
            "interpretability_audit_path": str(audit_path),
        },
    )
    apply_interpretability_judgment_to_report(
        report_path_value,
        interpretability_score=float(normalized["interpretability_score"]),
        judgment_path=normalized_path,
        audit_path=audit_path,
        audit_status=str(audit["audit_status"]),
        dimension_scores=cast(
            dict[str, dict[str, object]], normalized["dimension_scores"]
        ),
    )
    return normalized_path
