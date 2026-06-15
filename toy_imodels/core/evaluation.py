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
    validate_interpretability_judgment,
    write_interpretability_packet,
)
from toy_imodels.leaderboard import append_result, read_leaderboard, update_result
from toy_imodels.provenance import (
    GitProvenance,
    collect_git_provenance,
    comparable_baseline_run_id,
    update_journal_interpretability_judgment,
    write_experiment_journal,
)
from toy_imodels.reports import (
    apply_interpretability_judgment_to_report,
    write_run_report,
)
from toy_imodels.spec import EvaluationSpec


class CandidateContractError(RuntimeError):
    """Raised when the editable candidate model does not match the harness API."""


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
    cv_strategy_name: str
    cv_n_splits: int
    cv_random_state: int | None
    interpretability_score: float
    status: Literal["success"]
    report_path: str
    fold_metrics_path: str
    run_metadata_path: str
    candidate_snapshot_path: str
    metrics: dict[str, float]
    result_metrics: dict[str, float]
    artifact_paths: dict[str, str]
    loop_run_id: str = ""
    condition: str = ""
    iteration_index: int | None = None
    budget: int | None = None
    agent_model: str = ""
    baseline_commit: str = ""
    baseline_candidate_sha256: str = ""
    candidate_path: str = ""
    agent_input_manifest_path: str = ""
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
            if field.name not in {"metrics", "result_metrics", "artifact_paths"}
        }
        row.update(spec_metadata)
        row.update(self.result_metrics)
        row.update(self.artifact_paths)
        row.update(leaderboard_metrics)
        return row


def _load_candidate_class(
    module_name: str,
    *,
    forbidden_source_fragments: tuple[str, ...],
    import_path: list[Path] | None = None,
) -> type[BaseCandidateModel]:
    _reject_candidate_leakage_source(
        module_name,
        forbidden_source_fragments=forbidden_source_fragments,
        import_path=import_path,
    )
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


def _load_candidate_class_from_path(
    candidate_path: str | Path,
    *,
    forbidden_source_fragments: tuple[str, ...],
) -> tuple[str, type[BaseCandidateModel]]:
    path = Path(candidate_path).resolve()
    source = path.read_text()
    forbidden_hits = [
        fragment for fragment in forbidden_source_fragments if fragment in source
    ]
    if forbidden_hits:
        raise CandidateContractError(
            "CandidateModel must not reference competition data files, "
            "dataset loaders, or benchmark metadata. Forbidden fragments: "
            + ", ".join(forbidden_hits)
        )

    module_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module_name = f"_toy_imodels_loop_candidate_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CandidateContractError(f"Cannot import candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    if not hasattr(module, "CandidateModel"):
        raise CandidateContractError(f"{path} must define CandidateModel")
    candidate_class: Any = module.CandidateModel
    return module_name, candidate_class


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
    module_name: str,
    *,
    forbidden_source_fragments: tuple[str, ...],
    import_path: list[Path] | None = None,
) -> None:
    source = _candidate_source_before_import(module_name, import_path=import_path)
    forbidden_hits = [
        fragment for fragment in forbidden_source_fragments if fragment in source
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


def _write_run_metadata(
    run_dir: Path,
    *,
    run_id: str,
    timestamp: str,
    project_id: str,
    candidate_module: str,
    candidate_source_sha256: str,
    git_provenance: GitProvenance,
    spec_metadata: dict[str, object],
    report_path: Path,
    fold_metrics_path: Path,
    candidate_snapshot_path: Path,
    artifact_paths: dict[str, str],
    loop_metadata: dict[str, object] | None = None,
) -> Path:
    loop_metadata = loop_metadata or {}
    return _write_json_artifact(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "timestamp": timestamp,
            "project_id": project_id,
            "candidate_module": candidate_module,
            "candidate_source_sha256": candidate_source_sha256,
            "git_commit": git_provenance.git_commit,
            "git_parent_commit": git_provenance.git_parent_commit,
            "git_dirty": git_provenance.git_dirty,
            **loop_metadata,
            "spec_name": spec_metadata["spec_name"],
            "primary_metric": spec_metadata["primary_metric"],
            "primary_metric_direction": spec_metadata["primary_metric_direction"],
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
                "fold_metrics_path": str(fold_metrics_path),
                "candidate_snapshot_path": str(candidate_snapshot_path),
                **artifact_paths,
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
    candidate_path: str | Path | None = None,
    run_id: str | None = None,
    import_path: list[Path] | None = None,
    loop_run_id: str | None = None,
    condition: str | None = None,
    iteration_index: int | None = None,
    budget: int | None = None,
    agent_model: str | None = None,
    baseline_commit: str | None = None,
    baseline_candidate_sha256: str | None = None,
    agent_input_manifest_path: str | Path | None = None,
) -> EvaluationResult:
    run_id = run_id or _make_run_id()
    candidate_module = candidate_module or project.default_candidate_module
    candidate_path_value = str(Path(candidate_path).resolve()) if candidate_path else ""
    evaluation_spec = project.spec
    spec_metadata = evaluation_spec.report_metadata()
    cv_n_splits = cast(int, spec_metadata["cv_n_splits"])
    cv_random_state = cast(int | None, spec_metadata["cv_random_state"])
    timestamp = datetime.now(timezone.utc).isoformat()
    git_provenance = collect_git_provenance()
    results_path = Path(project.results_dir)
    submissions_dir = results_path / "submissions"
    runs_dir = results_path / "runs" / run_id
    submissions_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if candidate_path is not None:
            candidate_module, candidate_class = _load_candidate_class_from_path(
                candidate_path,
                forbidden_source_fragments=project.forbidden_candidate_source_fragments,
            )
        else:
            candidate_class = _load_candidate_class(
                candidate_module,
                forbidden_source_fragments=project.forbidden_candidate_source_fragments,
                import_path=import_path,
            )
        evaluation_data = project.load_data(project.data_dir)
        cv_metrics, fold_records, oof_predictions = _cross_validate_candidate(
            candidate_class,
            evaluation_data.x_labeled,
            evaluation_data.y_labeled,
            evaluation_spec=evaluation_spec,
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
        artifact_paths: dict[str, str] = {}
        diagnostics_path = None
        if project.write_diagnostics is not None:
            diagnostics_path = project.write_diagnostics(
                runs_dir,
                run_id,
                project.project_id,
                evaluation_data,
                oof_predictions,
            )
            if diagnostics_path is not None:
                artifact_paths["diagnostics_path"] = str(diagnostics_path)

        candidate = _make_candidate(candidate_class)
        candidate.fit(evaluation_data.x_labeled, evaluation_data.y_labeled)
        test_pred = candidate.predict(
            evaluation_data.x_test[evaluation_data.feature_columns]
        )
        model_string = str(candidate)

        interp_score = float("nan")
        write_interpretability_packet(
            runs_dir,
            run_id=run_id,
            model_string=model_string,
            project_id=project.project_id,
        )

        submission_path = project.write_submission(
            submissions_dir, run_id, evaluation_data, test_pred
        )
        artifact_paths["submission_path"] = str(submission_path)
        loop_metadata = {
            "loop_run_id": loop_run_id or "",
            "condition": condition or "",
            "iteration_index": iteration_index,
            "budget": budget,
            "agent_model": agent_model or "",
            "baseline_commit": baseline_commit or "",
            "baseline_candidate_sha256": baseline_candidate_sha256 or "",
            "candidate_path": candidate_path_value,
            "agent_input_manifest_path": (
                str(Path(agent_input_manifest_path).resolve())
                if agent_input_manifest_path
                else ""
            ),
        }
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
            model_string=model_string,
            fold_metrics_path=fold_metrics_path,
            diagnostics_path=diagnostics_path,
            run_metadata_path=planned_run_metadata_path,
            candidate_snapshot_path=candidate_snapshot_path,
            next_candidate_path=(
                candidate_path_value
                if candidate_path_value
                else f"{candidate_module.replace('.', '/')}.py"
            ),
        )
        run_metadata_path = _write_run_metadata(
            runs_dir,
            run_id=run_id,
            timestamp=timestamp,
            project_id=project.project_id,
            candidate_module=candidate_module,
            candidate_source_sha256=candidate_source_sha256,
            git_provenance=git_provenance,
            spec_metadata=spec_metadata,
            report_path=report_path,
            fold_metrics_path=fold_metrics_path,
            candidate_snapshot_path=candidate_snapshot_path,
            artifact_paths=artifact_paths,
            loop_metadata=loop_metadata,
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
            cv_strategy_name=str(spec_metadata["cv_strategy_name"]),
            cv_n_splits=cv_n_splits,
            cv_random_state=cv_random_state,
            interpretability_score=interp_score,
            status="success",
            report_path=str(report_path),
            fold_metrics_path=str(fold_metrics_path),
            run_metadata_path=str(run_metadata_path),
            candidate_snapshot_path=str(candidate_snapshot_path),
            loop_run_id=loop_metadata["loop_run_id"],
            condition=loop_metadata["condition"],
            iteration_index=iteration_index,
            budget=budget,
            agent_model=loop_metadata["agent_model"],
            baseline_commit=loop_metadata["baseline_commit"],
            baseline_candidate_sha256=loop_metadata["baseline_candidate_sha256"],
            candidate_path=candidate_path_value,
            agent_input_manifest_path=loop_metadata["agent_input_manifest_path"],
            metrics=cv_metrics,
            result_metrics=result_metrics,
            artifact_paths=artifact_paths,
        )
        append_result(
            results_path,
            result.to_leaderboard_row(
                spec_metadata=spec_metadata,
                leaderboard_metrics=leaderboard_metrics,
            ),
        )
        leaderboard = read_leaderboard(results_path)
        baseline_run_id = comparable_baseline_run_id(
            leaderboard,
            project_id=project.project_id,
            spec_name=str(spec_metadata["spec_name"]),
            primary_metric=str(spec_metadata["primary_metric"]),
            primary_metric_direction=str(spec_metadata["primary_metric_direction"]),
        )
        write_experiment_journal(
            results_dir=results_path,
            run_id=run_id,
            project_id=project.project_id,
            model_name=result.model_name,
            notes=result.notes,
            spec_name=result.spec_name,
            primary_metric=result.primary_metric,
            primary_metric_direction=result.primary_metric_direction,
            primary_metric_value=cv_metrics.get(result.primary_metric, float("nan")),
            candidate_source_sha256=candidate_source_sha256,
            git_provenance=git_provenance,
            report_path=report_path,
            run_metadata_path=run_metadata_path,
            candidate_snapshot_path=candidate_snapshot_path,
            interpretability_status="pending_agent_judgment",
            interpretability_score=interp_score,
            baseline_run_id=baseline_run_id,
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
                "loop_run_id": loop_run_id or "",
                "condition": condition or "",
                "iteration_index": iteration_index,
                "budget": budget,
                "agent_model": agent_model or "",
                "baseline_commit": baseline_commit or "",
                "baseline_candidate_sha256": baseline_candidate_sha256 or "",
                "candidate_path": candidate_path_value,
                "agent_input_manifest_path": (
                    str(Path(agent_input_manifest_path).resolve())
                    if agent_input_manifest_path
                    else ""
                ),
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
                "loop_run_id": loop_run_id or "",
                "condition": condition or "",
                "iteration_index": iteration_index,
                "budget": budget,
                "agent_model": agent_model or "",
                "baseline_commit": baseline_commit or "",
                "baseline_candidate_sha256": baseline_candidate_sha256 or "",
                "candidate_path": candidate_path_value,
                "agent_input_manifest_path": (
                    str(Path(agent_input_manifest_path).resolve())
                    if agent_input_manifest_path
                    else ""
                ),
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
    update_journal_interpretability_judgment(
        results_dir=results_dir,
        run_id=run_id,
        interpretability_score=float(normalized["interpretability_score"]),
        judgment_path=normalized_path,
        audit_path=audit_path,
        audit_status=str(audit["audit_status"]),
    )
    return normalized_path
