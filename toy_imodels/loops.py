from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from toy_imodels.core.evaluation import EvaluationResult, run_experiment
from toy_imodels.core.project import Project
from toy_imodels.provenance import collect_git_provenance, repo_root, sha256_file
from toy_imodels.reports import write_agent_view_report

ConditionName = Literal["blind", "representation"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionSpec:
    name: ConditionName
    allowed_artifacts: tuple[str, ...]
    forbidden_artifacts: tuple[str, ...]
    include_model_string: bool
    include_interpretability_packet: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LoopRunManifest:
    loop_run_id: str
    project_id: str
    condition: ConditionName
    budget: int
    agent_model: str
    baseline_commit: str
    baseline_candidate_sha256: str
    dataset_sha256: str
    spec_sha256: str
    primary_metric: str
    primary_metric_direction: str
    candidate_workspace_path: str


REPRESENTATION_SPEC = ConditionSpec(
    name="representation",
    allowed_artifacts=(
        "leaderboard",
        "report",
        "fold_metrics",
        "run_metadata",
        "error_traceback",
        "model_string",
        "interpretability_packet",
    ),
    forbidden_artifacts=(),
    include_model_string=True,
    include_interpretability_packet=True,
)

BLIND_SPEC = ConditionSpec(
    name="blind",
    allowed_artifacts=(
        "leaderboard",
        "sanitized_report",
        "fold_metrics",
        "run_metadata",
        "error_traceback",
        "score_movement",
    ),
    forbidden_artifacts=(
        "model_string",
        "interpretability_packet",
        "candidate_snapshot",
    ),
    include_model_string=False,
    include_interpretability_packet=False,
)

CONDITION_SPECS = {
    REPRESENTATION_SPEC.name: REPRESENTATION_SPEC,
    BLIND_SPEC.name: BLIND_SPEC,
}


def condition_spec(name: str | None = None) -> ConditionSpec:
    condition_name = name or REPRESENTATION_SPEC.name
    if condition_name not in CONDITION_SPECS:
        raise ValueError(f"Unknown condition: {condition_name}")
    return CONDITION_SPECS[condition_name]


def init_loop_run(
    project: Project,
    *,
    condition: str = REPRESENTATION_SPEC.name,
    budget: int,
    agent_model: str,
    loop_run_id: str | None = None,
) -> LoopRunManifest:
    spec = condition_spec(condition)
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if not agent_model.strip():
        raise ValueError("agent_model must not be empty")

    loop_run_id = loop_run_id or _make_loop_run_id(spec.name)
    loop_dir = loop_run_dir(project.results_dir, loop_run_id)
    workspace_dir = loop_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    candidate_source = (
        repo_root() / f"{project.default_candidate_module.replace('.', '/')}.py"
    )
    candidate_workspace_path = workspace_dir / "candidate_model.py"
    shutil.copy2(candidate_source, candidate_workspace_path)

    spec_metadata = project.spec.report_metadata()
    manifest = LoopRunManifest(
        loop_run_id=loop_run_id,
        project_id=project.project_id,
        condition=spec.name,
        budget=budget,
        agent_model=agent_model,
        baseline_commit=collect_git_provenance().git_commit,
        baseline_candidate_sha256=sha256_file(candidate_source),
        dataset_sha256=_hash_directory(project.data_dir),
        spec_sha256=_hash_spec(project),
        primary_metric=str(spec_metadata["primary_metric"]),
        primary_metric_direction=str(spec_metadata["primary_metric_direction"]),
        candidate_workspace_path=str(candidate_workspace_path.resolve()),
    )
    write_loop_manifest(project.results_dir, manifest)
    _ensure_iterations(project.results_dir, loop_run_id)
    return manifest


def prepare_iteration(
    results_dir: str | Path,
    loop_run_id: str,
    *,
    iteration_index: int,
) -> Path:
    manifest = load_loop_manifest(results_dir, loop_run_id)
    spec = condition_spec(manifest.condition)
    if iteration_index < 1 or iteration_index > manifest.budget:
        raise ValueError("iteration_index must be within the loop budget")

    bundle_dir = (
        loop_run_dir(results_dir, loop_run_id)
        / "agent_input_bundle"
        / f"iteration_{iteration_index}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, str]] = []

    leaderboard_path = Path(results_dir) / "leaderboard.csv"
    if leaderboard_path.exists():
        leaderboard_bundle_path = bundle_dir / "leaderboard.csv"
        _write_loop_leaderboard_slice(
            leaderboard_path,
            leaderboard_bundle_path,
            loop_run_id,
            condition_spec=spec,
        )
        files.append(_file_record("leaderboard", leaderboard_bundle_path))

    previous = _previous_iteration_row(results_dir, loop_run_id, iteration_index)
    if previous is not None:
        run_dir = Path(results_dir) / "runs" / str(previous["evaluation_run_id"])
        report_path = run_dir / "report.md"
        if report_path.exists():
            agent_report = bundle_dir / "report.md"
            write_agent_view_report(
                report_path,
                agent_report,
                include_model_string=spec.include_model_string,
            )
            files.append(
                _file_record(
                    "report" if spec.include_model_string else "sanitized_report",
                    agent_report,
                )
            )
        for artifact_name, filename in (
            ("fold_metrics", "fold_metrics.json"),
            ("run_metadata", "run_metadata.json"),
            ("error_traceback", "error_traceback.txt"),
        ):
            source = run_dir / filename
            if source.exists():
                target = bundle_dir / filename
                if artifact_name == "run_metadata" and spec.name == "blind":
                    _write_blind_run_metadata(source, target)
                else:
                    shutil.copy2(source, target)
                files.append(_file_record(artifact_name, target))
        packet_path = run_dir / "interpretability_packet.json"
        if spec.include_interpretability_packet and packet_path.exists():
            target = bundle_dir / "interpretability_packet.json"
            shutil.copy2(packet_path, target)
            files.append(_file_record("interpretability_packet", target))

    manifest_path = bundle_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "loop_run_id": loop_run_id,
                "condition": manifest.condition,
                "iteration_index": iteration_index,
                "allowed_artifacts": list(spec.allowed_artifacts),
                "forbidden_artifacts": list(spec.forbidden_artifacts),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest_path


def record_iteration(
    project: Project,
    loop_run_id: str,
    *,
    iteration_index: int,
    run_id: str | None = None,
) -> EvaluationResult:
    manifest = load_loop_manifest(project.results_dir, loop_run_id)
    if iteration_index < 1 or iteration_index > manifest.budget:
        raise ValueError("iteration_index must be within the loop budget")
    input_manifest_path = (
        loop_run_dir(project.results_dir, loop_run_id)
        / "agent_input_bundle"
        / f"iteration_{iteration_index}"
        / "input_manifest.json"
    )
    if not input_manifest_path.exists():
        input_manifest_path = prepare_iteration(
            project.results_dir,
            loop_run_id,
            iteration_index=iteration_index,
        )
    run_id = run_id or _make_evaluation_run_id(loop_run_id, iteration_index)

    try:
        result = run_experiment(
            project=project,
            candidate_path=manifest.candidate_workspace_path,
            run_id=run_id,
            loop_run_id=loop_run_id,
            condition=manifest.condition,
            iteration_index=iteration_index,
            budget=manifest.budget,
            agent_model=manifest.agent_model,
            baseline_commit=manifest.baseline_commit,
            baseline_candidate_sha256=manifest.baseline_candidate_sha256,
            agent_input_manifest_path=input_manifest_path,
        )
    except Exception:
        _append_failed_iteration(
            project.results_dir,
            manifest,
            run_id,
            iteration_index,
            input_manifest_path,
        )
        raise
    _append_iteration(project.results_dir, manifest, result, iteration_index)
    return result


def verify_loop_run(results_dir: str | Path, loop_run_id: str) -> list[str]:
    findings: list[str] = []
    manifest_path = loop_manifest_path(results_dir, loop_run_id)
    if not manifest_path.exists():
        return [f"FAIL manifest: missing {manifest_path}"]
    manifest = load_loop_manifest(results_dir, loop_run_id)
    try:
        spec = condition_spec(manifest.condition)
    except ValueError as exc:
        return [f"FAIL condition: {exc}"]
    findings.append(f"PASS manifest: {manifest_path}")

    candidate_path = Path(manifest.candidate_workspace_path)
    if candidate_path.exists():
        findings.append(f"PASS candidate_workspace: {candidate_path}")
    else:
        findings.append(f"FAIL candidate_workspace: missing {candidate_path}")

    iterations = read_iterations(results_dir, loop_run_id)
    for row in iterations.to_dict("records"):
        iteration = int(row["iteration_index"])
        input_manifest = (
            loop_run_dir(results_dir, loop_run_id)
            / "agent_input_bundle"
            / f"iteration_{iteration}"
            / "input_manifest.json"
        )
        findings.extend(_verify_input_manifest(input_manifest, spec))
    return findings


def compare_loop_runs(
    results_dir: str | Path,
    left_loop_run_id: str,
    right_loop_run_id: str,
    *,
    target: float | None = None,
) -> dict[str, object]:
    left_manifest = load_loop_manifest(results_dir, left_loop_run_id)
    right_manifest = load_loop_manifest(results_dir, right_loop_run_id)
    mismatches = _compatibility_mismatches(left_manifest, right_manifest)
    if mismatches:
        raise ValueError("Incompatible loop runs: " + "; ".join(mismatches))
    return {
        "left": _loop_summary(results_dir, left_manifest, target=target),
        "right": _loop_summary(results_dir, right_manifest, target=target),
    }


def write_loop_manifest(results_dir: str | Path, manifest: LoopRunManifest) -> Path:
    path = loop_manifest_path(results_dir, manifest.loop_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return path


def load_loop_manifest(results_dir: str | Path, loop_run_id: str) -> LoopRunManifest:
    payload = json.loads(loop_manifest_path(results_dir, loop_run_id).read_text())
    return LoopRunManifest(**payload)


def loop_manifest_path(results_dir: str | Path, loop_run_id: str) -> Path:
    return loop_run_dir(results_dir, loop_run_id) / "loop_manifest.json"


def loop_run_dir(results_dir: str | Path, loop_run_id: str) -> Path:
    return Path(results_dir) / "loop_runs" / loop_run_id


def read_iterations(results_dir: str | Path, loop_run_id: str) -> pd.DataFrame:
    path = _iterations_path(results_dir, loop_run_id)
    if not path.exists():
        return pd.DataFrame(columns=_ITERATION_COLUMNS)
    return pd.read_csv(path)


_ITERATION_COLUMNS = [
    "iteration_index",
    "evaluation_run_id",
    "status",
    "primary_metric",
    "primary_metric_direction",
    "primary_metric_value",
    "candidate_sha256",
    "agent_input_manifest_path",
    "retained",
]


def _make_loop_run_id(condition: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{condition}"


def _make_evaluation_run_id(loop_run_id: str, iteration_index: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{loop_run_id}_iter{iteration_index}_{stamp}"


def _ensure_iterations(results_dir: str | Path, loop_run_id: str) -> Path:
    path = _iterations_path(results_dir, loop_run_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_ITERATION_COLUMNS).to_csv(path, index=False)
    return path


def _iterations_path(results_dir: str | Path, loop_run_id: str) -> Path:
    return loop_run_dir(results_dir, loop_run_id) / "iterations.csv"


def _append_iteration(
    results_dir: str | Path,
    manifest: LoopRunManifest,
    result: EvaluationResult,
    iteration_index: int,
) -> None:
    path = _ensure_iterations(results_dir, manifest.loop_run_id)
    iterations = pd.read_csv(path)
    new_row = pd.DataFrame(
        [
            {
                "iteration_index": iteration_index,
                "evaluation_run_id": result.run_id,
                "status": result.status,
                "primary_metric": result.primary_metric,
                "primary_metric_direction": result.primary_metric_direction,
                "primary_metric_value": result.result_metrics[result.primary_metric],
                "candidate_sha256": sha256_file(manifest.candidate_workspace_path),
                "agent_input_manifest_path": result.agent_input_manifest_path,
                "retained": False,
            }
        ],
        columns=_ITERATION_COLUMNS,
    )
    iterations = iterations[iterations["iteration_index"] != iteration_index]
    pd.concat([iterations, new_row], ignore_index=True).sort_values(
        "iteration_index"
    ).to_csv(path, index=False)


def _append_failed_iteration(
    results_dir: str | Path,
    manifest: LoopRunManifest,
    run_id: str,
    iteration_index: int,
    input_manifest_path: Path,
) -> None:
    path = _ensure_iterations(results_dir, manifest.loop_run_id)
    iterations = pd.read_csv(path)
    new_row = pd.DataFrame(
        [
            {
                "iteration_index": iteration_index,
                "evaluation_run_id": run_id,
                "status": "error",
                "primary_metric": manifest.primary_metric,
                "primary_metric_direction": manifest.primary_metric_direction,
                "primary_metric_value": float("nan"),
                "candidate_sha256": sha256_file(manifest.candidate_workspace_path),
                "agent_input_manifest_path": str(input_manifest_path.resolve()),
                "retained": False,
            }
        ],
        columns=_ITERATION_COLUMNS,
    )
    iterations = iterations[iterations["iteration_index"] != iteration_index]
    pd.concat([iterations, new_row], ignore_index=True).sort_values(
        "iteration_index"
    ).to_csv(path, index=False)


def _previous_iteration_row(
    results_dir: str | Path, loop_run_id: str, iteration_index: int
) -> dict[str, object] | None:
    iterations = read_iterations(results_dir, loop_run_id)
    previous = iterations[iterations["iteration_index"] < iteration_index]
    if previous.empty:
        return None
    return previous.sort_values("iteration_index").iloc[-1].to_dict()


def _write_loop_leaderboard_slice(
    source_path: Path,
    target_path: Path,
    loop_run_id: str,
    *,
    condition_spec: ConditionSpec,
) -> None:
    leaderboard = pd.read_csv(source_path)
    if "loop_run_id" in leaderboard.columns:
        loop_values = leaderboard["loop_run_id"].fillna("")
        leaderboard = leaderboard[(loop_values == "") | (loop_values == loop_run_id)]
    if condition_spec.name == "blind":
        leaderboard = leaderboard.drop(
            columns=[
                column
                for column in leaderboard.columns
                if "candidate_snapshot" in column
                or "interpretability_packet" in column
            ],
            errors="ignore",
        )
    leaderboard.to_csv(target_path, index=False)


def _write_blind_run_metadata(source: Path, target: Path) -> None:
    payload = json.loads(source.read_text())
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in list(artifacts):
            if "candidate_snapshot" in key or "interpretability_packet" in key:
                del artifacts[key]
    payload.pop("candidate_snapshot_path", None)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _file_record(artifact: str, path: Path) -> dict[str, str]:
    return {
        "artifact": artifact,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def _hash_directory(path: str | Path) -> str:
    root = Path(path)
    chunks = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        chunks.append(f"{file_path.relative_to(root)}:{sha256_file(file_path)}")
    return _sha256_text("\n".join(chunks))


def _hash_spec(project: Project) -> str:
    metadata = project.spec.report_metadata()
    return _sha256_text(json.dumps(metadata, sort_keys=True))


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _verify_input_manifest(path: Path, spec: ConditionSpec) -> list[str]:
    if not path.exists():
        return [f"FAIL input_manifest: missing {path}"]
    payload = json.loads(path.read_text())
    findings = [f"PASS input_manifest: {path}"]
    if payload.get("condition") != spec.name:
        findings.append(
            f"FAIL input_manifest_condition: {payload.get('condition')} != {spec.name}"
        )
    bundle_text = "\n".join(
        file_path.read_text(errors="ignore")
        for file_path in path.parent.rglob("*")
        if file_path.is_file() and file_path.name != "input_manifest.json"
    )
    for forbidden in spec.forbidden_artifacts:
        if forbidden in bundle_text:
            findings.append(f"FAIL forbidden_artifact: found {forbidden}")
    if spec.name == "blind" and "## Model string" in bundle_text:
        findings.append("FAIL forbidden_artifact: found model string section")
    if not any(item.startswith("FAIL") for item in findings):
        findings.append(f"PASS condition_fairness: {spec.name}")
    return findings


def _compatibility_mismatches(
    left: LoopRunManifest, right: LoopRunManifest
) -> list[str]:
    fields = (
        "project_id",
        "budget",
        "agent_model",
        "baseline_commit",
        "baseline_candidate_sha256",
        "dataset_sha256",
        "spec_sha256",
        "primary_metric",
        "primary_metric_direction",
    )
    return [
        f"{field}: {getattr(left, field)} != {getattr(right, field)}"
        for field in fields
        if getattr(left, field) != getattr(right, field)
    ]


def _loop_summary(
    results_dir: str | Path, manifest: LoopRunManifest, *, target: float | None
) -> dict[str, object]:
    iterations = read_iterations(results_dir, manifest.loop_run_id)
    scores = [
        float(value)
        for value in iterations["primary_metric_value"].tolist()
        if not math.isnan(float(value))
    ]
    direction = manifest.primary_metric_direction
    best_score = _best(scores, direction)
    first_score = scores[0] if scores else float("nan")
    return {
        "loop_run_id": manifest.loop_run_id,
        "condition": manifest.condition,
        "best_score_at_budget": best_score,
        "gain_per_iteration": _gain_per_iteration(
            first_score,
            best_score,
            len(scores),
            direction,
        ),
        "regression_rate": _regression_rate(scores, direction),
        "failed_edit_rate": _failed_edit_rate(iterations),
        "iterations_to_target": _iterations_to_target(scores, direction, target),
    }


def _best(scores: list[float], direction: str) -> float:
    if not scores:
        return float("nan")
    if direction == "maximize":
        return max(scores)
    return min(scores)


def _gain_per_iteration(
    first_score: float, best_score: float, count: int, direction: str
) -> float:
    if count == 0 or math.isnan(first_score) or math.isnan(best_score):
        return float("nan")
    gain = (
        best_score - first_score
        if direction == "maximize"
        else first_score - best_score
    )
    return gain / count


def _regression_rate(scores: list[float], direction: str) -> float:
    if len(scores) <= 1:
        return 0.0
    regressions = 0
    best_so_far = scores[0]
    for score in scores[1:]:
        if direction == "maximize":
            regressions += int(score < best_so_far)
            best_so_far = max(best_so_far, score)
        else:
            regressions += int(score > best_so_far)
            best_so_far = min(best_so_far, score)
    return regressions / (len(scores) - 1)


def _failed_edit_rate(iterations: pd.DataFrame) -> float:
    if iterations.empty:
        return 0.0
    return float((iterations["status"] != "success").mean())


def _iterations_to_target(
    scores: list[float], direction: str, target: float | None
) -> int | None:
    if target is None:
        return None
    for index, score in enumerate(scores, start=1):
        if direction == "maximize" and score >= target:
            return index
        if direction != "maximize" and score <= target:
            return index
    return None
