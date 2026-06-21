from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict, cast

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
    experiment_name: str = ""
    experiment_id: str = ""


class LoopSummary(TypedDict):
    loop_run_id: str
    condition: ConditionName
    best_score_at_budget: float
    gain_per_iteration: float
    regression_rate: float
    failed_edit_rate: float
    iterations_to_target: int | None


class LoopComparison(TypedDict):
    left: LoopSummary
    right: LoopSummary


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

HANDOFF_PROMPT_FILENAME = "design_handoff_prompt.md"
PRE_DESIGN_RATIONALE_FILENAME = "pre_design_rationale.md"


def condition_spec(name: str | None = None) -> ConditionSpec:
    condition_name = name or REPRESENTATION_SPEC.name
    if condition_name not in CONDITION_SPECS:
        raise ValueError(f"Unknown condition: {condition_name}")
    return CONDITION_SPECS[cast(ConditionName, condition_name)]


def init_loop_run(
    project: Project,
    *,
    condition: str = REPRESENTATION_SPEC.name,
    budget: int,
    agent_model: str,
    loop_run_id: str | None = None,
    experiment_name: str = "",
    experiment_id: str = "",
) -> LoopRunManifest:
    spec = condition_spec(condition)
    if bool(experiment_name) != bool(experiment_id):
        raise ValueError("experiment_name and experiment_id must be provided together")
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
        experiment_name=experiment_name,
        experiment_id=experiment_id,
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

    candidate_workspace_path = Path(manifest.candidate_workspace_path)
    editable_files = [_file_record("candidate_workspace", candidate_workspace_path)]
    pre_design_rationale_path = bundle_dir / PRE_DESIGN_RATIONALE_FILENAME
    _write_pre_design_rationale_template(
        pre_design_rationale_path,
        manifest=manifest,
        spec=spec,
        iteration_index=iteration_index,
    )
    manifest_path = bundle_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "loop_run_id": loop_run_id,
                "condition": manifest.condition,
                "experiment_name": manifest.experiment_name,
                "experiment_id": manifest.experiment_id,
                "iteration_index": iteration_index,
                "candidate_workspace_path": str(candidate_workspace_path.resolve()),
                "pre_design_rationale_path": str(pre_design_rationale_path.resolve()),
                "editable_files": editable_files,
                "writable_artifacts": [
                    _file_record("pre_design_rationale", pre_design_rationale_path)
                ],
                "allowed_artifacts": list(spec.allowed_artifacts),
                "forbidden_artifacts": list(spec.forbidden_artifacts),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _write_design_handoff_prompt(
        bundle_dir / HANDOFF_PROMPT_FILENAME,
        manifest=manifest,
        spec=spec,
        input_manifest_path=manifest_path,
    )
    return manifest_path


def handoff_prompt_path(
    results_dir: str | Path,
    loop_run_id: str,
    *,
    iteration_index: int,
) -> Path:
    path = (
        loop_run_dir(results_dir, loop_run_id)
        / "agent_input_bundle"
        / f"iteration_{iteration_index}"
        / HANDOFF_PROMPT_FILENAME
    )
    if not path.exists():
        prepare_iteration(
            results_dir,
            loop_run_id,
            iteration_index=iteration_index,
        )
    return path


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
            experiment_name=manifest.experiment_name,
            experiment_id=manifest.experiment_id,
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
    findings.extend(_verify_iteration_progress(results_dir, manifest, iterations))
    return findings


def compare_loop_runs(
    results_dir: str | Path,
    left_loop_run_id: str,
    right_loop_run_id: str,
    *,
    target: float | None = None,
) -> LoopComparison:
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
    (
        pd.concat([iterations, new_row], ignore_index=True)
        .set_index("iteration_index", drop=False)
        .sort_index()
        .to_csv(path, index=False)
    )


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
    (
        pd.concat([iterations, new_row], ignore_index=True)
        .set_index("iteration_index", drop=False)
        .sort_index()
        .to_csv(path, index=False)
    )


def _previous_iteration_row(
    results_dir: str | Path, loop_run_id: str, iteration_index: int
) -> dict[str, object] | None:
    iterations = read_iterations(results_dir, loop_run_id)
    previous = iterations[iterations["iteration_index"] < iteration_index]
    if previous.empty:
        return None
    return (
        previous.set_index("iteration_index", drop=False)
        .sort_index()
        .iloc[-1]
        .to_dict()
    )


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
                if "candidate_snapshot" in column or "interpretability_packet" in column
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


def _write_design_handoff_prompt(
    path: Path,
    *,
    manifest: LoopRunManifest,
    spec: ConditionSpec,
    input_manifest_path: Path,
) -> None:
    forbidden = ", ".join(spec.forbidden_artifacts) or "none"
    rationale_path = input_manifest_path.with_name(PRE_DESIGN_RATIONALE_FILENAME)
    prompt = f"""# LoopRun Design Handoff

You are the condition-specific model designer for one LoopRun iteration.

- Loop run id: `{manifest.loop_run_id}`
- Condition: `{manifest.condition}`
- Input manifest: `{input_manifest_path.resolve()}`
- Editable candidate file: `{Path(manifest.candidate_workspace_path).resolve()}`
- Pre-design rationale artifact: `{rationale_path.resolve()}`
- Primary metric: `{manifest.primary_metric}`
- Primary metric direction: `{manifest.primary_metric_direction}`
- Budget: `{manifest.budget}`
- Forbidden artifacts for this condition: {forbidden}

Rules:

1. Read the input manifest first.
2. Inspect only files listed in the manifest `files` array.
3. Before editing, complete the pre-design rationale template in your response
   and mirror it into the `pre_design_rationale_path` from the input manifest
   when the session can write files. This rationale is evidence for whether
   condition-specific artifacts shaped the modeling hypothesis.
4. Edit only the manifest `candidate_workspace_path` or a path listed in
   `editable_files`; the only other writable artifact is the pre-design
   rationale file listed in `writable_artifacts`.
5. Use exactly one modeling hypothesis for this iteration.
6. Prefer a material `fit`/`predict` behavior change that can move the primary
   metric. A text-only representation change is allowed only if an
   interpretability judgment will be applied before the next iteration.
7. For representation condition, translate each representation-derived cue into
   a concrete predictive mechanism expected to improve the primary metric. Do
   not use representation evidence only to preserve readability, remove
   hard-to-render terms, or raise interpretability. If allowed evidence shows
   that squared terms, pairwise interactions, hinges, or other broader basis
   terms improved RMSE, prefer refining that predictive structure and exposing
   dominant terms in `__str__` over simplifying to a weaker additive model.
   The default iteration objective is primary-metric improvement; judged
   interpretability is evidence about the representation, not a substitute for
   a performance-improving candidate.
8. Do not run `prepare`, `record`, `verify`, or final comparison commands.
9. Do not inspect project-wide `results/`, raw competition data, hidden targets,
   generator/oracle internals, condition-disallowed artifacts, candidate
   snapshots unless explicitly listed, or artifacts from another condition.

## Pre-Design Rationale Template

Fill this before describing code edits:

- Allowed artifacts inspected:
- Score movement observed:
- Representation-derived cues used:
- Predictive mechanism inferred from representation cues:
- Candidate hypothesis:
- Why this hypothesis follows from the allowed evidence:
- Why this should improve the primary metric rather than only the model string:
- Why any interpretability tradeoff is acceptable for the performance goal:
- Alternatives considered and rejected:
- Expected metric movement:
- Failure signal:

For blind condition, `Representation-derived cues used` must be
`N/A - forbidden by condition`. For representation condition, cite the specific
model string, interpretability packet, report, or judgment cue that influenced
the hypothesis, or write `None` if the hypothesis did not use representation
cues.

Return the completed pre-design rationale, modeling hypothesis, files
inspected, candidate file changed, and one risk for the runner to check after
`record`.
"""
    path.write_text(prompt)


def _write_pre_design_rationale_template(
    path: Path,
    *,
    manifest: LoopRunManifest,
    spec: ConditionSpec,
    iteration_index: int,
) -> None:
    representation_cue_default = (
        "N/A - forbidden by condition"
        if spec.name == "blind"
        else "Manual entry required"
    )
    path.write_text(
        "\n".join(
            [
                "# Pre-Design Rationale",
                "",
                f"- Loop run id: `{manifest.loop_run_id}`",
                f"- Condition: `{manifest.condition}`",
                f"- Iteration: `{iteration_index}`",
                f"- Primary metric: `{manifest.primary_metric}`",
                f"- Primary metric direction: `{manifest.primary_metric_direction}`",
                "",
                "## Context Boundary",
                "",
                "- Allowed artifacts inspected: Manual entry required",
                "- Forbidden artifacts not inspected: Manual entry required",
                "- Files inspected: Manual entry required",
                "",
                "## Causal Trace",
                "",
                "- Score movement observed: Manual entry required",
                f"- Representation-derived cues used: {representation_cue_default}",
                "- Predictive mechanism inferred from representation cues: "
                "Manual entry required",
                "- Candidate hypothesis: Manual entry required",
                "- Why this hypothesis follows from the allowed evidence: "
                "Manual entry required",
                "- Why this should improve the primary metric rather than only "
                "the model string: Manual entry required",
                "- Why any interpretability tradeoff is acceptable for the "
                "performance goal: Manual entry required",
                "- Alternatives considered and rejected: Manual entry required",
                "- Expected metric movement: Manual entry required",
                "- Failure signal: Manual entry required",
                "",
            ]
        )
    )


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
        if file_path.is_file()
        and file_path.name
        not in {
            "input_manifest.json",
            HANDOFF_PROMPT_FILENAME,
            PRE_DESIGN_RATIONALE_FILENAME,
        }
    )
    for forbidden in spec.forbidden_artifacts:
        if forbidden in bundle_text:
            findings.append(f"FAIL forbidden_artifact: found {forbidden}")
    if spec.name == "blind" and "## Model string" in bundle_text:
        findings.append("FAIL forbidden_artifact: found model string section")
    if not any(item.startswith("FAIL") for item in findings):
        findings.append(f"PASS condition_fairness: {spec.name}")
    return findings


def _verify_iteration_progress(
    results_dir: str | Path,
    manifest: LoopRunManifest,
    iterations: pd.DataFrame,
) -> list[str]:
    if iterations.empty:
        return []
    successful = iterations[iterations["status"] == "success"]
    if successful.empty:
        return []

    findings: list[str] = []
    leaderboard = _read_optional_leaderboard(results_dir)
    previous_row: dict[str, object] | None = None
    for row in (
        successful.set_index("iteration_index", drop=False)
        .sort_index()
        .to_dict("records")
    ):
        iteration = int(row["iteration_index"])
        candidate_sha = str(row.get("candidate_sha256", ""))
        if previous_row is None:
            if candidate_sha == manifest.baseline_candidate_sha256:
                findings.append(
                    "FAIL iteration_progress: iteration 1 candidate matches "
                    "baseline candidate; design did not change the model"
                )
            else:
                findings.append(
                    "PASS iteration_progress: iteration 1 candidate differs "
                    "from baseline candidate"
                )
            previous_row = row
            continue

        previous_iteration = int(previous_row["iteration_index"])
        previous_sha = str(previous_row.get("candidate_sha256", ""))
        if candidate_sha == previous_sha:
            findings.append(
                "FAIL iteration_progress: iteration "
                f"{iteration} candidate hash matches iteration "
                f"{previous_iteration}; no candidate edit was recorded"
            )
            previous_row = row
            continue

        previous_score = _float_or_nan(previous_row.get("primary_metric_value"))
        current_score = _float_or_nan(row.get("primary_metric_value"))
        if (
            not math.isnan(previous_score)
            and not math.isnan(current_score)
            and math.isclose(previous_score, current_score, rel_tol=0.0, abs_tol=1e-12)
        ):
            if _interpretability_improved(
                leaderboard,
                previous_run_id=str(previous_row.get("evaluation_run_id", "")),
                current_run_id=str(row.get("evaluation_run_id", "")),
            ):
                findings.append(
                    "PASS iteration_progress: iteration "
                    f"{iteration} primary metric was unchanged, but judged "
                    "interpretability improved"
                )
            else:
                findings.append(
                    "FAIL iteration_progress: iteration "
                    f"{iteration} changed candidate hash but left "
                    f"{manifest.primary_metric} unchanged without judged "
                    "interpretability improvement"
                )
        else:
            findings.append(
                "PASS iteration_progress: iteration "
                f"{iteration} changed candidate hash and moved "
                f"{manifest.primary_metric} from {previous_score} to {current_score}"
            )
        previous_row = row
    return findings


def _read_optional_leaderboard(results_dir: str | Path) -> pd.DataFrame:
    path = Path(results_dir) / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _interpretability_improved(
    leaderboard: pd.DataFrame,
    *,
    previous_run_id: str,
    current_run_id: str,
) -> bool:
    if leaderboard.empty or "interpretability_score" not in leaderboard.columns:
        return False
    previous = leaderboard[leaderboard["run_id"] == previous_run_id]
    current = leaderboard[leaderboard["run_id"] == current_run_id]
    if previous.empty or current.empty:
        return False
    previous_score = _float_or_nan(previous.iloc[-1].get("interpretability_score"))
    current_score = _float_or_nan(current.iloc[-1].get("interpretability_score"))
    return (
        not math.isnan(previous_score)
        and not math.isnan(current_score)
        and current_score > previous_score
    )


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
) -> LoopSummary:
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
