from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROTECTED_EXPERIMENT_PATHS = (
    "toy_imodels/",
    "projects/synthetic_regression/spec.py",
    "projects/synthetic_regression/datasets.py",
    "projects/synthetic_regression/data/",
)


@dataclass(frozen=True, slots=True)
class GitProvenance:
    git_commit: str
    git_parent_commit: str
    git_dirty: bool


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_root(start: str | Path = ".") -> Path:
    completed = _run_git(["rev-parse", "--show-toplevel"], cwd=Path(start))
    if completed.returncode != 0:
        return Path(start).resolve()
    return Path(completed.stdout.strip()).resolve()


def collect_git_provenance(start: str | Path = ".") -> GitProvenance:
    root = repo_root(start)
    commit = _git_stdout(["rev-parse", "HEAD"], cwd=root)
    parent = _git_stdout(["rev-parse", "HEAD^"], cwd=root)
    dirty = bool(_git_stdout(["status", "--porcelain"], cwd=root))
    return GitProvenance(
        git_commit=commit,
        git_parent_commit=parent,
        git_dirty=dirty,
    )


def protected_paths_changed_since_commit(
    git_commit: str, *, root: str | Path = "."
) -> list[str]:
    if not git_commit:
        return []
    repo = repo_root(root)
    diff_changed = _git_stdout(
        ["diff", "--name-only", git_commit, "--", *PROTECTED_EXPERIMENT_PATHS],
        cwd=repo,
    )
    status_changed = _git_stdout(
        ["status", "--porcelain", "--", *PROTECTED_EXPERIMENT_PATHS],
        cwd=repo,
    )
    paths = set(filter(None, diff_changed.splitlines()))
    for line in status_changed.splitlines():
        if len(line) > 3:
            paths.add(line[3:])
    return sorted(paths)


def candidate_module_path(candidate_module: str, *, root: str | Path = ".") -> Path:
    return repo_root(root) / f"{candidate_module.replace('.', '/')}.py"


def journal_dir_for_results(results_dir: str | Path) -> Path:
    return Path(results_dir).resolve().parent / "experiments" / "journal"


def journal_path_for_run(results_dir: str | Path, run_id: str) -> Path:
    return journal_dir_for_results(results_dir) / f"{run_id}.md"


def write_experiment_journal(
    *,
    results_dir: str | Path,
    run_id: str,
    project_id: str,
    model_name: str,
    notes: str,
    spec_name: str,
    primary_metric: str,
    primary_metric_direction: str,
    primary_metric_value: float,
    candidate_source_sha256: str,
    git_provenance: GitProvenance,
    report_path: str | Path,
    run_metadata_path: str | Path,
    candidate_snapshot_path: str | Path,
    interpretability_status: str,
    interpretability_score: float,
    baseline_run_id: str,
) -> Path:
    journal_dir = journal_dir_for_results(results_dir)
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"{run_id}.md"
    if primary_metric_direction == "minimize":
        metric_direction = "lower is better"
    elif primary_metric_direction == "maximize":
        metric_direction = "higher is better"
    else:
        metric_direction = "see spec"
    metric_line = (
        f"- Primary metric: {primary_metric} "
        f"({primary_metric_direction}; {metric_direction})"
    )
    path.write_text(
        "\n".join(
            [
                f"# Experiment Journal: {run_id}",
                "",
                "## Provenance",
                "",
                f"- Project: {project_id}",
                f"- Run ID: {run_id}",
                f"- Commit: {git_provenance.git_commit}",
                f"- Parent commit: {git_provenance.git_parent_commit}",
                f"- Git dirty at run: {str(git_provenance.git_dirty).lower()}",
                f"- Candidate SHA256: {candidate_source_sha256}",
                f"- Spec: {spec_name}",
                metric_line,
                f"- Comparable baseline: {baseline_run_id}",
                "",
                "## Candidate",
                "",
                f"- Model: {model_name}",
                f"- Hypothesis: {notes or 'Manual entry required'}",
                "",
                "## Result",
                "",
                f"- {primary_metric}: {primary_metric_value:.6f}",
                f"- Interpretability status: {interpretability_status}",
                f"- Interpretability score: {interpretability_score:.4f}",
                "",
                "## Artifacts",
                "",
                f"- Report: {report_path}",
                f"- Run metadata: {run_metadata_path}",
                f"- Candidate snapshot: {candidate_snapshot_path}",
                "",
                "## Judgment Rationale",
                "",
                "- Manual entry required",
                "",
                "## Next Action",
                "",
                "- Manual entry required",
                "",
            ]
        )
    )
    return path


def update_journal_interpretability_judgment(
    *,
    results_dir: str | Path,
    run_id: str,
    interpretability_score: float,
    judgment_path: str | Path,
    audit_path: str | Path,
    audit_status: str,
) -> Path:
    path = journal_path_for_run(results_dir, run_id)
    if not path.exists():
        return path
    lines = path.read_text().splitlines()
    replacements = {
        "- Interpretability status:": "- Interpretability status: agent_judged",
        "- Interpretability score:": (
            f"- Interpretability score: {interpretability_score:.4f}"
        ),
    }
    updated = []
    for line in lines:
        replacement = None
        for prefix, value in replacements.items():
            if line.startswith(prefix):
                replacement = value
                break
        updated.append(replacement if replacement is not None else line)
    section = [
        "",
        "## Applied Interpretability Judgment",
        "",
        f"- Judgment artifact: {judgment_path}",
        f"- Audit artifact: {audit_path}",
        f"- Audit status: {audit_status}",
        f"- Final score: {interpretability_score:.4f}",
    ]
    heading = "## Applied Interpretability Judgment"
    if heading in updated:
        start = updated.index(heading)
        end = len(updated)
        for index in range(start + 1, len(updated)):
            if updated[index].startswith("## "):
                end = index
                break
        del updated[start:end]
    updated.extend(section)
    path.write_text("\n".join(updated).rstrip() + "\n")
    return path


def comparable_baseline_run_id(
    leaderboard: pd.DataFrame,
    *,
    project_id: str,
    spec_name: str,
    primary_metric: str,
    primary_metric_direction: str,
) -> str:
    if leaderboard.empty:
        return "missing"
    comparable = leaderboard[
        (leaderboard["project_id"] == project_id)
        & (leaderboard["status"] == "success")
        & (leaderboard["spec_name"] == spec_name)
        & (leaderboard["primary_metric"] == primary_metric)
        & (leaderboard["primary_metric_direction"] == primary_metric_direction)
    ]
    if comparable.empty:
        return "missing"
    return str(comparable.iloc[0]["run_id"])


def _git_stdout(args: list[str], *, cwd: Path) -> str:
    completed = _run_git(args, cwd=cwd)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
