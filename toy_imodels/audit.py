from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from projects.synthetic_regression import synthetic_regression_project
from toy_imodels.provenance import (
    candidate_module_path,
    comparable_baseline_run_id,
    protected_paths_changed_since_commit,
    repo_root,
    sha256_file,
    sha256_text,
)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    check: str
    ok: bool
    detail: str


def verify_experiment_run(
    *,
    results_dir: str | Path,
    run_id: str | None = None,
    project: Any | None = None,
) -> list[AuditFinding]:
    results_path = Path(results_dir)
    leaderboard = _read_leaderboard(results_path)
    target = _target_row(leaderboard, run_id=run_id)
    target_run_id = str(target["run_id"])
    metadata_path = Path(str(target["run_metadata_path"]))
    metadata: dict[str, Any] = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    active_project = project or synthetic_regression_project(results_dir=results_path)
    spec_metadata = active_project.spec.report_metadata()

    findings = [
        _check_metadata_present(metadata_path, metadata),
        _check_snapshot_hash(metadata),
        _check_active_candidate_hash(metadata),
        _check_active_spec(metadata, spec_metadata),
        _check_protected_paths(metadata),
        _check_comparable_baseline(
            leaderboard,
            metadata=metadata,
            project_id=str(target["project_id"]),
        ),
        _check_journal(results_path, target_run_id),
    ]
    return findings


def latest_successful_run_id(results_dir: str | Path) -> str:
    leaderboard = _read_leaderboard(Path(results_dir))
    target = _target_row(leaderboard, run_id=None)
    return str(target["run_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify experiment audit artifacts.")
    parser.add_argument(
        "run_id",
        nargs="?",
        help="Run id to verify; defaults to latest success.",
    )
    parser.add_argument(
        "--results-dir",
        default="projects/synthetic_regression/results",
        help="Project results directory.",
    )
    args = parser.parse_args(argv)

    findings = verify_experiment_run(results_dir=args.results_dir, run_id=args.run_id)
    for finding in findings:
        status = "PASS" if finding.ok else "FAIL"
        print(f"{status} {finding.check}: {finding.detail}")
    return 0 if all(finding.ok for finding in findings) else 1


def _read_leaderboard(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "leaderboard.csv"
    if not path.exists():
        raise ValueError(f"No leaderboard found at {path}")
    return pd.read_csv(path)


def _target_row(leaderboard: pd.DataFrame, *, run_id: str | None) -> pd.Series:
    if run_id is None:
        matches = leaderboard[leaderboard["status"] == "success"]
    else:
        matches = leaderboard[
            (leaderboard["run_id"] == run_id) & (leaderboard["status"] == "success")
        ]
    if matches.empty:
        target = "latest successful run" if run_id is None else f"run_id {run_id}"
        raise ValueError(f"No successful leaderboard row found for {target}")
    return matches.iloc[-1]


def _check_metadata_present(path: Path, metadata: dict[str, Any]) -> AuditFinding:
    if not path.exists():
        return AuditFinding("metadata_present", False, f"missing {path}")
    required = {
        "git_commit",
        "git_dirty",
        "candidate_source_sha256",
        "spec_name",
        "primary_metric",
        "primary_metric_direction",
    }
    missing = sorted(required - set(metadata))
    if missing:
        return AuditFinding(
            "metadata_present",
            False,
            "missing metadata keys: " + ", ".join(missing),
        )
    return AuditFinding("metadata_present", True, str(path))


def _check_snapshot_hash(metadata: dict[str, Any]) -> AuditFinding:
    snapshot_value = str(
        metadata.get("artifacts", {}).get("candidate_snapshot_path", "")
    )
    snapshot_path = Path(snapshot_value)
    expected = str(metadata.get("candidate_source_sha256", ""))
    if not snapshot_value or not expected:
        return AuditFinding(
            "snapshot_hash",
            False,
            "metadata is missing candidate snapshot path or source hash",
        )
    if not snapshot_path.is_file():
        return AuditFinding("snapshot_hash", False, f"missing {snapshot_path}")
    actual = sha256_file(snapshot_path)
    if actual != expected:
        return AuditFinding(
            "snapshot_hash",
            False,
            f"snapshot hash {actual} does not match metadata {expected}",
        )
    return AuditFinding("snapshot_hash", True, actual)


def _check_active_candidate_hash(metadata: dict[str, Any]) -> AuditFinding:
    module = str(metadata.get("candidate_module", ""))
    if not module:
        return AuditFinding(
            "active_candidate_hash",
            False,
            "metadata is missing candidate_module",
        )
    path = candidate_module_path(module, root=repo_root())
    expected = str(metadata.get("candidate_source_sha256", ""))
    if not expected:
        return AuditFinding(
            "active_candidate_hash",
            False,
            "metadata is missing candidate_source_sha256",
        )
    if not path.is_file():
        return AuditFinding("active_candidate_hash", False, f"missing {path}")
    actual = sha256_text(path.read_text())
    if actual != expected:
        return AuditFinding(
            "active_candidate_hash",
            False,
            f"active candidate hash {actual} does not match run {expected}",
        )
    return AuditFinding("active_candidate_hash", True, actual)


def _check_active_spec(
    metadata: dict[str, Any], spec_metadata: dict[str, object]
) -> AuditFinding:
    mismatches = []
    for key in ("spec_name", "primary_metric", "primary_metric_direction"):
        if str(metadata.get(key, "")) != str(spec_metadata[key]):
            mismatches.append(
                f"{key}: metadata={metadata.get(key, '')} active={spec_metadata[key]}"
            )
    if mismatches:
        return AuditFinding("active_spec", False, "; ".join(mismatches))
    return AuditFinding("active_spec", True, str(metadata.get("spec_name", "")))


def _check_protected_paths(metadata: dict[str, Any]) -> AuditFinding:
    if bool(metadata.get("git_dirty", False)):
        return AuditFinding(
            "protected_paths",
            True,
            "run metadata recorded a dirty tree; protected drift check is limited",
        )
    git_commit = str(metadata.get("git_commit", ""))
    changed = protected_paths_changed_since_commit(git_commit, root=repo_root())
    if changed:
        return AuditFinding(
            "protected_paths",
            False,
            "changed protected paths: " + ", ".join(changed),
        )
    return AuditFinding("protected_paths", True, "no protected harness/spec/data drift")


def _check_comparable_baseline(
    leaderboard: pd.DataFrame, *, metadata: dict[str, Any], project_id: str
) -> AuditFinding:
    baseline = comparable_baseline_run_id(
        leaderboard,
        project_id=project_id,
        spec_name=str(metadata.get("spec_name", "")),
        primary_metric=str(metadata.get("primary_metric", "")),
        primary_metric_direction=str(metadata.get("primary_metric_direction", "")),
    )
    if baseline == "missing":
        return AuditFinding(
            "comparable_baseline",
            False,
            "no successful comparable baseline row found",
        )
    return AuditFinding("comparable_baseline", True, baseline)


def _check_journal(results_dir: Path, run_id: str) -> AuditFinding:
    journal_path = (
        results_dir.resolve().parent / "experiments" / "journal" / f"{run_id}.md"
    )
    if not journal_path.exists():
        return AuditFinding("journal", False, f"missing {journal_path}")
    text = journal_path.read_text()
    required = ("Commit:", "Candidate SHA256:", "Comparable baseline:", "Next Action")
    missing = [item for item in required if item not in text]
    if missing:
        return AuditFinding(
            "journal",
            False,
            "missing journal fields: " + ", ".join(missing),
        )
    return AuditFinding("journal", True, str(journal_path))


if __name__ == "__main__":
    raise SystemExit(main())
