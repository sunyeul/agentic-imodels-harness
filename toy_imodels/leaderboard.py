from __future__ import annotations

from pathlib import Path

import pandas as pd

LEADERBOARD_COLUMNS = [
    "run_id",
    "timestamp",
    "project_id",
    "candidate_module",
    "model_name",
    "notes",
    "spec_name",
    "primary_metric",
    "primary_metric_direction",
    "cv_strategy_name",
    "cv_n_splits",
    "cv_random_state",
    "interpretability_score",
    "interpretability_rubric_version",
    "interpretability_judgment_path",
    "interpretability_audit_status",
    "interpretability_audit_path",
    "status",
    "submission_path",
    "report_path",
    "fold_metrics_path",
    "run_metadata_path",
    "candidate_snapshot_path",
    "error_traceback_path",
    "error",
]


def append_result(results_dir: str | Path, row: dict) -> Path:
    project_id = row.get("project_id")
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("leaderboard row must include non-empty project_id")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    leaderboard_path = results_path / "leaderboard.csv"

    if leaderboard_path.exists():
        old = pd.read_csv(leaderboard_path)
        columns = [
            *LEADERBOARD_COLUMNS,
            *[column for column in old.columns if column not in LEADERBOARD_COLUMNS],
            *[
                column
                for column in row
                if column not in LEADERBOARD_COLUMNS and column not in old.columns
            ],
        ]
    else:
        columns = [
            *LEADERBOARD_COLUMNS,
            *[column for column in row if column not in LEADERBOARD_COLUMNS],
        ]

    normalized = {column: row.get(column, "") for column in columns}
    new_row = pd.DataFrame([normalized], columns=columns)
    if leaderboard_path.exists():
        combined = pd.concat([old.reindex(columns=columns), new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(leaderboard_path, index=False)
    return leaderboard_path


def read_leaderboard(results_dir: str | Path = "results") -> pd.DataFrame:
    leaderboard_path = Path(results_dir) / "leaderboard.csv"
    if not leaderboard_path.exists():
        return pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    return pd.read_csv(leaderboard_path)


def update_result(
    results_dir: str | Path,
    *,
    run_id: str,
    updates: dict,
    project_id: str | None = None,
) -> Path:
    results_path = Path(results_dir)
    leaderboard_path = results_path / "leaderboard.csv"
    if not leaderboard_path.exists():
        raise ValueError("Cannot update interpretability score without leaderboard.csv")

    leaderboard = pd.read_csv(leaderboard_path)
    matches = leaderboard["run_id"] == run_id
    if project_id is not None:
        if "project_id" not in leaderboard.columns:
            raise ValueError(
                f"No leaderboard project_id column found while updating run_id {run_id}"
            )
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

    for column in updates:
        if column not in leaderboard.columns:
            leaderboard[column] = pd.Series([""] * len(leaderboard), dtype=object)
    for column, value in updates.items():
        if isinstance(value, str):
            leaderboard[column] = leaderboard[column].astype(object)
        leaderboard.loc[matches, column] = value

    columns = [
        *LEADERBOARD_COLUMNS,
        *[
            column
            for column in leaderboard.columns
            if column not in LEADERBOARD_COLUMNS
        ],
    ]
    leaderboard.reindex(columns=columns).to_csv(leaderboard_path, index=False)
    return leaderboard_path
