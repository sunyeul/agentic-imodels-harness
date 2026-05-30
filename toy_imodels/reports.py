from __future__ import annotations

from pathlib import Path
from typing import cast

INTERPRETABILITY_JUDGMENT_HEADING = "## Interpretability judgment"


def write_run_report(
    run_dir: str | Path,
    *,
    run_id: str,
    project_id: str,
    candidate_module: str,
    model_name: str,
    notes: str,
    spec_name: str,
    primary_metric: str,
    primary_metric_direction: str,
    cv_strategy_name: str,
    cv_n_splits: int,
    cv_random_state: int | None,
    cv_description: str,
    metric_lines: list[str],
    interpretability_score: float,
    model_string: str,
    next_candidate_path: str,
    fold_metrics_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    run_metadata_path: str | Path | None = None,
    candidate_snapshot_path: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    report_path = run_path / "report.md"
    artifact_lines: list[str] = []
    if fold_metrics_path is not None:
        artifact_lines.append(f"- Fold metrics: {fold_metrics_path}")
    if diagnostics_path is not None:
        artifact_lines.append(f"- Diagnostics: {diagnostics_path}")
    if run_metadata_path is not None:
        artifact_lines.append(f"- Run metadata: {run_metadata_path}")
    if candidate_snapshot_path is not None:
        artifact_lines.append(f"- Candidate snapshot: {candidate_snapshot_path}")
    artifact_section = (
        ["", "## Run artifacts", "", *artifact_lines] if artifact_lines else []
    )
    report_path.write_text(
        "\n".join(
            [
                f"# Run {run_id}",
                "",
                f"- Project: {project_id}",
                f"- Candidate module: {candidate_module}",
                f"- Model: {model_name}",
                f"- Notes: {notes}",
                f"- Evaluation spec: {spec_name}",
                f"- Primary metric: {primary_metric}",
                f"- Primary metric direction: {primary_metric_direction}",
                f"- CV strategy: {cv_strategy_name}",
                f"- CV splits: {cv_n_splits}",
                f"- CV random state: {cv_random_state}",
                f"- CV description: {cv_description}",
                "",
                "## Metrics",
                "",
                *metric_lines,
                f"- Static interpretability score: {interpretability_score:.4f}",
                *artifact_section,
                "",
                "## Model string",
                "",
                "```text",
                model_string,
                "```",
                "",
                "## Next-step hint",
                "",
                "Inspect leaderboard movement, then change only "
                f"`{next_candidate_path}` for the next run.",
                "",
            ]
        )
    )
    return report_path


def apply_interpretability_judgment_to_report(
    report_path: str | Path,
    *,
    interpretability_score: float,
    judgment_path: str | Path,
    audit_path: str | Path,
    audit_status: str,
    dimension_scores: dict[str, dict[str, object]],
) -> Path:
    path = Path(report_path)
    report_text = path.read_text()
    static_line = "- Static interpretability score:"
    updated_line = (
        f"- Agent-judged interpretability score: {interpretability_score:.4f}"
    )
    lines = [
        updated_line if line.startswith(static_line) else line
        for line in report_text.splitlines()
    ]

    if INTERPRETABILITY_JUDGMENT_HEADING in lines:
        start = lines.index(INTERPRETABILITY_JUDGMENT_HEADING)
        end = len(lines)
        for index in range(start + 1, len(lines)):
            if lines[index].startswith("## "):
                end = index
                break
        del lines[start:end]

    section = [
        INTERPRETABILITY_JUDGMENT_HEADING,
        "",
        f"- Judgment artifact: {judgment_path}",
        f"- Audit artifact: {audit_path}",
        f"- Audit status: {audit_status}",
        f"- Final score: {interpretability_score:.4f}",
    ]
    for dimension_name, dimension in dimension_scores.items():
        score = cast(float, dimension["score"])
        section.append(f"- {dimension_name}: {score:.4f} - {dimension['rationale']}")
    section.append("")

    try:
        insert_at = lines.index("## Model string")
    except ValueError:
        insert_at = len(lines)
    lines[insert_at:insert_at] = section

    path.write_text("\n".join(lines).rstrip() + "\n")
    return path
