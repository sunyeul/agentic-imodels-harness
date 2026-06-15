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
                "- Interpretability status: pending_agent_judgment",
                "- Interpretability score: NaN",
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


def write_agent_view_report(
    source_report_path: str | Path,
    target_report_path: str | Path,
    *,
    include_model_string: bool,
) -> Path:
    """Write the report an agent is allowed to inspect for a condition."""

    source_path = Path(source_report_path)
    target_path = Path(target_report_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lines = source_path.read_text().splitlines()
    if not include_model_string:
        lines = _remove_section(lines, "## Model string")
        lines = [
            line
            for line in lines
            if "Candidate snapshot:" not in line
            and "candidate_snapshot.py" not in line
            and "interpretability_packet.json" not in line
        ]
    target_path.write_text("\n".join(lines).rstrip() + "\n")
    return target_path


def _remove_section(lines: list[str], heading: str) -> list[str]:
    if heading not in lines:
        return lines
    start = lines.index(heading)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return [*lines[:start], *lines[end:]]


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
    pending_status_line = "- Interpretability status:"
    pending_score_line = "- Interpretability score:"
    updated_line = (
        f"- Agent-judged interpretability score: {interpretability_score:.4f}"
    )
    lines = []
    score_line_written = False
    for line in report_text.splitlines():
        if line.startswith(pending_status_line):
            lines.append("- Interpretability status: agent_judged")
        elif line.startswith(pending_score_line) or line.startswith(
            "- Static interpretability score:"
        ):
            lines.append(updated_line)
            score_line_written = True
        else:
            lines.append(line)
    if not score_line_written:
        lines.append(updated_line)

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
