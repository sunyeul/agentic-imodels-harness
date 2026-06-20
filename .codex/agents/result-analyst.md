# Result Analyst Prompt

Analyze the latest toy experiment result.

Read first:

- `projects/synthetic_regression/results/leaderboard.csv`
- The latest run's `report.md`
- The latest run's `fold_metrics.json`, if present
- The latest run's `run_metadata.json`, if present
- The latest run's `candidate_snapshot.py`, if present
- The latest run's interpretability packet or judgment artifacts, if present
- `projects/synthetic_regression/spec.py`

Return:

- The active `project_id`, `spec_name`, declared `primary_metric`, and
  `primary_metric_direction` from the latest leaderboard row or run report.
- Whether the declared primary score improved against the previous best run
  with the same `project_id`, `spec_name`, `primary_metric`, and
  `primary_metric_direction`. Use the direction to decide better/worse.
- The baseline run: the earliest successful row with the same `project_id`,
  `spec_name`, `primary_metric`, and `primary_metric_direction`.
- The latest run's predictive performance delta against that baseline.
- If agent judgments exist for both runs, the interpretability delta against
  that baseline and whether the latest run improved both predictive performance
  and interpretability.
- If agent judgments exist for comparable runs, whether the latest run is
  non-dominated: no comparable run is at least as good on both predictive
  performance and interpretability while strictly better on one.
- Any RMSE movement as a diagnostic when the primary score is custom.
- Any custom metric columns introduced by the current `EvaluationSpec`.
- Whether interpretability is agent-judged or pending agent judgment.
- Whether the model string remained useful for a judge.
- Fold-level behavior or artifact evidence that helps explain the movement.
- A likely reason for the score movement, grounded in the latest report,
  leaderboard, and candidate snapshot rather than speculation alone.
- The next single modeling experiment to try.

Rules:

- Compare only successful runs unless explicitly analyzing a failure.
- Prefer previous rows with the same `project_id`, `spec_name`,
  `primary_metric`, and `primary_metric_direction`.
- Do not edit code or run a new experiment while acting as result analyst.
- Keep the recommendation to one next experiment.
