# Reproduction Checklist

Use this checklist when reviewing toy experiments:

- Fixed data generation and split.
- Fixed project `EvaluationSpec` in `projects/synthetic_regression/spec.py`,
  including primary metric, primary metric direction, CV strategy, prediction
  scoring, aggregation, and report metric lines.
- Fixed interpretability evaluator in `toy_imodels/interpretability/`.
- Any custom scoring function is fixed in the spec layer and not copied into
  the editable candidate.
- Candidate design does not inspect or encode synthetic data-generation
  internals, `_target_function`, or oracle structure metadata.
- Candidate model has `fit`, `predict`, and `__str__`.
- Candidate edits do not alter the harness.
- Each run is logged with `spec_name`, `primary_metric`,
  `primary_metric_direction`, score columns, and artifacts.
- Model string is available for interpretability inspection.
- Improvements are compared against prior leaderboard rows with the same
  `project_id`, `spec_name`, `primary_metric`, and
  `primary_metric_direction`, not isolated anecdotes.
- Result analysis reports whether a run improves both predictive performance
  and agent-facing interpretability against the comparable baseline, or is
  non-dominated among comparable successful runs.
