# Experiment Runner Prompt

Run one toy AGENTIC-IMODELS experiment iteration.

Steps:

1. Inspect `projects/synthetic_regression/spec.py` and record the active `EvaluationSpec.name`,
   `primary_metric`, `primary_metric_direction`, CV strategy metadata, and
   custom metric/report policy.
2. Run `uv run python -m projects.synthetic_regression.run_experiment`.
3. Read the appended leaderboard row, including `spec_name`, `primary_metric`,
   `primary_metric_direction`, RMSE diagnostics, interpretability score, and any
   custom metric columns.
4. Read the generated run report.
5. Confirm generated artifact paths from the leaderboard row and report:
   `submission_path`, `report_path`, `fold_metrics_path`, `run_metadata_path`,
   `candidate_snapshot_path`, and any `error_traceback_path`.
6. If artifact paths exist, inspect `fold_metrics.json`, `run_metadata.json`,
   `candidate_snapshot.py`, and `interpretability_packet.json` enough to verify
   they correspond to the same run id and candidate source.
7. Report whether `interpretability_judgment_path` is present. If present,
   treat the leaderboard score as agent-judged; otherwise label it as static
   fallback and recommend `.codex/agents/interpretability-judge.md` when a judge
   score is needed.
8. If the run fails, read the leaderboard failure row and
   `error_traceback.txt` instead of hiding the failure.

Return:

- The run id, active evaluation spec, declared primary metric, primary metric direction,
   primary score, RMSE, MAE, R2 diagnostics, interpretability score source
   (agent-judged or static fallback), any custom score columns, and artifact
   paths.
- Fold-level metric range or notable outlier fold, if fold metrics exist.
- Whether run metadata and candidate snapshot were written.
- The latest score line printed by the command.

Do not edit code while acting as experiment runner.
