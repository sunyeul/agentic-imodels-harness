# Quickstart Runner Prompt

Prepare the toy AGENTIC-IMODELS repo for experiment iteration by selecting or
confirming the project evaluation spec, then verifying the harness.

Read first:

- `.codex/skills/agentic-imodels-quickstart/references/setup-checklist.md`
- `projects/synthetic_regression/spec.py`
- `toy_imodels/spec.py`

Steps:

1. Ask or infer the setup mode before verification:
   - `Use default synthetic_regression demo` (recommended)
   - `Customize the existing synthetic_regression evaluation spec`
   - `Verify an already-defined project evaluation spec`
2. If no setup mode was provided, use the default demo path and state that
   assumption in the return.
3. For the default demo path, inspect `projects/synthetic_regression/spec.py`
   and `toy_imodels/spec.py`, then summarize `project_id`, active spec file,
   `EvaluationSpec.name`, `primary_metric`, `primary_metric_direction`, CV
   strategy metadata, prediction metrics, aggregation outputs, and report
   metric lines. Do not edit the spec.
4. For the customize path, stop before verification unless the requested metric,
   CV strategy, or spec name changes have been summarized and explicitly
   confirmed. Do not silently change policy.
5. For the verify-only path, inspect the active spec location and summarize the
   policy before running verification.
6. Run `uv run python -m projects.synthetic_regression.datasets`.
7. Run `uv run python -m projects.synthetic_regression.run_experiment`.
8. Run `uv run --extra dev pytest`.
9. Confirm the synthetic dataset files exist.
10. Confirm `projects/synthetic_regression/results/leaderboard.csv`, a submission CSV, and a run report exist.

Return:

- Selected setup mode.
- Current spec summary.
- Dataset generation status.
- Baseline experiment status and latest score line.
- Test status.
- The latest leaderboard, submission, and report paths.

Rules:

- Do not edit code or model files while acting as quickstart runner.
- Do not edit project evaluation spec policy unless the user selected the
  customize path and explicitly confirmed the exact policy summary.
- Do not add a CLI wizard or code generator while acting as quickstart runner.
- Do not tune models or interpret scores beyond confirming readiness.
- Do not delete or reset existing generated artifacts.
- Do not claim readiness unless the dataset generation, baseline experiment,
  and test commands were run in this session.
