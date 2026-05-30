# Setup Checklist

The project evaluation spec setup is complete when all of the following are true:

- The user has selected one setup mode:
  - `Use default synthetic_regression demo`
  - `Customize the existing synthetic_regression evaluation spec`
  - `Verify an already-defined project evaluation spec`
- The active project policy has been summarized, including `project_id`, active
  spec file, `EvaluationSpec.name`, `primary_metric`,
  `primary_metric_direction`, CV strategy name, split count, random seed,
  description, prediction metric keys, fold aggregation outputs, and report
  metric lines.
- If the user chose the default demo, the active project is
  `synthetic_regression` with `DefaultEvaluationSpec`.
- If the user chose to customize the existing spec, the requested metric, CV
  strategy, or spec name changes have been summarized and explicitly confirmed
  before any spec edit or verification run.
- If the user chose verify-only, the active spec location and policy summary are
  clear before verification begins.

The quickstart is successful when all of the following are true:

- `uv run python -m projects.synthetic_regression.datasets` exits successfully.
- `projects/synthetic_regression/data/train.csv`, `valid.csv`, `test.csv`, `sample_submission.csv`,
  and `metadata.json` exist.
- `projects/synthetic_regression/spec.py` defines the project-specific default
  `EvaluationSpec` policy, including the default spec name, primary metric, and
  CV strategy metadata.
- `toy_imodels/spec.py` defines the shared default prediction metric keys, fold
  aggregation outputs, leaderboard metric mapping, result metric mapping, and
  report metric lines unless the project evaluation spec overrides them.
- `uv run python -m projects.synthetic_regression.run_experiment` exits successfully.
- `projects/synthetic_regression/results/leaderboard.csv` contains at least one successful run.
- `projects/synthetic_regression/results/submissions/` contains a submission CSV for the latest run.
- `projects/synthetic_regression/results/runs/` contains a report for the latest run.
- `uv run --extra dev pytest` exits successfully.

Report the selected setup mode, current spec summary, command outputs, and
artifact paths. Do not claim readiness unless the commands have just been run in
this session.
