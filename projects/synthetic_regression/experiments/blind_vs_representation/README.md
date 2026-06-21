# Blind vs Representation

Experiment folder for testing whether representation artifacts improve
candidate-search efficiency under an equal LoopRun budget.

This directory owns experiment-level records: the hypothesis, protocol metadata,
the overall plan, repeated experiment instances, audits, comparisons,
conclusions, and run journals. It does not own canonical evaluation outputs.

Canonical harness artifacts remain under:

- `projects/synthetic_regression/results/runs/<run_id>/`
- `projects/synthetic_regression/results/loop_runs/<loop_run_id>/`
- `projects/synthetic_regression/results/leaderboard.csv`

Each instance under `instances/<experiment_id>/` binds one repeated comparative
trial to its condition LoopRuns and linked evaluation runs.
