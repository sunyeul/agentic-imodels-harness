# Toy Experiment Workflow

1. Verify the committed public competition data loads:
   `uv run python -m projects.synthetic_regression.datasets`.
2. Resolve the target project before running:
   - If the user named a project, use it.
   - If exactly one project is available, state that project and its default spec.
   - If multiple projects are available and the user did not name one, list them
     and ask the user to choose.
   - Stay on the selected project's long-lived `project/` branch for candidate
     iteration unless the user explicitly asks for a separate branch.
3. Inspect the selected project's evaluation spec file and note the active
   `EvaluationSpec`, declared primary metric, primary metric direction, CV
   strategy, aggregation outputs, and report metric lines.
4. Establish the baseline as the earliest successful leaderboard row with the
   same `project_id`, `spec_name`, `primary_metric`, and
   `primary_metric_direction`.
   Use the latest three successful rows from that same comparable set as recent
   context. Do not inspect benchmark internals, hidden targets, or any data
   generation code when designing the next candidate.
5. Run the current candidate: `uv run python -m projects.synthetic_regression.run_experiment`.
6. Read `projects/synthetic_regression/results/leaderboard.csv`, including `spec_name`, `primary_metric`,
   `primary_metric_direction`, interpretability score, and any custom metric
   columns.
7. Open the latest `projects/synthetic_regression/results/runs/<run_id>/report.md` and
   `interpretability_packet.json`.
8. If no agent judgment exists, recommend using
   `.codex/agents/interpretability-judge.md`. If a draft judgment exists, apply
   it through the fixed harness with `apply_interpretability_judgment()`.
9. Compare against the baseline for predictive performance. Compare
   interpretability only after agent judgments exist for the relevant runs;
   otherwise label interpretability as `pending_agent_judgment`.
10. Form one modeling hypothesis against the declared primary metric and
    performance-agentic-interpretability frontier position. The hypothesis
    should name the improvement direction being explored, such as feature
    engineering, learned basis construction, additive structure, pruning,
    calibration, ensembling, or a clearer agent-readable model representation.
11. Edit only `projects/synthetic_regression/experiments/candidate_model.py`.
12. Re-run the experiment and compare scores. Recommend the next iteration, but
    wait for the user before running it.
13. If the user keeps the change, commit exactly that candidate improvement on
    the project branch. Use one commit per modeling hypothesis.
