---
name: agentic-imodels-toy-experiment
description: Use when running or improving the toy AGENTIC-IMODELS single-dataset experiment loop, especially when iterating on projects/synthetic_regression/experiments/candidate_model.py.
---

# Agentic Imodels Toy Experiment

Use this skill for repeatable model-iteration work in this repository.

## Workflow

1. Read `references/experiment-rules.md`.
2. Read `references/scoring-contract.md`.
3. Resolve the target project before proposing a new model:
   - If the user named a project, use that project.
   - If exactly one project is available, state that project and its default
     spec explicitly before proceeding.
   - If more than one project is available and the user did not name one, list
     the available projects and ask the user to choose before running or editing
     experiments.
   - Use the project's long-lived `project/<project-name>` branch for candidate
     iteration. Do not create per-experiment branches by default.
4. Inspect the selected project's evaluation spec file before proposing a new model. Note
   the current `EvaluationSpec.name`, `primary_metric`,
   `primary_metric_direction`, CV strategy metadata, metric aggregation policy,
   and report metric lines.
5. Inspect the current leaderboard before proposing a new model. Establish the
   baseline as the earliest successful run with the same `project_id`,
   `spec_name`, `primary_metric`, and `primary_metric_direction`; compare older
   rows only when those fields match. Use the latest three successful rows from
   that comparable set as recent context.
6. Choose one candidate improvement direction that serves the
   AGENTIC-IMODELS goal of improving the performance-agentic-interpretability
   frontier. Treat feature engineering, learned basis construction, additive
   structure, pruning, calibration, ensembling, and agent-readable rendering as
   searchable model-design dimensions.
7. Edit only `projects/synthetic_regression/experiments/candidate_model.py` unless the user explicitly asks to change the harness.
8. Run `uv run python -m projects.synthetic_regression.run_experiment`.
9. Inspect `projects/synthetic_regression/results/leaderboard.csv`, the latest run report, and
   `interpretability_packet.json`. If no agent judgment exists, recommend using
   `.codex/agents/interpretability-judge.md`; if a draft judgment exists,
   apply it through the fixed harness with `apply_interpretability_judgment()`.
10. Analyze performance and interpretability together. Prefer agent-judged
   interpretability scores when present; otherwise mark the static score as a
   fallback. Report whether the latest run improves both predictive performance
   and interpretability against the baseline, or whether it is non-dominated
   among comparable successful runs.
11. Recommend the next single experiment, but do not run another iteration
    until the user chooses to continue.
12. Use `.codex/agents/model-designer.md`, `.codex/agents/experiment-runner.md`, or `.codex/agents/result-analyst.md` for focused subagent work.
13. When the user asks to persist an improvement, create exactly one commit for
    the single modeling hypothesis. Do not mix harness, spec, data, or docs
    changes into candidate-improvement commits.

## PDCA Run Artifact Usage

- Plan: inspect previous `projects/synthetic_regression/results/leaderboard.csv` rows and relevant
  `projects/synthetic_regression/results/runs/<run_id>/` artifacts before choosing one modeling hypothesis.
  Baseline comparisons use the earliest successful run with the same project,
  spec, primary metric, and primary metric direction. Name the candidate
  improvement direction explicitly, and connect it to the
  performance-agentic-interpretability frontier.
- Do: edit only `projects/synthetic_regression/experiments/candidate_model.py` and run the fixed harness.
- Check: inspect the current leaderboard row, `report.md`, `fold_metrics.json`,
  `run_metadata.json`, interpretability artifacts, and any `error_traceback.txt`.
  Prefer agent-judged interpretability when available; use the static score only
  as a fallback.
- Act: choose the next single experiment using score movement, fold behavior,
  active spec metadata, `candidate_snapshot.py`, model string quality, and
  performance-agentic-interpretability frontier position. Stop after
  recommending the next iteration unless the user asks to run it.
- Commit: if the iteration is kept, commit the candidate change on the current
  `project/` branch with one modeling hypothesis per commit.

## Guardrails

- Do not modify data files, scoring, or leaderboard logic to improve scores.
- Treat the selected project's spec file as the project policy surface for
  user-requested evaluation changes. For the current toy project this is
  `projects/synthetic_regression/spec.py`. Do not change it during model
  iteration unless the user explicitly asks to change the experiment policy.
- Do not use validation or test targets inside candidate code.
- Do not inspect raw public competition files, hidden targets, generator code,
  or benchmark internals when designing a candidate. Candidate runtime must not perform filesystem or network I/O
  from `fit`, `predict`, or `__str__`.
- Do not create or mutate run logs from candidate code. Candidate models may
  provide `model_name`, `notes`, and `__str__`; the fixed harness owns log
  writing and artifact persistence.
- Keep `__str__` informative enough for another agent to reason about the model.
- During candidate iteration, keep edits inside `projects/synthetic_regression/experiments/candidate_model.py`.
  That file is the model-design search space: feature engineering, learned
  transformations, helper functions, custom transformers, additive components,
  and internal model classes belong there when they serve one clear candidate
  hypothesis.
- Use commit history as the experiment path: one candidate improvement equals
  one commit on the project branch. Create separate branches only when the user
  explicitly asks for branch-based isolation.
