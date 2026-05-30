---
name: agentic-imodels-quickstart
description: Use when setting up this repo for the first time, choosing or confirming the project EvaluationSpec, regenerating the synthetic dataset, running the smoke experiment, or checking that the toy AGENTIC-IMODELS harness is ready for iteration.
---

# Agentic Imodels Quickstart

Use this skill before model-iteration work when the repo may not be initialized,
when the active project evaluation spec is not yet confirmed, when generated data is
missing, or when a user asks for the quickstart path.

## Workflow

1. Read `references/setup-checklist.md`.
2. Start with Project Evaluation Spec Setup. Ask the user which setup mode they want:
   - `Use default synthetic_regression demo` (recommended)
   - `Customize the existing synthetic_regression evaluation spec`
   - `Verify an already-defined project evaluation spec`
3. For the default demo path, inspect `projects/synthetic_regression/spec.py`
   and `toy_imodels/spec.py`, summarize the current policy, and continue to
   verification without changing the spec.
4. For the customize path, ask briefly what should change about the metric, CV
   strategy, or spec name. Summarize the requested policy and get explicit user
   confirmation before any evaluation spec edit or verification run. Do not
   silently change experiment policy.
5. For the verify-only path, inspect the active project evaluation spec file, summarize it,
   and continue only after the active policy is clear.
6. Confirm the project evaluation spec summary before running the baseline. Include:
   `project_id`, active spec file, `EvaluationSpec.name`, `primary_metric`,
   `primary_metric_direction`, CV strategy name, split count, random seed,
   description, prediction metric keys, aggregation outputs, and report metric
   lines.
7. Run `uv run python -m projects.synthetic_regression.datasets` to create or refresh the synthetic competition files.
8. Run `uv run python -m projects.synthetic_regression.run_experiment` to verify the baseline candidate can train, score, write a submission, and append the leaderboard.
9. Run `uv run --extra dev pytest` for the full smoke test suite.
10. Inspect `projects/synthetic_regression/data/metadata.json`, `projects/synthetic_regression/results/leaderboard.csv`, `projects/synthetic_regression/results/submissions/`, and `projects/synthetic_regression/results/runs/` only enough to confirm artifacts exist.
11. Use `.codex/agents/quickstart-runner.md` when dispatching this setup as a focused subagent task.

## Spec Ownership

- Project-specific default policy lives in `projects/synthetic_regression/spec.py`.
- Shared metric scoring, fold aggregation, leaderboard metric selection, result
  metric mapping, and report metric line behavior live in `toy_imodels/spec.py`
  unless a project evaluation spec overrides them.
- The default demo uses project id `synthetic_regression` and
  `DefaultEvaluationSpec`.

## Boundaries

- Do not edit `projects/synthetic_regression/experiments/candidate_model.py` while acting as quickstart setup.
- Inspect `projects/synthetic_regression/spec.py` during quickstart, but do not
  change experiment policy unless the user explicitly asks for that change.
- Change evaluation spec policy only after the user selects a customize path
  and confirms the exact policy summary.
- This skill defines the conversational setup procedure; do not add a CLI wizard
  or code generator unless the user explicitly asks for one.
- Do not tune models or interpret scores beyond confirming the harness works.
- Do not delete existing leaderboard or run artifacts unless the user explicitly asks for a clean reset.
