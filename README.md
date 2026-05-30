# Agentic Imodels Toy Harness

This repository contains a small, Kaggle-like toy harness inspired by
AGENTIC-IMODELS.

The core code is organized into three layers:

| Layer | Path | Role | Normal candidate-iteration edits |
| --- | --- | --- | --- |
| Fixed harness | `toy_imodels/` | Owns candidate loading, CV execution, metric aggregation hooks, reports, leaderboard artifacts, and fixed interpretability evaluation. | No |
| Project definition | `projects/synthetic_regression/` | Owns the benchmark data contract, dataset loading/generation, project id, and project `EvaluationSpec` policy. | Only when intentionally changing the benchmark project |
| Candidate experiment | `projects/synthetic_regression/experiments/candidate_model.py` | Owns the editable sklearn-compatible candidate regressor, including model-specific preprocessing, feature engineering, model structure, and `__str__` rendering. | Yes |

`projects/synthetic_regression/run_experiment.py` is the project-scoped
entrypoint for running the fixed harness against the current candidate.

Repo-local Codex guidance in `.codex/` explains how to set up the harness,
map it back to the paper, and iterate on models with focused agent prompts.

## Quick start

Use the repo-local quickstart skill:
`.codex/skills/agentic-imodels-quickstart/SKILL.md`.

The quickstart first asks you to choose or confirm the project evaluation spec:

- use the default `synthetic_regression` demo
- customize the existing `synthetic_regression` evaluation spec
- verify an already-defined project evaluation spec

For the default demo, the project-specific evaluation policy lives in
`projects/synthetic_regression/spec.py`, while shared metric scoring,
aggregation, and report-line behavior live in `toy_imodels/spec.py`.

For delegated setup, use:
`.codex/agents/quickstart-runner.md`.

After quickstart, expect:

- synthetic competition files under `projects/synthetic_regression/data/`
- a baseline experiment row in `projects/synthetic_regression/results/leaderboard.csv`
- a submission CSV under `projects/synthetic_regression/results/submissions/`
- a run report under `projects/synthetic_regression/results/runs/`
- the smoke test suite to pass

## Codex harness

Use `.codex/skills/agentic-imodels-paper/SKILL.md` for paper-specific context.
Use `.codex/skills/agentic-imodels-toy-experiment/SKILL.md` for iteration
workflow. Focused agent prompts live under `.codex/agents/`.

The usual loop is:

1. quickstart setup
2. design one candidate-model change
3. run the experiment
4. inspect the leaderboard and report
5. analyze the result and choose the next single experiment
