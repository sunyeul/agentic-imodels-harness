---
name: agentic-imodels-toy-experiment
description: Use when running or improving a toy AGENTIC-IMODELS single-dataset experiment loop for any project under projects/, especially when iterating on a selected project's experiments/candidate_model.py.
---

# Agentic Imodels Toy Experiment

Use this skill for repeatable model-iteration work in this repository.

Follow repo-wide role boundaries in `AGENTS.md` when present.

## Required Reading

Before acting, read:

- `references/experiment-rules.md`
- `references/scoring-contract.md`
- `references/workflow.md`

Use the workflow reference to choose the current session scope before inspecting
artifacts or editing code.

## Session Scope

- Setup, management, verification, and comparison requests stay in that narrow
  scope. Do not inspect design-only artifacts unless the plan explicitly allows
  them.
- Condition-specific LoopRun design starts only after `prepare` creates that
  iteration's `input_manifest.json`.
- Candidate design sessions inspect only their allowed manifest-listed files and
  edit only the recorded candidate file.
- Do not reuse a design session across LoopRun conditions.

## Workflow

Follow `references/workflow.md` for setup/management, LoopRun condition
isolation, standard candidate iteration, and PDCA artifact handling.

## Guardrails

- Do not modify data files, scoring, or leaderboard logic to improve scores.
- Treat the selected project's spec file as the project policy surface for
  user-requested evaluation changes. This is usually
  `projects/<project_name>/spec.py`. Do not change it during model
  iteration unless the user explicitly asks to change the experiment policy.
- Do not use validation or test targets inside candidate code.
- Do not inspect raw public competition files, hidden targets, generator code,
  or benchmark internals when designing a candidate. Candidate runtime must not
  perform filesystem or network I/O from `fit`, `predict`, or `__str__`.
- Do not create or mutate run logs from candidate code. Candidate models may
  provide `model_name`, `notes`, and `__str__`; the fixed harness owns log
  writing and artifact persistence.
- Keep `__str__` informative enough for another agent to reason about the model.
- During candidate iteration, keep edits inside
  `projects/<project_name>/experiments/candidate_model.py`. That file is the
  model-design search space.
- Use commit history as the experiment path: one candidate improvement equals
  one commit on the project branch. Create separate branches only when the user
  explicitly asks for branch-based isolation.

## Focused Subagents

Use `.codex/agents/experiment-planner.md`, `.codex/agents/model-designer.md`,
`.codex/agents/experiment-runner.md`, `.codex/agents/result-analyst.md`, or
`.codex/agents/experiment-auditor.md` for isolated planning, design, execution,
analysis, and audit work.
