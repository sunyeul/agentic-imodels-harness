@/home/res1235/.codex/RTK.md

## Experiment Role Boundaries

Use skills for persistent repository workflows and subagents for isolated roles.

- Planning: define objective, condition/session split, budget, comparable baseline policy, commands, expected artifacts, and audit gates.
- Design: choose one modeling hypothesis and edit only the allowed candidate file.
- Run: execute the fixed harness and inspect generated artifacts. Do not edit code.
- Analyze: compare only compatible successful runs and recommend one next experiment.
- Audit: independently check plan/design/run integrity before retaining or comparing results.

## Hard Guardrails

- Do not inspect raw competition data, hidden targets, generator/oracle internals, or condition-disallowed artifacts during candidate design.
- Do not modify data, scoring, leaderboard logic, or evaluation spec during candidate iteration unless explicitly requested.
- Keep one modeling hypothesis per iteration and per candidate-improvement commit.
- Treat `candidate_model.py` as the design search space; the harness owns logs, reports, leaderboards, and artifacts.
- Run experiment verification before retaining a candidate result.

## Role-Specific Prompts

- Use `.codex/agents/experiment-planner.md` before starting or materially changing a multi-iteration or multi-condition experiment plan.
- Use `.codex/agents/experiment-auditor.md` for an independent plan/design/run audit before retaining a candidate, comparing conditions, or treating a LoopRun as clean.
- Use `.codex/agents/model-designer.md`, `.codex/agents/experiment-runner.md`, and `.codex/agents/result-analyst.md` for isolated design, execution, and analysis work.
- Keep setup/management, condition-specific design, execution, analysis, and audit contexts separate when a plan requires condition isolation.
