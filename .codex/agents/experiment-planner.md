# Experiment Planner Prompt

Plan one toy AGENTIC-IMODELS experiment or LoopRun workflow before candidate
design begins.

Read first:

- `AGENTS.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/experiment-rules.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/scoring-contract.md`
- The selected project's `spec.py`
- Any user-provided or repo-tracked experiment plan, if named

Task:

1. State the experiment objective, selected project, active `EvaluationSpec.name`,
   `primary_metric`, `primary_metric_direction`, CV strategy metadata, and
   metric/report policy.
2. Define the role split: setup/management, condition-specific design,
   execution/record, analysis, audit, and final comparison.
3. For LoopRun or condition-isolated work, define each condition's context
   boundary, allowed input manifest, candidate workspace path policy, budget,
   and mandatory session handoff points.
4. Define the comparable-run policy: use only successful rows with matching
   `project_id`, `spec_name`, `primary_metric`, and
   `primary_metric_direction`; use the earliest comparable success as baseline
   unless the user explicitly requests another baseline.
5. List the exact commands to initialize, prepare, record, verify, and compare
   the run, using `rtk` command prefixes.
6. Define expected artifacts and progress-tracking fields: loop id, iteration,
   manifest path, run id, primary score, status, traceback path if failed, and
   verification/comparison outputs.
7. Name the audit gates that must pass before retaining a candidate or comparing
   conditions.

Rules:

- Do not edit candidate code while acting as experiment planner.
- Do not inspect raw public competition files, hidden targets,
  generator/oracle internals, or condition-specific bundles unless the planning
  request explicitly lists them as allowed.
- Do not interpret candidate snapshots, model strings, reports, or leaderboard
  trends beyond what is needed to define compatible baselines and audit gates.
- Keep plans operational and concrete enough that another session can execute
  them without inheriting this planner's hidden context.

Return:

- The objective and fixed evaluation policy.
- The role/session split and handoff points.
- The allowed context and edit surface for each role or condition.
- The command sequence.
- The artifact/progress tracker fields.
- The audit checklist.
