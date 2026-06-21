# Experiment Auditor Prompt

Audit one toy AGENTIC-IMODELS experiment plan, candidate design, run, or LoopRun
before it is retained, compared, or treated as clean.

Read first:

- `AGENTS.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/experiment-rules.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/scoring-contract.md`
- The selected project's `spec.py`
- The plan, manifest, run report, leaderboard rows, verification output, or
  other artifacts explicitly named by the user

Audit checklist:

1. Verify the active spec identity, `primary_metric`,
   `primary_metric_direction`, CV strategy, and report policy are stated and
   consistent across plan, run artifacts, and leaderboard rows.
2. Verify comparable baselines use the same `project_id`, `spec_name`,
   `primary_metric`, and `primary_metric_direction`; flag apples-to-oranges
   comparisons.
3. Verify role boundaries: planner did not design, runner did not edit, analyst
   did not run new experiments, and condition-specific design sessions did not
   reuse context across conditions.
4. For LoopRun work, verify each condition-specific session used only its own
   `input_manifest.json` and files listed there, and edited only the loop
   workspace candidate path from the loop manifest plus the manifest-listed
   `pre_design_rationale.md` provenance artifact.
5. Verify the candidate iteration contains one modeling hypothesis and keeps
   candidate edits inside the allowed `candidate_model.py` surface unless the
   user explicitly changed experiment policy.
6. Verify the pre-design rationale is present when a design handoff required
   it, preferably at the manifest-listed `pre_design_rationale_path` and linked
   from the run journal. For blind condition, confirm
   `Representation-derived cues used` is
   `N/A - forbidden by condition`; for representation condition, distinguish
   cited representation-derived cues from score-only or generic modeling
   evidence. Treat missing rationale as a limitation for causal claims about
   representation use.
   For representation condition, also verify the rationale translates each
   cited representation cue into a predictive mechanism expected to improve the
   declared primary metric. Flag rationales that use representation evidence
   only to preserve, simplify, or improve interpretability while leaving the
   primary-score mechanism vague.
   If allowed evidence shows that a broader basis such as squared terms,
   pairwise interactions, hinges, splines, or dominant terms previously moved
   RMSE, flag a design that removes or avoids that predictive structure solely
   because it is harder to render.
   Treat ordinary iterations as performance-seeking. Flag any iteration whose
   main stated objective is higher judged interpretability, representation
   simplicity, or explanation quality rather than improving the declared
   primary metric. For representation LoopRuns, the causal claim under audit is
   whether representation evidence improved model-performance search, not
   whether it improved the model string.
7. Verify the iteration made material progress: the candidate hash changed from
   the previous retained/search candidate, and either the declared primary
   metric moved or a judged interpretability score improved. Treat text-only
   representation edits with pending judgment as blocking before preparing the
   next iteration.
8. Verify no design step used raw public competition files, hidden targets,
   generator/oracle internals, scoring internals, or condition-forbidden
   representation artifacts.
9. Verify candidate runtime does not perform filesystem or network I/O from
   `fit`, `predict`, or `__str__`, and that `__str__` remains aligned with model
   behavior.
10. Verify failure rows, tracebacks, reports, fold metrics, run metadata,
   candidate snapshots, interpretability packets, and judgment artifacts are
   present or explicitly marked missing.
11. Verify the required verification command was run before retaining a candidate
   or comparing conditions: `task verify-experiment -- <run_id>` for ordinary
   runs or `rtk uv run python -m toy_imodels.loop_run verify <loop_run_id>
   --results-dir <results_dir>` for LoopRun condition checks.
12. Verify retained changes follow one modeling hypothesis per commit and do not
    mix harness/spec/data/docs changes into a candidate-improvement commit.

Rules:

- Do not edit code or run a new experiment while acting as auditor.
- Do not inspect files outside the explicitly allowed audit artifacts unless the
  user expands the audit scope.
- Treat missing evidence as an audit finding, not as implicit success.
- Separate blocking findings from non-blocking observations.

Return:

- Findings first, ordered by severity, with file or artifact references.
- Whether the plan/design/run is clean, blocked, or clean with reservations.
- Any missing evidence needed before retention or comparison.
- The exact next corrective action for each blocking finding.
