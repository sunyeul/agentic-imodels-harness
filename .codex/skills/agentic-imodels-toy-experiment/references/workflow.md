# Toy Experiment Workflow

## Scope Selection

Use the narrowest workflow that satisfies the user's request.

- Setup or management only: resolve the project and branch, then run only the
  requested setup, manifest, verification, comparison, or tracking-doc commands.
  Do not inspect leaderboard rows, reports, candidate snapshots,
  interpretability packets, raw data, hidden targets, generator/oracle
  internals, or condition-specific bundles unless the plan explicitly lists
  them as allowed.
- Candidate design: start only when the task asks for a candidate proposal,
  candidate edit, run interpretation, or frontier analysis, and the allowed
  context has been prepared.
- Comparison: compare only compatible successful runs after the relevant
  conditions have stopped or exhausted their budgets. Do not design or edit
  candidates in a final comparison session.

## LoopRun Condition Sessions

Condition-isolated LoopRun experiments are split by session role, not by branch.

- Use one shared setup/management session for branch setup, loop
  initialization, manifest compatibility checks, progress-table updates,
  `prepare`, `verify`, and final `compare_loop_runs`.
- Use a separate condition-specific design session for each condition and
  iteration after `prepare` creates that iteration's `input_manifest.json`.
- A condition-specific design session must inspect only its own
  `input_manifest.json` and files listed in that manifest.
- Candidate editing must be limited to the loop workspace candidate file
  recorded in the loop manifest. The only additional write allowed during
  design is completing the manifest-listed `pre_design_rationale.md`
  provenance artifact before candidate editing.
- Each design session must produce a material candidate improvement. Prefer a
  `fit`/`predict` behavior change that can move the declared primary metric.
  Treat text-only representation edits as exceptional; do not prepare the next
  iteration after a text-only edit unless an agent interpretability judgment has
  been applied and shows improvement.
- In representation sessions, require a performance-causal translation step:
  name the representation cue, name the predictive structure it suggests, and
  explain why that structure should lower `cv_rmse_mean`. Treat a change that
  only preserves readability, removes hard-to-render terms, or increases
  judged interpretability as insufficient unless it also has a concrete
  primary-metric mechanism.
- Save that translation step in
  `agent_input_bundle/iteration_<n>/pre_design_rationale.md`. This file is the
  durable evidence for the chain from allowed condition-specific context to
  candidate edit and expected primary-metric movement.
- The default LoopRun iteration goal is primary-metric improvement. Do not let
  representation-condition evidence turn the iteration into an interpretability
  optimization loop unless the user explicitly changes the objective.
- When representation evidence shows that broader structures such as squared
  terms, pairwise interactions, or hinge bases improved RMSE, prefer refining
  that broader basis and rendering its dominant terms over simplifying to a
  more readable but weaker additive model.
- Do not reuse a design session across conditions. A `blind` session must never
  inspect representation-only artifacts, and a `representation` session must not
  carry its context into a later `blind` design step.
- After a condition-specific `record` completes, return to setup/management
  before preparing or designing the next iteration.

Mandatory split points are after loop setup before the first condition design,
after every `prepare` before candidate design, after every `record` before the
next condition or iteration, and before final comparison.

LoopRun artifact relationship:

- Treat `results/loop_runs/<loop_run_id>/` as the loop controller state, not as
  the storage location for full evaluation results.
- Use `results/loop_runs/<loop_run_id>/iterations.csv` to map
  `iteration_index` to the evaluation `run_id` in column `evaluation_run_id`.
- Inspect full evaluation evidence under `results/runs/<run_id>/` after
  resolving the `run_id` from `iterations.csv`, subject to the current session
  scope and condition manifest.
- Inspect the paired run journal at
  `experiments/<experiment_name>/instances/<experiment_id>/journals/<run_id>.md`
  when auditing, retaining, or comparing a result. For standalone runs without
  an explicit experiment instance, use
  `experiments/standalone/instances/default/journals/<run_id>.md`.
- Journals should link the input manifest and pre-design rationale, then add an
  outcome review stating whether the observed primary-metric movement matched
  the pre-design expectation.
- Do not compare LoopRuns by scanning arbitrary run directories. Compare only
  successful rows recorded by each compatible LoopRun's `iterations.csv` and
  linked `results/runs/<run_id>/` evidence.

## Standard Candidate Iteration

1. Resolve the target project before running:
   - If the user named a project, use it.
   - If exactly one project is available, state that project and its default spec.
   - If multiple projects are available and the user did not name one, list them
     and ask the user to choose.
   - Stay on the selected project's long-lived `project/` branch for candidate
     iteration unless the user explicitly asks for a separate branch.
   - Let `<project_name>` be the selected package-safe project directory under
     `projects/`. Derive project paths and module commands from that name.
2. Verify the committed public competition data loads:
   `uv run python -m projects.<project_name>.datasets`.
3. Inspect the selected project's evaluation spec file and note the active
   `EvaluationSpec`, declared primary metric, primary metric direction, CV
   strategy, aggregation outputs, and report metric lines.
4. Establish the baseline as the earliest successful leaderboard row with the
   same `project_id`, `spec_name`, `primary_metric`, and
   `primary_metric_direction`.
   Use the latest three successful rows from that same comparable set as recent
   context. Do not inspect benchmark internals, hidden targets, or any data
   generation code when designing the next candidate.
5. Run the current candidate: `uv run python -m projects.<project_name>.run_experiment`.
6. Read `projects/<project_name>/results/leaderboard.csv`, including `spec_name`, `primary_metric`,
   `primary_metric_direction`, interpretability score, and any custom metric
   columns.
7. Open the latest `projects/<project_name>/results/runs/<run_id>/report.md` and
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
    Unless the user explicitly requests a judged representation-only iteration,
    the hypothesis should change `fit`/`predict` behavior.
    For representation-conditioned work, also state how any model string,
    interpretability packet, report, or judgment cue is expected to improve the
    primary metric. If the cue mainly says the model is hard to explain, prefer
    better rendering of the predictive structure over deleting the structure.
    Record the causal chain from representation evidence to predicted RMSE
    movement; later analysis should judge whether that chain succeeded.
11. Edit only `projects/<project_name>/experiments/candidate_model.py`.
12. Re-run the experiment and compare scores. Recommend the next iteration, but
    wait for the user before running it.
13. If the user keeps the change, commit exactly that candidate improvement on
    the project branch. Use one commit per modeling hypothesis.

## PDCA Run Artifact Usage

- Plan: inspect previous `projects/<project_name>/results/leaderboard.csv` rows
  and relevant `projects/<project_name>/results/runs/<run_id>/` artifacts before
  choosing one modeling hypothesis. Baseline comparisons use the earliest
  successful run with the same project, spec, primary metric, and primary metric
  direction. Name the candidate improvement direction explicitly, and connect it
  to the performance-agentic-interpretability frontier.
- Do: edit only `projects/<project_name>/experiments/candidate_model.py` and run
  the fixed harness.
- Check: inspect the current leaderboard row, `report.md`, `fold_metrics.json`,
  `run_metadata.json`, interpretability artifacts, and any
  `error_traceback.txt`. Treat interpretability as pending until
  `apply_interpretability_judgment()` has written the judgment, audit artifact,
  leaderboard update, report update, and journal update.
- Audit: run `task verify-experiment -- <run_id>` before retaining a candidate
  to check Git provenance, candidate snapshot hash, active spec metadata,
  protected harness/spec/data drift, comparable baseline, and the tracked
  journal entry. For LoopRuns, also run `toy_imodels.loop_run verify`; it should
  reject repeated candidate hashes and candidate edits that leave the primary
  metric unchanged without judged interpretability improvement.
- Act: choose the next single experiment using score movement, fold behavior,
  active spec metadata, `candidate_snapshot.py`, model string quality, and
  performance-agentic-interpretability frontier position. Stop after
  recommending the next iteration unless the user asks to run it.
- Commit: if the iteration is kept, commit the candidate change on the current
  `project/` branch with one modeling hypothesis per commit.
