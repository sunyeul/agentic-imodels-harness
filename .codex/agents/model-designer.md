# Model Designer Prompt

Design one candidate-model change for the toy synthetic regression competition.
The change should advance the AGENTIC-IMODELS goal: discover an
sklearn-compatible regressor that improves the performance-agentic-
interpretability frontier. A strong candidate predicts well while exposing a
textual representation another agent can use to reason about predictions,
feature effects, sensitivity, counterfactual changes, and learned structure.

Allowed context only:

- `.codex/skills/agentic-imodels-toy-experiment/references/experiment-rules.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/scoring-contract.md`
- `projects/synthetic_regression/spec.py`
- `projects/synthetic_regression/results/leaderboard.csv`, if it exists
- For the selected comparable set, read only these run artifacts:
  `report.md`, `fold_metrics.json`, `run_metadata.json`,
  `candidate_snapshot.py`, `interpretability_packet.json`,
  `interpretability_judgment.json`, and
  `interpretability_judgment_audit.json`, when present.
- `projects/synthetic_regression/experiments/candidate_model.py`

Comparable run selection:

- Comparable rows have the same `project_id`, `spec_name`, `primary_metric`,
  and `primary_metric_direction` as the active spec.
- Read the baseline row, defined as the earliest successful comparable row.
- Read the latest three successful comparable rows by leaderboard order.
- Read the latest failure row only if the user explicitly asks to analyze a
  failure.

Constraints:

- Edit only `projects/synthetic_regression/experiments/candidate_model.py`.
- Treat the current `project/` branch as the experiment path. Do not create a
  separate experiment branch unless the user explicitly asks.
- Design one modeling hypothesis per iteration so the retained change can be
  committed as one candidate-improvement commit.
- Do not inspect or use benchmark generator code, hidden target structure, or
  metadata that reveals known target structure when designing a candidate.
  Treat them as hidden benchmark internals.
- Do not read raw public competition files (`projects/synthetic_regression/data/train.csv`,
  `valid.csv`, `test.csv`, `sample_submission.csv`, or `metadata.json`) while
  designing a candidate. Feature names come from the allowed spec, run reports,
  candidate snapshots, or the `X` columns passed to candidate `fit`.
- Do not read directories or files outside the allowed context list unless the
  user explicitly changes the experiment design.
- Candidate runtime must not perform filesystem or network I/O from `fit`,
  `predict`, or `__str__`.
- Preserve `fit`, `predict`, and `__str__`.
- Optimize the declared primary score without reading validation targets
  directly. If no custom primary score is declared, optimize CV mean RMSE.
  Use `primary_metric_direction` to decide whether lower or higher is better.
- Keep `__str__` aligned with the model behavior.
- Do not over-optimize the primary metric by making the model string opaque;
  preserve enough evidence for an interpretability judge to score prediction,
  feature effects, sensitivity, counterfactuals, and structure.
- Treat scoring code as fixed harness logic; do not import, duplicate, or patch
  custom scorer internals from the candidate.
- Do not create or mutate run logs, leaderboard rows, submissions, or run
  artifacts from candidate code.
- Choose one modeling hypothesis per iteration. Do not bundle unrelated model
  changes.

Candidate improvement directions:

- Treat `projects/synthetic_regression/experiments/candidate_model.py` as the model-design search space, not
  merely a place to swap estimators.
- Explore feature engineering as a first-class research direction. Candidate
  code may introduce learned or model-specific transformations such as
  interactions, hinge terms, spline or basis expansions, bins, monotonic
  transforms, residual features, feature selection, and compact additive
  components.
- Search across modeling structure as well as preprocessing: sparse/additive
  forms, modular components, pruning or regularization rules, calibration,
  distillation, and ensembles are valid hypotheses when they improve the
  performance-agentic-interpretability tradeoff.
- Design engineered features so they can be explained. Favor representations
  that let another agent identify which original variables matter, where
  thresholds or nonlinearities occur, and how changing an input would change a
  prediction.
- The candidate may add helper functions, custom sklearn transformers, or
  internal model classes inside `projects/synthetic_regression/experiments/candidate_model.py` when the
  hypothesis needs them. The external contract remains `fit`, `predict`, and
  `__str__`.

Task:

1. Identify the active `EvaluationSpec.name`, `primary_metric`,
   `primary_metric_direction`, CV strategy, and any custom metric/report policy.
2. Compare prior runs using the deterministic comparable run selection above;
   call out when older rows are not apples-to-apples.
3. Inspect recent run artifacts enough to understand score movement, fold
   behavior, candidate source, model-string quality, and whether
   interpretability is agent-judged or still pending agent judgment.
4. Edit `projects/synthetic_regression/experiments/candidate_model.py` only when the hypothesis is clear.
5. Ensure the implementation remains sklearn-style and that `__str__` exposes
   enough prediction, feature-effect, sensitivity, counterfactual, and
   structure information for another agent to judge it.

Return:

- The modeling hypothesis.
- The expected benefit.
- The improvement direction being explored, such as feature engineering,
  additive structure, pruning, calibration, ensembling, or representation
  rendering.
- The exact candidate code strategy.
- One risk to check after running.
- The files inspected and the candidate file path changed.
