# Model Designer Prompt

Design one candidate-model change for the selected toy AGENTIC-IMODELS
project. The change should advance the goal: discover an sklearn-compatible
regressor that improves the performance-agentic-interpretability frontier. A
strong candidate predicts well while exposing a textual representation another
agent can use to reason about predictions, feature effects, sensitivity,
counterfactual changes, and learned structure.

For candidate iteration, the primary optimization target is model performance:
improve the declared primary metric, usually lower `cv_rmse_mean`.
Interpretability is a required reporting and reasoning surface, but it is not
the objective that should drive ordinary iterations. In representation LoopRun
mode, representation evidence is an experimental treatment for improving model
search; use it only when it produces a concrete performance hypothesis.

Allowed context only:

LoopRun mode:

- If the task names a LoopRun, provides an `input_manifest.json`, or provides a
  `design_handoff_prompt.md`, treat that iteration manifest as the authoritative
  context boundary.
- Read the `input_manifest.json` first.
- Inspect only files listed in the manifest `files` array.
- Before editing, write a pre-design rationale using the handoff prompt's
  template when one is provided. If no template is provided, include the same
  fields in your final response.
  When the manifest includes `pre_design_rationale_path`, also write the
  completed rationale there before candidate edits.
- Edit only the manifest `candidate_workspace_path` or paths listed in
  `editable_files`; the only non-candidate write allowed in LoopRun mode is the
  manifest-listed pre-design rationale artifact.
- Do not read `loop_manifest.json`, project-wide `results/`, run directories,
  candidate snapshots, interpretability packets, raw data, or artifacts from
  another condition unless those exact paths are listed in the manifest.
- Do not run LoopRun `prepare`, `record`, `verify`, or comparison commands.
- Return after one modeling hypothesis has been implemented.

Standard project mode:

- `.codex/skills/agentic-imodels-toy-experiment/references/experiment-rules.md`
- `.codex/skills/agentic-imodels-toy-experiment/references/scoring-contract.md`
- `projects/<project_name>/spec.py`
- `projects/<project_name>/results/leaderboard.csv`, if it exists
- For the selected comparable set, read only these run artifacts:
  `report.md`, `fold_metrics.json`, `run_metadata.json`,
  `candidate_snapshot.py`, `interpretability_packet.json`,
  `interpretability_judgment.json`, and
  `interpretability_judgment_audit.json`, when present.
- `projects/<project_name>/experiments/candidate_model.py`

Comparable run selection:

- Comparable rows have the same `project_id`, `spec_name`, `primary_metric`,
  and `primary_metric_direction` as the active spec.
- Read the baseline row, defined as the earliest successful comparable row.
- Read the latest three successful comparable rows by leaderboard order.
- Read the latest failure row only if the user explicitly asks to analyze a
  failure.

Constraints:

- Edit only `projects/<project_name>/experiments/candidate_model.py` in
  standard project mode, or the manifest-listed LoopRun workspace
  `candidate_model.py` when a LoopRun manifest is active. In LoopRun mode,
  also complete the manifest-listed `pre_design_rationale.md` artifact when it
  exists; do not write any other experiment artifact.
- Treat the current `project/` branch as the experiment path. Do not create a
  separate experiment branch unless the user explicitly asks.
- Design one modeling hypothesis per iteration so the retained change can be
  committed as one candidate-improvement commit.
- Do not inspect or use benchmark generator code, hidden target structure, or
  metadata that reveals known target structure when designing a candidate.
  Treat them as hidden benchmark internals.
- Do not read raw public competition files
  (`projects/<project_name>/data/train.csv`, `valid.csv`, `test.csv`,
  `sample_submission.csv`, or `metadata.json`) while designing a candidate.
  Feature names come from the allowed spec, run reports, candidate snapshots,
  or the `X` columns passed to candidate `fit`.
- Do not read directories or files outside the allowed context list unless the
  user explicitly changes the experiment design.
- Candidate runtime must not perform filesystem or network I/O from `fit`,
  `predict`, or `__str__`.
- Preserve `fit`, `predict`, and `__str__`.
- Optimize the declared primary score without reading validation targets
  directly. If no custom primary score is declared, optimize CV mean RMSE.
  Use `primary_metric_direction` to decide whether lower or higher is better.
- Prefer a material `fit`/`predict` behavior change for each iteration. Do not
  spend an iteration only changing `model_name`, `notes`, or `__str__` unless
  the user explicitly asks for a judged representation-only iteration.
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
- The pre-design rationale must be written before or alongside the edit summary,
  not reconstructed from later run results.
- In blind LoopRun mode, set `Representation-derived cues used` to
  `N/A - forbidden by condition`.
- In representation LoopRun mode, cite the specific allowed model string,
  interpretability packet, report, or judgment cue that shaped the hypothesis,
  or write `None` if no representation cue was used.
- In representation LoopRun mode, representation-derived cues are useful only
  when they are translated into a predictive modeling mechanism. The rationale
  must explicitly state how the cue is expected to lower the primary metric,
  not merely how it preserves or improves interpretability.
- Do not use judged interpretability improvement as a substitute for primary
  metric improvement when choosing the next candidate. A representation cue
  should change the search path toward a better predictive structure, such as
  a feature basis, interaction family, regularization strategy, calibration
  rule, pruning rule, or ensemble component.
- Do not choose an interpretability-preserving refinement when the allowed
  evidence points to a stronger predictive structure. For example, if recent
  representation evidence shows dominant squared, interaction, hinge, or basis
  terms that improved RMSE, prefer testing a regularized version of that
  broader predictive basis over simplifying back to an additive rendering.
- If a previous representation attempt failed because important terms were not
  displayed, first consider improving the representation of the predictive
  structure, such as exposing dominant interaction terms and omitted-term
  policy, rather than removing the predictive structure solely to raise a
  judged interpretability score.
- If the hypothesis is representation-only, state that it is an exception and
  require an immediate interpretability judgment before another iteration is
  prepared.

Candidate improvement directions:

- Treat `projects/<project_name>/experiments/candidate_model.py` as the model-design search space, not
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
  internal model classes inside `projects/<project_name>/experiments/candidate_model.py` when the
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
4. Edit `projects/<project_name>/experiments/candidate_model.py`, or the
   manifest-listed LoopRun candidate file, only when the hypothesis is clear
   and it materially changes the candidate. Prefer changing model-owned
   preprocessing, feature construction, estimator structure, regularization,
   calibration, or pruning while updating `__str__` to explain that behavior.
5. Ensure the implementation remains sklearn-style and that `__str__` exposes
   enough prediction, feature-effect, sensitivity, counterfactual, and
   structure information for another agent to judge it.

Return:

- The completed pre-design rationale:
  allowed artifacts inspected, score movement observed, representation-derived
  cues used, predictive mechanism inferred from those cues, candidate
  hypothesis, why the hypothesis follows from the allowed evidence, why this is
  expected to improve the primary metric rather than only the model string,
  why any interpretability tradeoff is acceptable for the performance goal,
  alternatives considered and rejected, expected metric movement, and failure
  signal.
- The modeling hypothesis.
- The expected benefit.
- The improvement direction being explored, such as feature engineering,
  additive structure, pruning, calibration, ensembling, or representation
  rendering.
- The exact candidate code strategy.
- One risk to check after running.
- The files inspected and the candidate file path changed.
