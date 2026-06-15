# Experiment Rules

The harness owns fairness and repeatability.

- Candidate code may learn only from `X` and `y` passed to `fit`.
- Candidate code must not read `projects/<project_name>/data/valid.csv` or test targets.
- Candidate runtime must not perform filesystem or network I/O from `fit`,
  `predict`, or `__str__`.
- Candidate design must not inspect or encode benchmark generator code, hidden
  target structure, or metadata fields that reveal oracle structure. Treat those
  as benchmark internals, not public competition information.
- Candidate design may use only the active spec, generic tabular methods,
  current candidate code, completed leaderboard rows, and deterministic
  comparable run artifacts. It must not read raw public competition files while
  designing a candidate, and must not hand-code transformations discovered from
  benchmark internals.
- During a LoopRun iteration, the condition-specific
  `agent_input_bundle/iteration_<n>/input_manifest.json` is the authoritative
  context boundary. The designing agent must inspect only files listed in that
  manifest and must edit only the loop workspace candidate file.
- Comparable runs are successful rows with the same `project_id`, `spec_name`,
  `primary_metric`, and `primary_metric_direction`. Use the earliest comparable
  success as baseline and the latest three comparable successes as recent
  context unless the user explicitly asks for a different audit.
- Candidate iteration stays on the long-lived project branch. Each retained
  candidate improvement should be one commit representing one modeling
  hypothesis.
- The primary metric is 5-fold CV mean RMSE over the labeled data.
- If the experiment defines a custom primary score, optimize that fixed harness
  score instead of RMSE while still reporting RMSE as a diagnostic metric.
- Failed runs should remain visible in the leaderboard.
- The default editable surface is `projects/<project_name>/experiments/candidate_model.py`.
- Runtime harness files under `toy_imodels/` are stable unless the user asks to
  change the experiment design.
- Custom scoring logic is part of the runtime harness. Do not move it into the
  editable candidate or let candidate code special-case the scorer directly.
- User-requested changes to CV strategy or primary scoring should be made
  through the project `EvaluationSpec` layer in
  `projects/<project_name>/spec.py` whenever possible.
- Interpretability scoring belongs to the fixed harness under
  `toy_imodels/interpretability/`.

## Candidate Improvement Goal

Candidate iteration is the model-design search process at the center of
AGENTIC-IMODELS. The goal is not merely to improve predictive accuracy or to
produce a human-simple model. The goal is to discover sklearn-compatible
regressors that move the performance-agentic-interpretability frontier: models
that predict well and expose a textual representation another agent can use to
reason about predictions, feature effects, sensitivity, counterfactual changes,
and learned structure.

The candidate file is the search space for that model design. Iterations should
explore modeling and representation choices, including:

- model-owned preprocessing and learned feature construction
- feature engineering with interactions, hinge terms, spline or basis features,
  bins, monotonic transforms, residual features, and feature selection
- sparse, additive, modular, rule-like, tree-derived, or component-based model
  structure
- pruning, regularization, calibration, model selection, distillation, and
  ensembles when they improve the performance-agentic-interpretability tradeoff
- compact `__str__` rendering that exposes the learned feature logic and helps
  another agent reason about prediction behavior

Use the implementation structure that best serves the hypothesis inside
`projects/<project_name>/experiments/candidate_model.py`: compose sklearn pipelines, add helper
functions, define custom transformers, generate bases, store additive
components, or introduce internal model classes when those mechanisms make the
candidate more predictive and more useful to an interpreting agent.
