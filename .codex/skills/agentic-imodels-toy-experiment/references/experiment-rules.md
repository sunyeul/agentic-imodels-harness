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
  manifest and must edit only the loop workspace candidate file plus the
  manifest-listed `pre_design_rationale.md` provenance artifact.
- Condition-isolated LoopRun experiments must split sessions by role:
  setup/management sessions may initialize, prepare, verify, compare, and update
  progress docs; condition-specific design sessions start only after `prepare`
  creates that iteration's `input_manifest.json`; design sessions must not be
  reused across conditions; after `record`, return to setup/management before
  preparing or designing the next iteration.
- Comparable runs are successful rows with the same `project_id`, `spec_name`,
  `primary_metric`, and `primary_metric_direction`. Use the earliest comparable
  success as baseline and the latest three comparable successes as recent
  context unless the user explicitly asks for a different audit.
- Candidate iteration stays on the long-lived project branch. Each retained
  candidate improvement should be one commit representing one modeling
  hypothesis.
- Each iteration must make a material candidate improvement. Prefer a change
  that alters `fit`/`predict` behavior against the declared primary metric.
  A representation-only change is allowed only when the current iteration will
  also receive an agent interpretability judgment before another iteration is
  prepared; otherwise it is not sufficient evidence of progress.
- In representation LoopRun mode, representation evidence must be used as a
  performance-search signal, not just as a prompt to simplify the model string.
  The designer must state the predictive mechanism implied by each cited
  representation cue and why that mechanism should move the primary metric.
- Each LoopRun design iteration must preserve its causal trace in
  `agent_input_bundle/iteration_<n>/pre_design_rationale.md` before candidate
  editing. For blind condition, `Representation-derived cues used` must be
  `N/A - forbidden by condition`. For representation condition, the rationale
  must cite specific allowed representation evidence or explicitly state
  `None`.
- The default objective of a candidate iteration is primary-metric improvement.
  Interpretability evidence may guide the search, but judged interpretability
  improvement does not by itself satisfy the modeling objective unless the user
  explicitly requested an interpretability-only experiment.
- If representation evidence exposes dominant squared terms, pairwise
  interactions, threshold hinges, spline/basis terms, or feature groups that
  improved RMSE, the next hypothesis should normally test or refine that
  predictive structure with appropriate regularization and an agent-readable
  rendering. Do not discard a predictive structure solely because it is harder
  to explain; instead expose dominant terms, grouped effects, shrinkage, and
  omitted-term policy in `__str__`.
- Do not spend consecutive iterations on text-only `model_name`, `notes`, or
  `__str__` changes while the primary metric and judged interpretability are
  unchanged or pending.
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

Representation quality matters, but `__str__` should normally follow a real
modeling change rather than replace it. Use a representation-only iteration
only as a deliberate auditable exception, and judge it immediately before
continuing the loop.

For performance-causal representation experiments, the key evidence is not
that representation context changed the next candidate; it is that the
representation context changed the next candidate toward a defensible
primary-metric mechanism. A run that only raises judged interpretability while
leaving RMSE worse is evidence of representation drift, not evidence that
representation improved model search.

When the experiment hypothesis is that representation evidence helps model
improvement, the design and audit records must preserve that causal chain:
representation cue -> predictive mechanism -> candidate edit -> primary metric
movement. If the chain instead ends in a clearer `__str__` or a higher
interpretability score without RMSE improvement, report that as hypothesis-
contrary evidence.

Use the implementation structure that best serves the hypothesis inside
`projects/<project_name>/experiments/candidate_model.py`: compose sklearn pipelines, add helper
functions, define custom transformers, generate bases, store additive
components, or introduce internal model classes when those mechanisms make the
candidate more predictive and more useful to an interpreting agent.
