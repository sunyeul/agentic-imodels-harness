# LoopRun Design Handoff

You are the condition-specific model designer for one LoopRun iteration.

- Loop run id: `bvr_20260621_001_representation_causal`
- Condition: `representation`
- Input manifest: `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/agent_input_bundle/iteration_1/input_manifest.json`
- Editable candidate file: `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py`
- Primary metric: `cv_rmse_mean`
- Primary metric direction: `minimize`
- Budget: `5`
- Forbidden artifacts for this condition: none

Rules:

1. Read the input manifest first.
2. Inspect only files listed in the manifest `files` array.
3. Before editing, write a pre-design rationale in your response using the
   template below. This rationale is evidence for whether condition-specific
   artifacts shaped the modeling hypothesis.
4. Edit only the manifest `candidate_workspace_path` or a path listed in
   `editable_files`.
5. Use exactly one modeling hypothesis for this iteration.
6. Prefer a material `fit`/`predict` behavior change that can move the primary
   metric. A text-only representation change is allowed only if an
   interpretability judgment will be applied before the next iteration.
7. For representation condition, translate each representation-derived cue into
   a concrete predictive mechanism expected to improve the primary metric. Do
   not use representation evidence only to preserve readability, remove
   hard-to-render terms, or raise interpretability. If allowed evidence shows
   that squared terms, pairwise interactions, hinges, or other broader basis
   terms improved RMSE, prefer refining that predictive structure and exposing
   dominant terms in `__str__` over simplifying to a weaker additive model.
   The default iteration objective is primary-metric improvement; judged
   interpretability is evidence about the representation, not a substitute for
   a performance-improving candidate.
8. Do not run `prepare`, `record`, `verify`, or final comparison commands.
9. Do not inspect project-wide `results/`, raw competition data, hidden targets,
   generator/oracle internals, condition-disallowed artifacts, candidate
   snapshots unless explicitly listed, or artifacts from another condition.

## Pre-Design Rationale Template

Fill this before describing code edits:

- Allowed artifacts inspected:
- Score movement observed:
- Representation-derived cues used:
- Predictive mechanism inferred from representation cues:
- Candidate hypothesis:
- Why this hypothesis follows from the allowed evidence:
- Why this should improve the primary metric rather than only the model string:
- Why any interpretability tradeoff is acceptable for the performance goal:
- Alternatives considered and rejected:
- Expected metric movement:
- Failure signal:

For blind condition, `Representation-derived cues used` must be
`N/A - forbidden by condition`. For representation condition, cite the specific
model string, interpretability packet, report, or judgment cue that influenced
the hypothesis, or write `None` if the hypothesis did not use representation
cues.

Return the completed pre-design rationale, modeling hypothesis, files
inspected, candidate file changed, and one risk for the runner to check after
`record`.
