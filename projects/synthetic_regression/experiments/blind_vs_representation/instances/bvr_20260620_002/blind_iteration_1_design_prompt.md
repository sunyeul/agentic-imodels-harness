# Blind Iteration 1 Design Prompt

Follow AGENTS.md and repo instructions. Apply the
`agentic-imodels-toy-experiment` skill. This is a condition-specific `blind`
design session for:

- project: `synthetic_regression`
- experiment_name: `blind_vs_representation`
- experiment_id: `bvr_20260620_002`
- condition: `blind`
- loop_run_id: `bvr_20260620_002_blind`
- iteration: 1
- primary metric: `cv_rmse_mean`
- direction: minimize
- budget: 5 iterations for this condition

Strict context boundary:

- Read first:
  `projects/synthetic_regression/results/loop_runs/bvr_20260620_002_blind/agent_input_bundle/iteration_1/input_manifest.json`
- After that, inspect only files listed in that input manifest.
- Do not inspect representation condition artifacts.
- Do not inspect raw competition data, hidden targets, generator/oracle
  internals, model strings, interpretability packets, or condition-disallowed
  artifacts.
- Do not reuse context from any representation design session.
- Edit only the candidate workspace path recorded in the blind input manifest
  or loop manifest.
- Keep exactly one modeling hypothesis for this iteration.
- Prefix every shell command with `rtk`.

Task:

1. Read the blind iteration 1 input manifest.
2. Select one blind-allowed modeling hypothesis using only manifest-allowed
   context.
3. Edit only the blind loop workspace `candidate_model.py`.
4. Stop after design edits and report the hypothesis, edited path, and the
   command for the separate record step. Do not run `record` unless explicitly
   instructed.
