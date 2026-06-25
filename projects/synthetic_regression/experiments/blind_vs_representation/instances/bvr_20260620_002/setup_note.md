# bvr_20260620_002 Setup Note

Scope: setup/management only. No candidate design, record, retention, final
comparison, condition-specific artifact interpretation, raw competition data,
hidden target, generator, or oracle internals were inspected.

Branch:

- `project/synthetic_regression`

Evaluation policy:

- Project: `synthetic_regression`
- Spec: `DefaultEvaluationSpec`, name `default`
- CV: `kfold_5_shuffle_seed42`
- Primary metric: `cv_rmse_mean`
- Direction: minimize
- Budget: 5 iterations per condition

LoopRuns:

- blind control: `bvr_20260620_002_blind`
- representation treatment: `bvr_20260620_002_representation`

Compatibility check:

- baseline commit matches: `f78d837c191512d80dcf19e4ab0c6508186cca3f`
- baseline candidate sha256 matches:
  `0ec48409ff694e26e37ea211aab5f71875cc372b330c271dd5d9cca6254f70f0`
- dataset sha256 matches:
  `626305c9f84d3167c79af8f53fae3f9af67f60f6ccea063aa04889b8ded5c67f`
- spec sha256 matches:
  `509f350448cdf8bccce621b3dd1dffc2b11ad597cc24e4c466bad572ce933bbc`
- primary metric, direction, and budget match across both loop manifests.

Prepared handoff:

- blind iteration 1 input manifest:
  `projects/synthetic_regression/results/loop_runs/bvr_20260620_002_blind/agent_input_bundle/iteration_1/input_manifest.json`

Next action:

- Start a fresh blind-only design session for iteration 1.
- The blind design session may read only its input manifest and files listed in
  that manifest, and may edit only the manifest-designated loop workspace
  candidate file.
