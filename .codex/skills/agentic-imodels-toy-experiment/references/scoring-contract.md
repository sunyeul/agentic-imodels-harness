# Scoring Contract

Candidate models must expose:

```python
class CandidateModel:
    model_name = "short_name"
    notes = "brief description"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return predictions

    def __str__(self):
        return "human- and agent-readable model description"
```

Scores:

- Primary: 5-fold CV mean RMSE, lower is better.
- Primary metric direction is recorded as `primary_metric_direction`; default
  RMSE uses `minimize`.
- Secondary: CV mean MAE, CV mean R2, fold standard deviations, and
  interpretability score after an agent judgment has been applied. Until then,
  leaderboard score is `NaN` and report/journal status is
  `pending_agent_judgment`.

Custom scoring:

- A project may define a domain-specific scorer as the primary or secondary
  score.
- Custom scorers must live in the project evaluation spec, usually under
  `projects/<project_name>/spec.py`, not in `projects/<project_name>/experiments/candidate_model.py`.
- Custom CV strategies and scoring policy should be represented through an
  `EvaluationSpec` so user-requested experiment policy changes stay in one
  harness-owned layer.
- The leaderboard and run report should name and record the custom score so
  candidate iterations can compare against previous runs. Comparisons use
  `project_id`, `spec_name`, `primary_metric`, and `primary_metric_direction`.
- Candidate models may use score feedback from completed runs, but must not
  import, inspect, duplicate, or alter private scoring logic unless the user
  explicitly changes the experiment design.
- Candidate models and model-design agents must not inspect or encode benchmark
  generator code, hidden targets, or oracle structure metadata. Use only public
  feature schemas, training data passed to `fit`, and completed run artifacts.

Artifact ownership:

- `projects/<project_name>/results/runs/<run_id>/` is the canonical artifact
  directory for one evaluation run. Keep run reports, fold metrics, run
  metadata, candidate snapshots, diagnostics, interpretability packets, and
  error tracebacks there.
- `projects/<project_name>/results/loop_runs/<loop_run_id>/` is the canonical
  directory for one LoopRun controller state. Keep the loop manifest, loop
  workspace candidate, condition-specific `agent_input_bundle/iteration_<n>/`
  manifests, design handoff prompts, and `iterations.csv` there. LoopRun rows
  reference evaluation evidence through `iterations.csv` column
  `evaluation_run_id`; do not duplicate full run directories under
  `loop_runs/`.
- `projects/<project_name>/experiments/<experiment_name>/instances/<experiment_id>/journals/<run_id>.md`
  is the tracked run-level experiment journal. Use it for provenance,
  hypothesis, result summary, artifact links, design rationale links, outcome
  review, judgment status, and next action.
  It summarizes and links to run artifacts; it is not the canonical storage
  location for raw run outputs. For standalone runs without an explicit
  experiment instance, use
  `projects/<project_name>/experiments/standalone/instances/default/journals/<run_id>.md`.
- Expected relationship: one LoopRun has many iterations; each recorded
  iteration has one evaluation `run_id`; each `run_id` has one
  `results/runs/<run_id>/` directory and one
  experiment-instance journal.

Core artifacts:

- `projects/<project_name>/results/leaderboard.csv`
- `projects/<project_name>/results/submissions/<run_id>.csv`
- `projects/<project_name>/results/runs/<run_id>/report.md`
- `projects/<project_name>/results/runs/<run_id>/run_metadata.json`
- `projects/<project_name>/results/runs/<run_id>/fold_metrics.json`
- `projects/<project_name>/results/runs/<run_id>/candidate_snapshot.py`
- `projects/<project_name>/experiments/<experiment_name>/instances/<experiment_id>/journals/<run_id>.md`
- `projects/<project_name>/results/loop_runs/<loop_run_id>/loop_manifest.json`
- `projects/<project_name>/results/loop_runs/<loop_run_id>/iterations.csv`
- `projects/<project_name>/results/loop_runs/<loop_run_id>/agent_input_bundle/iteration_<n>/input_manifest.json`
- `projects/<project_name>/results/loop_runs/<loop_run_id>/agent_input_bundle/iteration_<n>/pre_design_rationale.md`
- `projects/<project_name>/results/loop_runs/<loop_run_id>/agent_input_bundle/iteration_<n>/design_handoff_prompt.md`
