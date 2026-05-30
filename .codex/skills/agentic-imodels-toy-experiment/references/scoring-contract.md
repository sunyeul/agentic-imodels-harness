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
  interpretability score. Prefer agent-judged interpretability when a judgment
  has been applied; otherwise treat the static score as a fallback.

Custom scoring:

- A project may define a domain-specific scorer as the primary or secondary
  score.
- Custom scorers must live in the project evaluation spec, usually under
  `projects/synthetic_regression/spec.py`, not in `projects/synthetic_regression/experiments/candidate_model.py`.
- Custom CV strategies and scoring policy should be represented through an
  `EvaluationSpec` so user-requested experiment policy changes stay in one
  harness-owned layer.
- The leaderboard and run report should name and record the custom score so
  candidate iterations can compare against previous runs. Comparisons use
  `project_id`, `spec_name`, `primary_metric`, and `primary_metric_direction`.
- Candidate models may use score feedback from completed runs, but must not
  import, inspect, duplicate, or alter private scoring logic unless the user
  explicitly changes the experiment design.
- Candidate models and model-design agents must not inspect or encode project
  data-generation internals such as `_target_function` or oracle structure
  metadata. Use only public feature schemas, training data passed to `fit`, and
  completed run artifacts.

Artifacts:

- `projects/synthetic_regression/results/leaderboard.csv`
- `projects/synthetic_regression/results/submissions/<run_id>.csv`
- `projects/synthetic_regression/results/runs/<run_id>/report.md`
