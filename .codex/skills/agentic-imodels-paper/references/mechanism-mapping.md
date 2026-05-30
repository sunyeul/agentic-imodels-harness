# Mechanism Mapping

Map the paper to this toy repo as follows:

| Paper mechanism | Toy harness mechanism |
| --- | --- |
| Multiple tabular datasets | One deterministic synthetic regression dataset |
| Mean RMSE rank | Declared `EvaluationSpec.primary_metric` and `primary_metric_direction` plus RMSE diagnostics in the leaderboard |
| Task-specific evaluation metric | Project `EvaluationSpec` policy in `projects/synthetic_regression/spec.py` |
| `interpretable_regressor.py` | `projects/synthetic_regression/experiments/candidate_model.py` |
| Agent modifies model code | Agent edits only the candidate model file |
| LLM interpretability tests | Static fallback plus fixed judge packet/application harness in `toy_imodels/interpretability/` |
| Full autoresearch loop | Repeat run, inspect leaderboard, propose next candidate |

The toy version should keep the fixed-harness discipline. Do not move scoring,
splitting, aggregation, interpretability scoring, or leaderboard policy into
the editable candidate surface.

If a custom predictive scoring function is used, it belongs in the project
evaluation spec as part of the active `EvaluationSpec` and should be logged like
any other metric. The candidate may optimize for feedback from that score, but
it must not read, patch, or duplicate the scorer.
