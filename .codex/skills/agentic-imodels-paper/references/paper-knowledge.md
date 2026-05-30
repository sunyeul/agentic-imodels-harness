# AGENTIC-IMODELS Paper Knowledge

AGENTIC-IMODELS reframes interpretability for LLM agents. Instead of asking
whether a human can easily read a model, it asks whether an agent can use the
model's textual representation to answer questions about predictions, feature
effects, sensitivity, counterfactuals, and structure.

The paper's loop is:

1. Keep a fixed evaluation harness.
2. Let a coding agent modify a small sklearn-compatible regressor file.
3. Evaluate prediction quality and agent-facing interpretability.
4. Record results and iterate toward a better tradeoff.

Important mechanisms:

- The model exposes `fit`, `predict`, and `__str__`.
- Prediction quality is measured with regression metrics across tabular data.
- Domain-specific scoring functions can replace or supplement the default
  prediction metrics when they are treated as part of the fixed evaluation
  harness rather than as editable model logic.
- Interpretability is measured from the textual model representation.
- The key result is a better performance-agentic-interpretability frontier, not
  a single universally best model. Candidate improvements should search for
  models that both predict well and give another agent a useful textual handle
  on prediction logic, feature effects, sensitivity, counterfactual changes,
  and learned structure.
