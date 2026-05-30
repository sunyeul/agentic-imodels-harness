# Interpretability Judge Prompt

Judge one toy AGENTIC-IMODELS run's agent-facing interpretability.

Input:

- Read `projects/synthetic_regression/results/runs/<run_id>/interpretability_packet.json`.
- Use only the packet contents and the fixed rubric inside that packet.
- Do not inspect or edit `projects/synthetic_regression/experiments/candidate_model.py` while judging.

Task:

1. Read the packet's `run_id`, `rubric_version`, `model_string`,
   `scoring_dimensions`, questions, calibration guide, and output schema.
2. Score each dimension from `0.0` to `1.0`:
   - `prediction`: whether the model string exposes a prediction equation,
     rules, or usable output calculation procedure.
   - `feature_effects`: whether important features, coefficients, effect
     directions, importances, or signs are clear.
   - `sensitivity`: whether feature changes can be tied to output direction or
     magnitude changes.
   - `counterfactual`: whether increase/decrease or threshold counterfactuals
     can be answered from the model string.
   - `structure`: whether the model family, transformations, interactions,
     rules, intervals, or simplicity constraints are explicit.
3. Write `projects/synthetic_regression/results/runs/<run_id>/interpretability_judgment.draft.json` matching
   the packet's output schema.
4. Set `interpretability_score` to the mean of the five dimension scores,
   rounded to 4 decimals.
5. Return the draft judgment path, final score, and one compact sentence
   explaining the main evidence.

Rules:

- Do not update `projects/synthetic_regression/results/leaderboard.csv`.
- Do not edit `report.md`.
- Do not run `apply_interpretability_judgment`.
- Keep rationales evidence-based and grounded in the packet's `model_string`.
- Use the packet's calibration guide to avoid automatic `1.0` scores when the
  model string only partially supports a dimension.
- If the packet is missing, malformed, or has an unsupported schema, stop and
  report the issue without writing a judgment.
