# Run 20260620T170011Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_78e0298ecd80ef05
- Model: poly2_ridgecv
- Notes: RidgeCV over standardized linear, squared, and pairwise interaction features to test whether smooth second-order structure lowers RMSE.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 1.038394
- cv_rmse_std: 0.063996
- cv_mae_mean: 0.823323
- cv_mae_std: 0.043499
- cv_r2_mean: 0.910290
- cv_r2_std: 0.014621
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8300

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T170011Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T170011Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T170011Z/run_metadata.json

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T170011Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T170011Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8300
- prediction: 0.7800 - The model string gives the preprocessing, polynomial RidgeCV family, intercept, selected alpha, and top terms, but it does not provide the complete coefficient set needed for exact prediction.
- feature_effects: 0.9000 - The top learned basis terms include signed coefficients and relative magnitudes, making the main feature effects and interaction directions clear.
- sensitivity: 0.8000 - Feature changes can be tied to output direction and approximate magnitude for the listed dominant terms, while interactions, squared terms, and omitted coefficients prevent complete local sensitivity calculations.
- counterfactual: 0.7200 - Dominant positive, negative, squared, and interaction terms support qualitative increase/decrease reasoning, but exact threshold counterfactuals are limited by omitted smaller coefficients and standardized feature scaling.
- structure: 0.9500 - The model family, z-scoring, degree-2 polynomial expansion, ridge regularization, CV selection, and counts of linear, squared, and pairwise interaction terms are explicit.

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
