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
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T170011Z/candidate_snapshot.py

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

## Model string

```text
Polynomial RidgeCV regression. Inputs are z-scored, expanded to degree-2 polynomial features, then fit with ridge regularization selected by inner 5-fold CV. Selected alpha: 3.162. Intercept on the standardized polynomial basis: 0.348. Top learned basis terms by absolute coefficient: x0 (2.391), x1 (-1.626), x3 x4 (1.326), x2 (0.665), x7 (0.496), x6 x8 (-0.390), x2^2 (0.330), x8^2 (0.324), x8 (-0.275), x0 x10 (0.216). Basis includes 15 linear terms, 15 squared terms, and 105 pairwise interaction terms. Positive coefficients increase predictions as their standardized basis term rises; negative coefficients decrease predictions. Counterfactual sensitivity can be read from the dominant linear, squared, and interaction terms above, while smaller omitted terms remain ridge-shrunk rather than hard-pruned.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
