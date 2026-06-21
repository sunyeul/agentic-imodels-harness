# Run 20260620T173855Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_78e0298ecd80ef05
- Model: poly2_basis_scaled_ridgecv
- Notes: RidgeCV over degree-2 polynomial features with basis-wise standardization so linear, squared, and interaction terms receive comparable shrinkage.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 1.036101
- cv_rmse_std: 0.063348
- cv_mae_mean: 0.820770
- cv_mae_std: 0.042655
- cv_r2_mean: 0.910718
- cv_r2_std: 0.014313
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8500

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T173855Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T173855Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T173855Z/run_metadata.json

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T173855Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T173855Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8500
- prediction: 0.8500 - The model family, preprocessing, degree-2 basis expansion, ridge selection, intercept, and dominant coefficients are described, but the full coefficient list needed for exact output calculation is not shown.
- feature_effects: 0.9000 - Top learned basis terms are listed with signed coefficients and the text explains that positive coefficients increase predictions while negative coefficients decrease them.
- sensitivity: 0.8000 - Sensitivity can be inferred for dominant linear, squared, and interaction terms from their signs and magnitudes, though feature-level effects are context-dependent and smaller terms are omitted.
- counterfactual: 0.7500 - The text states how positive and negative dominant standardized basis terms affect predictions, so directional increase/decrease counterfactuals are possible, but exact changes and thresholds are limited by omitted ridge-shrunk terms.
- structure: 0.9500 - The representation clearly names Polynomial RidgeCV regression, z-scoring, standardized degree-2 linear, squared, and interaction basis terms, alpha selection, basis counts, and the degree constraint.

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
