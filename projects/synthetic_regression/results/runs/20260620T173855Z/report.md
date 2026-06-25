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
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T173855Z/candidate_snapshot.py

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

## Model string

```text
Polynomial RidgeCV regression. Inputs are z-scored, expanded to degree-2 polynomial features (linear, squared, and pairwise interaction terms), then each generated basis column is standardized before ridge fitting. The second standardization makes the ridge penalty comparable across linear, squared, and interaction columns instead of letting higher-variance basis terms receive different effective shrinkage. Ridge regularization is selected by inner 5-fold CV over an alpha grid from 0.1 to 100000. Selected alpha: 3.162. Intercept after input and basis standardization: 0.817. Top learned basis terms by absolute coefficient: x0 (2.391), x1 (-1.626), x3 x4 (1.308), x2 (0.665), x7 (0.496), x2^2 (0.496), x8^2 (0.452), x6 x8 (-0.373), x8 (-0.275), x0 x10 (0.250), x9 (0.207), x5 (0.201). Basis includes 15 linear terms, 120 quadratic terms, with 105 cross-feature interaction terms. Positive coefficients increase predictions as their standardized basis term rises; negative coefficients decrease predictions. Counterfactual sensitivity can be read from the dominant linear, squared, and interaction terms above, while smaller omitted terms remain ridge-shrunk rather than hard-pruned. The basis stays at degree 2 because the prior degree-3 expansion showed worse cross-validated RMSE.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
