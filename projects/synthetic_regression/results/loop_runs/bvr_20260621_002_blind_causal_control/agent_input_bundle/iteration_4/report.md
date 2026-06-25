# Run 20260620T172633Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_78e0298ecd80ef05
- Model: poly2_regularized_ridgecv
- Notes: RidgeCV over standardized degree-2 polynomial features with a denser regularization grid to reduce cubic-basis variance and lower RMSE.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 1.036508
- cv_rmse_std: 0.063416
- cv_mae_mean: 0.822031
- cv_mae_std: 0.042391
- cv_r2_mean: 0.910611
- cv_r2_std: 0.014532
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8400

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T172633Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T172633Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T172633Z/run_metadata.json

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T172633Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T172633Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8400
- prediction: 0.8000 - The model family, preprocessing, polynomial expansion, intercept, alpha, and leading coefficients provide a usable partial calculation procedure, but the full equation is not recoverable from the text.
- feature_effects: 0.9000 - The top learned basis terms include coefficients and signs, making relative importance and effect direction clear for the dominant features, though not for every fitted term.
- sensitivity: 0.8000 - Feature changes can be tied to output direction and approximate standardized-basis magnitude for the listed dominant terms, while interactions, squared terms, and omitted coefficients limit local sensitivity detail.
- counterfactual: 0.7500 - The text says dominant linear, squared, and interaction terms can support counterfactual sensitivity, but exact increase/decrease answers are limited by omitted smaller terms and missing standardization details.
- structure: 0.9500 - The representation explicitly names the Polynomial RidgeCV family, z-scoring, degree-2 expansion, linear, squared and pairwise interaction structure, ridge regularization, alpha selection, and excluded cubic terms.

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
