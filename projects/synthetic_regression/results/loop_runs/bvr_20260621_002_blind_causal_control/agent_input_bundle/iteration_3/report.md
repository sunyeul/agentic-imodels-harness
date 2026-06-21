# Run 20260620T171443Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_78e0298ecd80ef05
- Model: poly3_ridgecv
- Notes: RidgeCV over standardized degree-3 polynomial features to test whether additional smooth nonlinear terms lower RMSE.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 1.701282
- cv_rmse_std: 0.087568
- cv_mae_mean: 1.304793
- cv_mae_std: 0.050054
- cv_r2_mean: 0.759843
- cv_r2_std: 0.032161
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8100

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T171443Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T171443Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T171443Z/run_metadata.json

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T171443Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T171443Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8100
- prediction: 0.7500 - It gives the prediction procedure, intercept, selected alpha, polynomial degree, and dominant coefficients, but not all coefficients or standardization parameters needed for exact calculation.
- feature_effects: 0.9000 - The model string lists top learned basis terms with signed coefficients and explains positive versus negative effects, giving clear relative effects for the dominant terms.
- sensitivity: 0.7500 - A feature or basis-term change can be tied to output direction and rough magnitude for listed terms, while local effects remain incomplete for interactions and omitted terms.
- counterfactual: 0.7000 - Dominant term signs support some increase/decrease counterfactual reasoning, but z-scoring, nonlinear interactions, and omitted ridge-shrunk terms prevent complete input-change answers.
- structure: 0.9500 - The model family, z-scoring, degree-3 polynomial expansion, ridge regularization, CV alpha selection, term counts, and interaction structure are explicit.

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
