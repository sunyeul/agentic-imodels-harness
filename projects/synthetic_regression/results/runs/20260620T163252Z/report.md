# Run 20260620T163252Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_da49e5f7dbeadc36
- Model: direct_sparse_cubic_elasticnet_cv
- Notes: ElasticNetCV directly fits a sparse cubic polynomial basis, retaining the higher-order terms that improved RMSE while preserving shrinkage that the post-selection Ridge refit appeared to remove.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 0.848607
- cv_rmse_std: 0.038458
- cv_mae_mean: 0.673407
- cv_mae_std: 0.024712
- cv_r2_mean: 0.940273
- cv_r2_std: 0.007649
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8500

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T163252Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T163252Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T163252Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T163252Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T163252Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T163252Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8500
- prediction: 0.8500 - It provides an intercept and explicit retained-term equation for dominant nonzero expanded terms, but the equation is truncated while predict() still uses omitted active terms.
- feature_effects: 0.9000 - The model string lists signed coefficients and top absolute active terms, making dominant expanded-feature effects and relative importance clear despite truncation.
- sensitivity: 0.8000 - The additive expanded-basis coefficients tie term changes to output direction and magnitude, though marginal effects for original features require accounting for nonlinear and interaction terms.
- counterfactual: 0.7500 - Positive and negative retained terms support local increase/decrease reasoning, but original-feature counterfactuals are complicated by powers, interactions, and omitted active terms.
- structure: 0.9500 - The family, cubic polynomial basis, regularization settings, sparsity counts, degree breakdown, and retained interactions are explicitly described.

## Model string

```text
Direct sparse cubic ElasticNetCV regression over original features, squared terms, pairwise interactions, and third-order polynomial terms. ElasticNetCV predicts directly from the regularized cubic basis, correcting the prior post-selection Ridge refit by keeping coefficient shrinkage on retained high-order terms. Prediction equation for nonzero retained expanded terms: y = 0.385 + 2.351*x0 + -1.521*x1 + 1.268*x3 x4 + 0.425*x7 + 0.398*x2 + 0.319*x2^2 + -0.269*x6 x8 + 0.268*x8^2 + 0.197*x6 x7^2 + 0.165*x5 + -0.162*x8 + 0.139*x9 + -0.134*x6 x8^2 + 0.083*x6 + -0.063*x7 x9 x12 + 0.054*x2^3 + -0.042*x5^3 + -0.040*x2 x5 x12 + -0.039*x6^2 x8 + 0.037*x0 x2 x4 + 0.035*x0^2 x2 + 0.035*x2 x10^2 + 0.034*x1 x2 + -0.034*x9^2 + 0.031*x5 x7 + 0.029*x4 x7 x10 + 0.028*x0 x7 + -0.028*x7 x12 + -0.027*x3 x4 x8 + 0.027*x6 x12 x14 + -0.027*x1 x4^2 + 0.026*x5 x12^2 + 0.026*x5 x8^2 + -0.022*x2 x8 x10 + 0.020*x2 x8^2 + -0.017*x9 x10 x12 + -0.015*x0 x11 + -0.015*x4^2 x8 + -0.014*x7^2 x13 + 0.013*x4^2 x5 + -0.012*x5 x9 x13 + 0.012*x6^2 x7 + -0.012*x6 x13 + -0.011*x0 x2 + 0.011*x0^2 x6 + 0.011*x10 x12 + -0.010*x4 x11 + -0.009*x3^2 x7 + -0.009*x3 x14 + 0.009*x8 x10 x13 + 0.009*x6 x7 x12 + -0.009*x3 x8^2 + 0.008*x3 x7 x14 + -0.007*x3 x6 x14 + 0.007*x9 x14 + 0.007*x5 x14^2 + -0.007*x8^3 + -0.006*x2 x7 x13 + 0.006*x3 x10^2 + -0.006*x0^2 + .... ElasticNet alpha: 0.04044; l1_ratio: 1. Active expanded terms after direct shrinkage: 83; zeroed weak expanded terms: 732; active terms by degree: linear: 8, quadratic: 18, cubic: 57. Top absolute active coefficient terms: x0 (2.351), x1 (-1.521), x3 x4 (1.268), x7 (0.425), x2 (0.398), x2^2 (0.319), x6 x8 (-0.269), x8^2 (0.268), x6 x7^2 (0.197), x5 (0.165), x8 (-0.162), x9 (0.139), x6 x8^2 (-0.134), x6 (0.083), x7 x9 x12 (-0.063). Top retained cubic terms: x6 x7^2 (0.197), x6 x8^2 (-0.134), x7 x9 x12 (-0.063), x2^3 (0.054), x5^3 (-0.042), x2 x5 x12 (-0.040), x6^2 x8 (-0.039), x0 x2 x4 (0.037). Coefficients are shown after converting standardized polynomial terms back to their expanded-feature units. If the equation is truncated, omitted active terms are smaller than the displayed dominant terms but are still used by predict(); terms not shown were removed by the sparse selection step. Positive active terms raise the prediction as their expanded value increases; negative active terms lower it. Counterfactual sensitivity is local and additive in this retained expanded basis, so changing an original feature can affect its linear term, squared/cubic powers, and retained interactions that include it.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py` for the next run.
