# Run 20260620T162417Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_da49e5f7dbeadc36
- Model: post_lasso_cubic_ridge_cv
- Notes: L1-select a sparse cubic polynomial basis, then refit RidgeCV on the retained terms to reduce shrinkage bias while preserving the higher-order structure that improved RMSE.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 0.915205
- cv_rmse_std: 0.082955
- cv_mae_mean: 0.716320
- cv_mae_std: 0.051197
- cv_r2_mean: 0.930319
- cv_r2_std: 0.012947
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.9000

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T162417Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T162417Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T162417Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T162417Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T162417Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T162417Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.9000
- prediction: 0.9500 - The retained expanded-term prediction equation is explicitly shown with intercept, coefficients, polynomial and interaction terms, plus the Lasso selection and Ridge refit procedure.
- feature_effects: 0.9500 - The model string lists signed coefficients for retained terms and identifies the top absolute active coefficient terms, making directions and relative strengths clear for expanded features.
- sensitivity: 0.8500 - The signed additive polynomial basis supports local sensitivity reasoning, but sensitivity for any original feature depends on its current value and retained interactions involving that feature.
- counterfactual: 0.8000 - Increase/decrease reasoning is supported by the signed retained polynomial terms and the note that effects are local and additive, but original-feature counterfactuals remain context-dependent because powers and interactions can change together.
- structure: 0.9500 - The model family, cubic expanded basis, sparse selection step, Ridge refit, active term counts by degree, alphas, and retained interaction structure are explicit.

## Model string

```text
Post-Lasso cubic RidgeCV regression over original features, squared terms, pairwise interactions, and third-order polynomial terms. A LassoCV selector first retains the sparse cubic basis; RidgeCV then refits the prediction equation on only those retained expanded terms to reduce L1 shrinkage bias. Prediction equation in retained expanded terms: y = 0.388 + 2.322*x0 + -1.427*x1 + 0.332*x2 + 0.319*x5 + 0.098*x6 + 0.471*x7 + -0.162*x8 + 0.183*x9 + -0.029*x0^2 + -0.042*x0 x2 + 0.072*x0 x7 + -0.068*x0 x11 + 0.045*x1 x2 + 0.350*x2^2 + 1.236*x3 x4 + -0.075*x3 x14 + -0.076*x4 x11 + 0.065*x5 x7 + -0.312*x6 x8 + -0.045*x6 x13 + -0.087*x7 x12 + 0.297*x8^2 + -0.078*x9^2 + 0.040*x9 x14 + 0.068*x10 x12 + 0.051*x0^2 x2 + 0.039*x0^2 x6 + 0.080*x0 x2 x4 + 0.038*x0 x4^2 + -0.019*x1^3 + -0.008*x1^2 x5 + -0.017*x1 x2^2 + -0.054*x1 x4^2 + -0.031*x1 x12^2 + 0.059*x2^3 + 0.037*x2^2 x13 + -0.082*x2 x5 x12 + 0.020*x2 x6^2 + -0.065*x2 x7 x13 + 0.042*x2 x8^2 + -0.081*x2 x8 x10 + 0.047*x2 x10^2 + -0.055*x3^2 x7 + -0.059*x3 x4 x8 + -0.031*x3 x6 x14 + 0.043*x3 x7 x14 + -0.032*x3 x8^2 + 0.068*x3 x10^2 + 0.008*x4^2 x5 + 0.025*x4^2 x6 + -0.028*x4^2 x8 + 0.073*x4 x7 x10 + -0.103*x5^3 + -0.004*x5 x6 x8 + 0.000*x5 x7^2 + 0.048*x5 x8^2 + -0.092*x5 x9 x13 + 0.040*x5 x12^2 + 0.048*x5 x14^2 + 0.036*x6^2 x7 + -0.058*x6^2 x8 + 0.209*x6 x7^2 + 0.024*x6 x7 x12 + -0.185*x6 x8^2 + 0.002*x6 x10^2 + 0.079*x6 x12 x14 + -0.033*x7^2 x13 + -0.063*x7 x9 x12 + 0.003*x7 x10^2 + -0.011*x8^3 + 0.054*x8 x10 x13 + -0.094*x9 x10 x12. Selection Lasso alpha: 0.04344; refit Ridge alpha: 10. Active expanded terms after selection/refit: 72; zeroed weak expanded terms: 743; active terms by degree: linear: 8, quadratic: 17, cubic: 47. Top absolute active coefficient terms: x0 (2.322), x1 (-1.427), x3 x4 (1.236), x7 (0.471), x2^2 (0.350), x2 (0.332), x5 (0.319), x6 x8 (-0.312), x8^2 (0.297), x6 x7^2 (0.209), x6 x8^2 (-0.185), x9 (0.183), x8 (-0.162), x5^3 (-0.103), x6 (0.098). Top retained cubic terms: x6 x7^2 (0.209), x6 x8^2 (-0.185), x5^3 (-0.103), x9 x10 x12 (-0.094), x5 x9 x13 (-0.092), x2 x5 x12 (-0.082), x2 x8 x10 (-0.081), x0 x2 x4 (0.080). Coefficients are shown after converting standardized polynomial terms back to their expanded-feature units. If the equation is truncated, omitted active terms are smaller than the displayed dominant terms but are still used by predict(); terms not shown were removed by the sparse selection step. Positive active terms raise the prediction as their expanded value increases; negative active terms lower it. Counterfactual sensitivity is local and additive in this retained expanded basis, so changing an original feature can affect its linear term, squared/cubic powers, and retained interactions that include it.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py` for the next run.
