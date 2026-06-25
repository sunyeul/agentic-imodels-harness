# Run 20260620T161400Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_da49e5f7dbeadc36
- Model: sparse_cubic_elasticnet_cv
- Notes: ElasticNetCV over standardized polynomial features up to degree 3 to test whether sparse higher-order curvature/interactions reduce RMSE beyond the retained quadratic structure.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 0.849184
- cv_rmse_std: 0.039659
- cv_mae_mean: 0.673879
- cv_mae_std: 0.026376
- cv_r2_mean: 0.940161
- cv_r2_std: 0.007908
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.9000

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T161400Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T161400Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T161400Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T161400Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T161400Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T161400Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.9000
- prediction: 0.9500 - The model string gives an explicit retained-term prediction equation with intercept and coefficients, plus notes on zeroed terms and predict-time use.
- feature_effects: 0.9500 - Feature signs, coefficients, and top absolute active terms are shown, including dominant linear, interaction, quadratic, and cubic effects.
- sensitivity: 0.8500 - The representation explains local additive sensitivity in the expanded basis, though original-feature sensitivity is complicated by powers and interactions involving the changed feature.
- counterfactual: 0.8000 - The text supports local increase/decrease counterfactual reasoning through signed retained terms, but exact original-feature counterfactuals require the current feature values because nonlinear powers and interactions also change.
- structure: 0.9500 - The model family, cubic polynomial feature expansion, sparsity, active term counts by degree, selected ElasticNet settings, and retained interaction structure are explicit.

## Model string

```text
Sparse cubic ElasticNetCV regression over original features, squared terms, pairwise interactions, and third-order polynomial terms. Prediction equation in retained expanded terms: y = 0.386 + 2.349*x0 + -1.522*x1 + 0.403*x2 + 0.149*x5 + 0.082*x6 + 0.420*x7 + -0.161*x8 + 0.128*x9 + -0.004*x0^2 + -0.010*x0 x2 + 0.024*x0 x7 + -0.011*x0 x11 + 0.033*x1 x2 + 0.317*x2^2 + 1.269*x3 x4 + -0.005*x3 x14 + -0.006*x4 x11 + 0.027*x5 x7 + -0.264*x6 x8 + -0.009*x6 x13 + -0.023*x7 x12 + 0.265*x8^2 + -0.033*x9^2 + 0.004*x9 x14 + 0.007*x10 x12 + 0.034*x0^2 x2 + 0.009*x0^2 x6 + 0.034*x0 x2 x4 + 0.003*x0 x4^2 + -0.003*x1^3 + 0.007*x1^2 x5 + -0.004*x1 x2^2 + -0.026*x1 x4^2 + -0.003*x1 x12^2 + 0.053*x2^3 + 0.000*x2^2 x13 + -0.036*x2 x5 x12 + 0.003*x2 x6^2 + -0.002*x2 x7 x13 + 0.019*x2 x8^2 + -0.018*x2 x8 x10 + 0.034*x2 x10^2 + -0.006*x3^2 x7 + -0.025*x3 x4 x8 + -0.006*x3 x6 x14 + 0.006*x3 x7 x14 + -0.007*x3 x8^2 + 0.002*x3 x10^2 + 0.015*x4^2 x5 + 0.003*x4^2 x6 + -0.015*x4^2 x8 + 0.027*x4 x7 x10 + -0.037*x5^3 + -0.000*x5 x6 x8 + 0.004*x5 x7^2 + 0.024*x5 x8^2 + -0.006*x5 x9 x13 + 0.025*x5 x12^2 + 0.004*x5 x14^2 + 0.011*x6^2 x7 + -0.037*x6^2 x8 + 0.195*x6 x7^2 + 0.007*x6 x7 x12 + -0.129*x6 x8^2 + 0.003*x6 x10^2 + 0.024*x6 x12 x14 + -0.013*x7^2 x13 + -0.063*x7 x9 x12 + 0.005*x7 x10^2 + -0.007*x8^3 + 0.007*x8 x10 x13 + -0.011*x9 x10 x12. Selected ElasticNet alpha: 0.04344; selected l1_ratio: 1.00. Active expanded terms: 72; zeroed weak expanded terms: 743; active terms by degree: linear: 8, quadratic: 17, cubic: 47. Top absolute active coefficient terms: x0 (2.349), x1 (-1.522), x3 x4 (1.269), x7 (0.420), x2 (0.403), x2^2 (0.317), x8^2 (0.265), x6 x8 (-0.264), x6 x7^2 (0.195), x8 (-0.161), x5 (0.149), x6 x8^2 (-0.129), x9 (0.128), x6 (0.082), x7 x9 x12 (-0.063). Top retained cubic terms: x6 x7^2 (0.195), x6 x8^2 (-0.129), x7 x9 x12 (-0.063), x2^3 (0.053), x6^2 x8 (-0.037), x5^3 (-0.037), x2 x5 x12 (-0.036), x2 x10^2 (0.034). Coefficients are shown after converting standardized polynomial terms back to their expanded-feature units. If the equation is truncated, omitted active terms are smaller than the displayed dominant terms but are still used by predict(); terms not shown because they were zeroed have exact zero coefficients from the sparse penalty. Positive active terms raise the prediction as their expanded value increases; negative active terms lower it. Counterfactual sensitivity is local and additive in this retained expanded basis, so changing an original feature can affect its linear term, squared/cubic powers, and retained interactions that include it.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py` for the next run.
