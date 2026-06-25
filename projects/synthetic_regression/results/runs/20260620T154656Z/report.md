# Run 20260620T154656Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_da49e5f7dbeadc36
- Model: quadratic_ridge_cv
- Notes: RidgeCV over standardized first- and second-order polynomial features to capture smooth curvature and pairwise interactions.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 1.034703
- cv_rmse_std: 0.061089
- cv_mae_mean: 0.819039
- cv_mae_std: 0.040754
- cv_r2_mean: 0.911023
- cv_r2_std: 0.013733
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.9260

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T154656Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T154656Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T154656Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T154656Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T154656Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T154656Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.9260
- prediction: 0.9800 - A complete prediction equation is provided with intercept, original features, squared terms, pairwise interactions, and coefficients.
- feature_effects: 0.9500 - The model string gives signed coefficients for all expanded terms and lists the top absolute coefficient terms, making major directions and relative effects clear.
- sensitivity: 0.9000 - Coefficient signs and magnitudes support local additive sensitivity reasoning in the expanded basis, while original-feature sensitivity is more complex because each feature participates in linear, squared, and interaction terms.
- counterfactual: 0.8500 - The text states positive expanded terms raise predictions and negative terms lower them, and explains local additive counterfactual sensitivity, but exact original-feature counterfactuals require accounting for squared and interaction terms at a specific input.
- structure: 0.9500 - The model family, RidgeCV alpha, quadratic feature expansion, squared terms, pairwise interactions, and expanded-feature coefficient units are explicit.

## Model string

```text
Quadratic RidgeCV regression over original features, squared terms, and pairwise interactions. Prediction equation: y = 0.499 + 2.331*x0 + -1.580*x1 + 0.691*x2 + -0.005*x3 + -0.046*x4 + 0.186*x5 + 0.171*x6 + 0.499*x7 + -0.288*x8 + 0.217*x9 + -0.007*x10 + -0.014*x11 + 0.002*x12 + -0.048*x13 + 0.018*x14 + -0.117*x0^2 + 0.002*x0 x1 + -0.065*x0 x2 + 0.050*x0 x3 + 0.006*x0 x4 + 0.018*x0 x5 + 0.069*x0 x6 + 0.089*x0 x7 + -0.090*x0 x8 + 0.009*x0 x9 + 0.200*x0 x10 + -0.030*x0 x11 + 0.057*x0 x12 + 0.003*x0 x13 + -0.047*x0 x14 + 0.002*x1^2 + 0.063*x1 x2 + 0.011*x1 x3 + -0.044*x1 x4 + 0.053*x1 x5 + 0.009*x1 x6 + 0.041*x1 x7 + 0.017*x1 x8 + 0.029*x1 x9 + -0.041*x1 x10 + -0.037*x1 x11 + -0.028*x1 x12 + 0.005*x1 x13 + 0.064*x1 x14 + 0.325*x2^2 + -0.022*x2 x3 + -0.010*x2 x4 + -0.089*x2 x5 + 0.027*x2 x6 + 0.031*x2 x7 + 0.056*x2 x8 + -0.049*x2 x9 + 0.007*x2 x10 + -0.029*x2 x11 + 0.069*x2 x12 + -0.030*x2 x13 + -0.021*x2 x14 + 0.051*x3^2 + 1.274*x3 x4 + -0.024*x3 x5 + 0.030*x3 x6 + 0.003*x3 x7 + 0.019*x3 x8 + 0.059*x3 x9 + -0.111*x3 x10 + 0.064*x3 x11 + 0.019*x3 x12 + -0.001*x3 x13 + -0.068*x3 x14 + 0.009*x4^2 + -0.012*x4 x5 + -0.003*x4 x6 + 0.025*x4 x7 + -0.045*x4 x8 + -0.017*x4 x9 + -0.044*x4 x10 + -0.036*x4 x11 + 0.021*x4 x12 + -0.013*x4 x13 + -0.054*x4 x14 + -0.020*x5^2 + 0.014*x5 x6 + 0.056*x5 x7 + -0.000*x5 x8 + 0.057*x5 x9 + -0.010*x5 x10 + -0.019*x5 x11 + 0.020*x5 x12 + -0.044*x5 x13 + -0.002*x5 x14 + -0.030*x6^2 + 0.013*x6 x7 + -0.353*x6 x8 + 0.044*x6 x9 + -0.048*x6 x10 + 0.026*x6 x11 + -0.028*x6 x12 + -0.058*x6 x13 + 0.024*x6 x14 + -0.023*x7^2 + 0.077*x7 x8 + 0.050*x7 x9 + -0.033*x7 x10 + 0.007*x7 x11 + -0.056*x7 x12 + -0.042*x7 x13 + 0.066*x7 x14 + 0.297*x8^2 + -0.010*x8 x9 + 0.134*x8 x10 + 0.030*x8 x11 + -0.011*x8 x12 + -0.041*x8 x13 + -0.002*x8 x14 + -0.061*x9^2 + 0.061*x9 x10 + 0.034*x9 x11 + -0.078*x9 x12 + 0.016*x9 x13 + 0.083*x9 x14 + -0.142*x10^2 + -0.079*x10 x11 + 0.057*x10 x12 + -0.026*x10 x13 + 0.023*x10 x14 + 0.000*x11^2 + 0.051*x11 x12 + -0.024*x11 x13 + -0.026*x11 x14 + -0.022*x12^2 + -0.042*x12 x13 + 0.019*x12 x14 + 0.011*x13^2 + -0.017*x13 x14 + 0.010*x14^2. Selected ridge alpha: 10. Top absolute coefficient terms: x0 (2.331), x1 (-1.580), x3 x4 (1.274), x2 (0.691), x7 (0.499), x6 x8 (-0.353), x2^2 (0.325), x8^2 (0.297). Coefficients are shown after converting standardized polynomial terms back to their expanded-feature units. Positive terms raise the prediction as their expanded value increases; negative terms lower it. Counterfactual sensitivity is local and additive in this expanded basis, so changing an original feature can affect its linear term, squared term, and all pairwise interaction terms that include it.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py` for the next run.
