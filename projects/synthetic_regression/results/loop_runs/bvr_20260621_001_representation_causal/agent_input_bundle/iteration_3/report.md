# Run 20260620T155743Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_da49e5f7dbeadc36
- Model: sparse_quadratic_elasticnet_cv
- Notes: ElasticNetCV over standardized first- and second-order polynomial features to retain strong curvature/interactions while pruning weak expanded terms.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 0.948808
- cv_rmse_std: 0.058150
- cv_mae_mean: 0.746068
- cv_mae_std: 0.044157
- cv_r2_mean: 0.925336
- cv_r2_std: 0.010663
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.9360

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T155743Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T155743Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T155743Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T155743Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T155743Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T155743Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.9360
- prediction: 0.9800 - The model string gives an explicit additive prediction equation with intercept, coefficients, squared terms, and pairwise interactions.
- feature_effects: 0.9800 - Feature-level coefficients, signs, and top absolute active terms are listed, including zeroed weak terms.
- sensitivity: 0.9000 - Local additive sensitivity is described and coefficient signs/magnitudes support marginal reasoning, though original-feature sensitivity can depend on retained squared and interaction terms.
- counterfactual: 0.8500 - The text states positive terms raise predictions and negative terms lower them, so increase/decrease counterfactuals are answerable locally, but original-feature changes require combining linear, squared, and interaction effects.
- structure: 0.9700 - The family, sparse ElasticNetCV fit, polynomial expansion, interactions, selected alpha/l1 ratio, active term count, and zeroed terms are explicit.

## Model string

```text
Sparse quadratic ElasticNetCV regression over original features, squared terms, and pairwise interactions. Prediction equation: y = 0.422 + 2.339*x0 + -1.573*x1 + 0.654*x2 + 0.137*x5 + 0.140*x6 + 0.469*x7 + -0.240*x8 + 0.127*x9 + -0.008*x13 + -0.021*x0^2 + -0.026*x0 x2 + 0.032*x0 x7 + -0.010*x0 x11 + 0.008*x0 x12 + 0.046*x1 x2 + 0.021*x1 x5 + 0.293*x2^2 + -0.057*x2 x5 + 0.008*x2 x8 + -0.004*x2 x9 + 0.022*x2 x12 + 1.272*x3 x4 + -0.002*x3 x14 + -0.006*x4 x8 + -0.007*x4 x14 + 0.041*x5 x7 + 0.003*x5 x9 + 0.001*x6 x7 + -0.298*x6 x8 + 0.017*x6 x9 + -0.014*x6 x13 + 0.029*x7 x8 + 0.020*x7 x9 + -0.002*x7 x12 + 0.011*x7 x14 + 0.270*x8^2 + -0.007*x8 x13 + -0.028*x9^2 + -0.025*x9 x12 + 0.018*x9 x14 + 0.010*x10 x12. Selected ElasticNet alpha: 0.04062; selected l1_ratio: 1.00. Active expanded terms: 41; zeroed weak expanded terms: 94. Top absolute active coefficient terms: x0 (2.339), x1 (-1.573), x3 x4 (1.272), x2 (0.654), x7 (0.469), x6 x8 (-0.298), x2^2 (0.293), x8^2 (0.270), x8 (-0.240), x6 (0.140). Coefficients are shown after converting standardized polynomial terms back to their expanded-feature units; terms not shown have exact zero coefficients from the sparse penalty. Positive active terms raise the prediction as their expanded value increases; negative active terms lower it. Counterfactual sensitivity is local and additive in this retained expanded basis, so changing an original feature can affect its linear term, squared term, and retained pairwise interaction terms that include it.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_001_representation_causal/workspace/candidate_model.py` for the next run.
