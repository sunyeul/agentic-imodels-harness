# Run 20260620T175142Z

- Project: synthetic_regression
- Candidate module: _toy_imodels_loop_candidate_78e0298ecd80ef05
- Model: poly2_sparse_elasticnetcv
- Notes: ElasticNetCV over standardized degree-2 polynomial features to prune weak quadratic terms while retaining ridge-like shrinkage when useful.
- Evaluation spec: default
- Primary metric: cv_rmse_mean
- Primary metric direction: minimize
- CV strategy: kfold_5_shuffle_seed42
- CV splits: 5
- CV random state: 42
- CV description: 5-fold shuffled KFold over all labeled rows.

## Metrics

- cv_rmse_mean (primary): 0.951539
- cv_rmse_std: 0.057074
- cv_mae_mean: 0.746778
- cv_mae_std: 0.044072
- cv_r2_mean: 0.924890
- cv_r2_std: 0.010804
- Interpretability status: agent_judged
- Agent-judged interpretability score: 0.8000

## Run artifacts

- Fold metrics: projects/synthetic_regression/results/runs/20260620T175142Z/fold_metrics.json
- Diagnostics: projects/synthetic_regression/results/runs/20260620T175142Z/residual_diagnostics.json
- Run metadata: projects/synthetic_regression/results/runs/20260620T175142Z/run_metadata.json
- Candidate snapshot: projects/synthetic_regression/results/runs/20260620T175142Z/candidate_snapshot.py

## Interpretability judgment

- Judgment artifact: projects/synthetic_regression/results/runs/20260620T175142Z/interpretability_judgment.json
- Audit artifact: projects/synthetic_regression/results/runs/20260620T175142Z/interpretability_judgment_audit.json
- Audit status: pass
- Final score: 0.8000
- prediction: 0.7000 - The representation gives the model family, preprocessing, basis expansion, intercept, selected hyperparameters, and top coefficients, but it does not provide all 42 nonzero coefficients or the standardization parameters needed for exact calculation.
- feature_effects: 0.9000 - The model string lists top learned basis terms with signed coefficients and explains sign direction, giving clear relative importance for the dominant effects, though not every nonzero term is shown.
- sensitivity: 0.7500 - Dominant signed coefficients support direction and some relative magnitude reasoning for standardized basis changes, but squared and interaction terms plus omitted coefficients make full local sensitivity incomplete.
- counterfactual: 0.7000 - The text states that positive standardized basis terms increase predictions and negative terms decrease predictions, and lists dominant linear, squared, and interaction terms, but omitted nonzero terms and missing standardization details limit complete input-change answers.
- structure: 0.9500 - The text explicitly describes sparse polynomial ElasticNetCV regression, z-scoring, degree-2 basis construction, basis standardization, CV-selected penalty settings, and the number of linear, quadratic, interaction, and nonzero terms.

## Model string

```text
Sparse polynomial ElasticNetCV regression. Inputs are z-scored, expanded to degree-2 polynomial features (linear, squared, and pairwise interaction terms), then each generated basis column is standardized before elastic-net fitting. The second standardization makes the shrinkage penalty comparable across linear, squared, and interaction columns instead of letting higher-variance basis terms receive different effective shrinkage. Elastic-net alpha and L1 ratio are selected by inner 5-fold CV; small L1 ratios allow nearly ridge-like behavior, while larger ratios can prune weak polynomial terms. Selected alpha: 0.03816. Selected L1 ratio: 1.00. Intercept after input and basis standardization: 0.817. Top learned basis terms by absolute coefficient: x0 (2.344), x1 (-1.605), x3 x4 (1.288), x2 (0.628), x7 (0.462), x2^2 (0.451), x8^2 (0.411), x6 x8 (-0.318), x8 (-0.230), x6 (0.152), x5 (0.145), x9 (0.123). Nonzero basis terms: 42 of 135. Basis includes 15 linear terms, 120 quadratic terms, with 105 cross-feature interaction terms. Positive coefficients increase predictions as their standardized basis term rises; negative coefficients decrease predictions. Counterfactual sensitivity can be read from the dominant linear, squared, and interaction terms above, while smaller omitted terms are either shrunk or set to zero by the elastic-net penalty. The basis stays at degree 2 because the prior degree-3 expansion showed worse cross-validated RMSE.
```

## Next-step hint

Inspect leaderboard movement, then change only `/home/res1235/workspace/agentic-imodels-harness/projects/synthetic_regression/results/loop_runs/bvr_20260621_002_blind_causal_control/workspace/candidate_model.py` for the next run.
