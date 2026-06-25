# Audit

Status: clean with reservations.

Blocking findings: none.

Reservations:

- Journal evidence is uneven. Representation journals include explicit design
  retrospective material and design handoff links; blind journals satisfy the
  required run-level fields but still contain manual-entry placeholders.
- One failed blind iteration-1 attempt is preserved in the leaderboard:
  `synthetic_regression_blind_iter1_20260620T085522Z` failed with
  `contract_error` before the successful blind iteration-1 rerun.

Clean evidence:

- Active branch: `project/synthetic_regression`.
- Plan and spec align on `DefaultEvaluationSpec`, spec name `default`,
  `cv_rmse_mean`, `minimize`, and `kfold_5_shuffle_seed42`.
- Both LoopRun verify commands passed for iterations 1-5.
- Loop manifests match on baseline commit, baseline candidate hash, dataset
  hash, spec hash, primary metric, and metric direction.
- Both condition ledgers have successful iterations 1-5.
- Blind manifests forbid `model_string`, `interpretability_packet`, and
  `candidate_snapshot`.
- Representation manifests allow representation artifacts for that condition.
