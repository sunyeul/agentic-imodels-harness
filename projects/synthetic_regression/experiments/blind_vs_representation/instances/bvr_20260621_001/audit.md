# Audit

Status: clean with reservations.

Blocking findings: none.

Independent audit:

- A separate experiment-auditor pass found no blocking findings and returned
  clean with reservations.
- Residual reservations were limited to intentionally uninspected candidate
  source contents, design rationale artifacts, journals, reports, fold metrics,
  and interpretability packets because this session was restricted to
  setup/management/audit/comparison scope.

Cleanup performed:

- Removed only the orphan main-leaderboard row for run `20260620T164931Z`.
- The removed row belonged to `bvr_20260621_002_blind_causal_control`,
  `iteration_index` 1, had the same `cv_rmse_mean` as recorded run
  `20260620T170011Z`, was absent from the blind `iterations.csv`, and had blank
  baseline metadata and no interpretability judgment.
- Recorded blind iteration rows 1-5 were not altered.

Readiness evidence:

- `rtk uv run python -m toy_imodels.loop_run verify bvr_20260621_001_representation_causal --results-dir projects/synthetic_regression/results`
  exited 0 and reported only `PASS` findings.
- `rtk uv run python -m toy_imodels.loop_run verify bvr_20260621_002_blind_causal_control --results-dir projects/synthetic_regression/results`
  exited 0 before and after cleanup and reported only `PASS` findings.
- Both loop manifests match on `project_id`, `primary_metric`,
  `primary_metric_direction`, `budget`, `agent_model`, baseline commit,
  baseline candidate hash, dataset hash, and spec hash.
- Run metadata for all ten recorded iterations reports spec name `default`,
  primary metric `cv_rmse_mean`, direction `minimize`, agent model `codex`, and
  budget 5.
- Each LoopRun `iterations.csv` has exactly five successful iterations.
- `20260620T164931Z` is absent from the main
  `projects/synthetic_regression/results/leaderboard.csv` after cleanup.

Hash reservation:

- `iterations.csv` column `candidate_sha256` records the loop workspace
  candidate file hash at record time.
- `run_metadata.json` field `candidate_source_sha256` records the persisted
  `CandidateModel` source snapshot hash.
- For all ten recorded iterations, each `candidate_source_sha256` matches the
  SHA-256 of its `candidate_snapshot.py`. These values are expected to differ
  from the `iterations.csv` workspace hash.

Official comparison:

- Command:
  `rtk uv run python -m toy_imodels.compare_loop_runs --left bvr_20260621_001_representation_causal --right bvr_20260621_002_blind_causal_control --results-dir projects/synthetic_regression/results`
- Exit status: 0.
- Source of truth: LoopRun `iterations.csv` through
  `toy_imodels.compare_loop_runs`.
- The orphan row did not affect the official comparison because
  `compare_loop_runs` reads the LoopRun iteration ledgers, not arbitrary
  leaderboard rows.

Reservations:

- This is one matched pair, so it is evidence for this repeated trial only, not
  a population-level estimate.
- Historical `agent_input_bundle` leaderboard snapshots may still contain the
  previously available orphan row because those bundles are immutable design
  inputs. They are not used by the official final comparison.
