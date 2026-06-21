# synthetic_regression blind vs representation 실험 계획

이 문서는 `synthetic_regression` 프로젝트에서 `blind` 조건과
`representation` 조건을 오염 없이 비교하기 위한 운영 계획과 진행 상황
트래커다. 목적은 여러 Codex 세션에 걸친 실험 진행 중 각 조건의 정보
경계가 섞이지 않도록 하는 것이다.

## 실험 목적

- Project: `synthetic_regression`
- Spec: `DefaultEvaluationSpec`
- Primary metric: `cv_rmse_mean`
- Direction: minimize
- CV: `kfold_5_shuffle_seed42`, 5-fold shuffled KFold
- 비교 조건:
  - `blind`: model string, interpretability packet, candidate snapshot을 보지 않음
  - `representation`: model string과 interpretability packet을 포함한 표현 정보를 봄
- 비교 질문: representation artifact가 주어질 때 같은 budget 안에서 더 좋은
  candidate 탐색 효율을 보이는가?

## 고정 원칙

- 장기 실험 브랜치는 `project/synthetic_regression` 하나만 사용한다.
- 조건 격리는 브랜치가 아니라 `loop_run_id`, loop workspace,
  `agent_input_bundle/iteration_<n>/input_manifest.json`로 보장한다.
- 두 loop run은 같은 baseline commit에서 초기화한다.
- 각 조건의 설계 세션은 자기 조건의 `input_manifest.json`에 listed 된 파일만 본다.
- candidate 수정은 loop manifest에 기록된
  `candidate_workspace_path`의 `candidate_model.py`에만 한다.
- fixed harness, spec, data, scoring code는 변경하지 않는다.
- candidate code는 `fit`, `predict`, `__str__`에서 filesystem/network I/O를 하지 않는다.
- raw public competition files, hidden targets, generator/oracle 구조, metadata oracle
  fields는 설계 중 열람하거나 인코딩하지 않는다.
- 결과 비교 전 각 loop run에 대해 `verify`를 실행해 조건별 forbidden artifact가
  bundle에 섞이지 않았는지 확인한다.

## 세션 전환 정책

세션은 조건별 정보 경계를 지키기 위해 다음 타이밍에만 전환한다.

1. 공용 setup 세션
   - 브랜치 생성, 두 loop run 초기화, 초기 manifest 확인까지만 수행한다.
   - candidate 설계나 이전 run artifact 해석은 하지 않는다.
   - 이 세션에서 생성된 loop id와 workspace path만 진행표에 기록한다.
   - `agentic-imodels-toy-experiment` skill은 실험 guardrail과 project branch
     규칙에만 적용하고, candidate iteration workflow의 leaderboard/report 해석
     단계는 이 setup 세션에서 수행하지 않는다.

2. 조건별 iteration 시작 시점
   - 각 iteration은 반드시 먼저 공용 setup/관리 세션 또는 해당 조건 세션에서
     `prepare`를 실행해 `input_manifest.json`을 만든 뒤 시작한다.
   - `prepare`가 끝나면 해당 조건 전용 세션으로 전환한다.
   - 전환 후 조건 세션은 그 iteration의 manifest와 manifest에 listed 된 파일만 읽는다.

3. candidate 편집 후 평가 시점
   - 조건 세션은 자기 loop workspace candidate를 편집한 뒤 `record`로 평가를 실행한다.
   - `record` 결과의 run id와 primary score를 진행표에 기록한다.
   - 평가가 끝나면 같은 조건의 다음 iteration을 바로 설계하지 않는다.

4. iteration 사이 전환 시점
   - 한 조건의 `record`가 끝난 뒤에는 관리 세션으로 돌아와 진행표를 갱신한다.
   - 다음으로 실행할 조건을 선택하고, 그 조건의 다음 `prepare`를 만든 뒤 해당
     조건 세션으로 전환한다.
   - 권장 순서는 조건 간 시간/맥락 편향을 줄이기 위해 교차 진행이다:
     `blind i` -> `representation i` -> `blind i+1` -> `representation i+1`.

5. 최종 비교 시점
   - 두 조건 모두 budget을 소진하거나 중단 기준에 도달한 뒤에만 비교 세션으로 전환한다.
   - 비교 세션에서는 `verify`와 `compare_loop_runs`만 수행하고, candidate 개선 설계는 하지 않는다.

## 권장 명령

브랜치 생성:

```bash
rtk git switch -c project/synthetic_regression
```

loop run 초기화:

```bash
rtk uv run python -m toy_imodels.loop_run init \
  --project-module projects.synthetic_regression:synthetic_regression_project \
  --condition blind \
  --budget 5 \
  --agent-model codex \
  --loop-run-id synthetic_regression_blind \
  --results-dir projects/synthetic_regression/results
```

```bash
rtk uv run python -m toy_imodels.loop_run init \
  --project-module projects.synthetic_regression:synthetic_regression_project \
  --condition representation \
  --budget 5 \
  --agent-model codex \
  --loop-run-id synthetic_regression_representation \
  --results-dir projects/synthetic_regression/results
```

iteration 준비:

```bash
rtk uv run python -m toy_imodels.loop_run prepare synthetic_regression_blind \
  --iteration 1 \
  --results-dir projects/synthetic_regression/results
```

```bash
rtk uv run python -m toy_imodels.loop_run prepare synthetic_regression_representation \
  --iteration 1 \
  --results-dir projects/synthetic_regression/results
```

iteration 평가 기록:

```bash
rtk uv run python -m toy_imodels.loop_run record synthetic_regression_blind \
  --iteration 1 \
  --project-module projects.synthetic_regression:synthetic_regression_project \
  --results-dir projects/synthetic_regression/results
```

```bash
rtk uv run python -m toy_imodels.loop_run record synthetic_regression_representation \
  --iteration 1 \
  --project-module projects.synthetic_regression:synthetic_regression_project \
  --results-dir projects/synthetic_regression/results
```

조건 격리 검증:

```bash
rtk uv run python -m toy_imodels.loop_run verify synthetic_regression_blind \
  --results-dir projects/synthetic_regression/results
```

```bash
rtk uv run python -m toy_imodels.loop_run verify synthetic_regression_representation \
  --results-dir projects/synthetic_regression/results
```

최종 비교:

```bash
rtk uv run python -m toy_imodels.compare_loop_runs \
  --left synthetic_regression_blind \
  --right synthetic_regression_representation \
  --results-dir projects/synthetic_regression/results
```

## 진행 상황

| 항목                               | 상태   | 근거/다음 행동                                      |
| ---------------------------------- | ------ | --------------------------------------------------- |
| 실험 계획 문서                     | 완료   | 이 문서                                             |
| 실험 브랜치                        | 완료   | `project/synthetic_regression` 생성 및 checkout     |
| baseline leaderboard               | 완료   | `projects/synthetic_regression/results/leaderboard.csv` 존재 확인 |
| blind loop init                    | 완료   | `synthetic_regression_blind`, workspace: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/workspace/candidate_model.py` |
| representation loop init           | 완료   | `synthetic_regression_representation`, workspace: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/workspace/candidate_model.py` |
| blind iteration 1 prepare          | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_1/input_manifest.json` 생성 |
| blind iteration 2 prepare          | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_2/input_manifest.json` 생성 |
| blind iteration 3 prepare          | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_3/input_manifest.json` 생성 |
| blind iteration 4 prepare          | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_4/input_manifest.json` 생성 |
| blind iteration 5 prepare          | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_5/input_manifest.json` 생성 |
| representation iteration 1 prepare | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_1/input_manifest.json` 생성 |
| representation iteration 2 prepare | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_2/input_manifest.json` 생성 |
| representation iteration 3 prepare | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_3/input_manifest.json` 생성 |
| representation iteration 4 prepare | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_4/input_manifest.json` 생성 |
| representation iteration 5 prepare | 완료   | `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_5/input_manifest.json` 생성 |
| blind iterations 1-5 record        | 완료   | best: `synthetic_regression_blind_iter5_20260620T090107Z`, `cv_rmse_mean=0.9488791094313402` |
| representation iterations 1-5 record | 완료 | best: `synthetic_regression_representation_iter5_20260620T092729Z`, `cv_rmse_mean=0.8499420743055246` |
| condition verify                   | 완료   | blind verify PASS; representation verify PASS      |
| final comparison                   | 완료   | representation 우세: best `0.8499420743055246` vs blind best `0.9488791094313402`; predictive-only 비교 |

Setup 확인:

- baseline_commit: `1b33c64a5eb71ce1b9704b9b00524e018e7cc6ab`
- 두 loop manifest의 `baseline_commit`, `baseline_candidate_sha256`,
  `dataset_sha256`, `spec_sha256`, `primary_metric`,
  `primary_metric_direction` 일치 확인 완료
- primary_metric: `cv_rmse_mean`
- primary_metric_direction: `minimize`

Final comparison 결과:

- Command: `rtk uv run python -m toy_imodels.compare_loop_runs --left synthetic_regression_blind --right synthetic_regression_representation --results-dir projects/synthetic_regression/results`
- blind best_score_at_budget: `0.9488791094313402`
- representation best_score_at_budget: `0.8499420743055246`
- blind gain_per_iteration: `0.018251488450421505`
- representation gain_per_iteration: `0.036952175682801514`
- blind failed_edit_rate: `0.0`; representation failed_edit_rate: `0.0`
- blind regression_rate: `0.5`; representation regression_rate: `0.75`
- iterations_to_target: `null` for both conditions
- Interpretability status is still `pending_agent_judgment`; this is a predictive-only final comparison, not a judged interpretability-frontier comparison.

## Iteration 로그

| 조건           | iteration | 세션                | manifest | run id | primary score | 상태   | 메모                       |
| -------------- | --------: | ------------------- | -------- | ------ | ------------: | ------ | -------------------------- |
| blind          |         1 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_1/input_manifest.json` | `synthetic_regression_blind_iter1_20260620T085522Z` | - | 실패 | contract_error; traceback: `projects/synthetic_regression/results/runs/synthetic_regression_blind_iter1_20260620T085522Z/error_traceback.txt` |
| blind          |         1 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_1/input_manifest.json` | `synthetic_regression_blind_iter1_20260620T085539Z` | 1.0401365516834478 | 완료 | degree-2 polynomial RidgeCV |
| representation |         1 | representation 전용 | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_1/input_manifest.json` | `synthetic_regression_representation_iter1_20260620T091538Z` | 1.0347029527195322 | 완료 | quadratic Ridge |
| blind          |         2 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_2/input_manifest.json` | `synthetic_regression_blind_iter2_20260620T085714Z` | 1.6285194233669926 | 완료 | degree-3 polynomial RidgeCV |
| representation |         2 | representation 전용 | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_2/input_manifest.json` | `synthetic_regression_representation_iter2_20260620T091949Z` | 1.688898863280491 | 완료 | cubic Ridge |
| blind          |         3 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_3/input_manifest.json` | `synthetic_regression_blind_iter3_20260620T085821Z` | 1.338933474619829 | 완료 | histogram gradient boosting |
| representation |         3 | representation 전용 | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_3/input_manifest.json` | `synthetic_regression_representation_iter3_20260620T092157Z` | 1.0485992519240621 | 완료 | quadratic Ridge with stronger shrinkage |
| blind          |         4 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_4/input_manifest.json` | `synthetic_regression_blind_iter4_20260620T085933Z` | 0.9525695919472372 | 완료 | degree-2 polynomial ElasticNetCV |
| representation |         4 | representation 전용 | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_4/input_manifest.json` | `synthetic_regression_representation_iter4_20260620T092425Z` | 1.0379832878654451 | 완료 | quadratic RidgeCV |
| blind          |         5 | blind 전용          | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_blind/agent_input_bundle/iteration_5/input_manifest.json` | `synthetic_regression_blind_iter5_20260620T090107Z` | 0.9488791094313402 | 완료 | stacked degree-2 RidgeCV/ElasticNetCV; current best |
| representation |         5 | representation 전용 | 생성됨: `projects/synthetic_regression/results/loop_runs/synthetic_regression_representation/agent_input_bundle/iteration_5/input_manifest.json` | `synthetic_regression_representation_iter5_20260620T092729Z` | 0.8499420743055246 | 완료 | quadratic hinge RidgeCV; current best |

## 갱신 규칙

- 공용 setup 세션은 브랜치, loop init, prepare, verify, compare 상태만 갱신한다.
- 조건별 세션은 자기 조건의 row만 갱신한다.
- 조건별 세션은 다른 조건 row의 manifest, report, run artifact 내용을 열람하지 않는다.
- `record` 완료 시 최소한 run id, primary score, status를 기록한다.
- 실패 run도 지우지 않고 상태와 traceback artifact 유무를 기록한다.
- retained candidate를 남기기로 결정한 경우 한 modeling hypothesis당 하나의 commit만 만든다.
