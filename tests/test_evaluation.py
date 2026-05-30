import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.model_selection import KFold

from projects.synthetic_regression import (
    DefaultEvaluationSpec,
    synthetic_regression_project,
)
from projects.synthetic_regression.datasets import DEFAULT_DATA_DIR, FEATURE_COLUMNS
from projects.synthetic_regression.project import DEFAULT_RESULTS_DIR
from toy_imodels.audit import verify_experiment_run
from toy_imodels.core.evaluation import apply_interpretability_judgment, run_experiment
from toy_imodels.core.project import EvaluationData, Project
from toy_imodels.interpretability import (
    RUBRIC_VERSION,
    audit_interpretability_judgment,
    build_interpretability_packet,
    validate_interpretability_judgment,
)
from toy_imodels.leaderboard import append_result, update_result
from toy_imodels.provenance import sha256_file
from toy_imodels.spec import CVStrategy, EvaluationSpec

DEFAULT_CANDIDATE_MODULE = "projects.synthetic_regression.experiments.candidate_model"


def _copy_competition_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(DEFAULT_DATA_DIR, data_dir, dirs_exist_ok=True)
    return data_dir


def _unused_submission_writer(
    _path: Path, _run_id: str, _data: EvaluationData, _predictions: Any
) -> Path:
    pytest.fail("write_submission should not be called")


class AccuracyEvaluationSpec(EvaluationSpec):
    def __init__(self):
        super().__init__(
            name="accuracy",
            primary_metric="accuracy",
            primary_metric_direction="maximize",
            cv_strategy=CVStrategy(
                name="kfold_3_no_shuffle",
                n_splits=3,
                random_state=None,
                description="3-fold KFold for classification testing.",
                splitter=KFold(n_splits=3, shuffle=False),
            ),
        )

    def score_predictions(self, y_true, y_pred):
        return {
            "accuracy": float(
                (pd.Series(y_pred).to_numpy() == y_true.to_numpy()).mean()
            )
        }

    def aggregate_fold_scores(self, fold_scores):
        values = [score["accuracy"] for score in fold_scores]
        return {"accuracy": sum(values) / len(values)}


def _write_fake_submission(
    submissions_dir: Path, run_id: str, _data: EvaluationData, predictions: Any
) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    path = submissions_dir / f"{run_id}.txt"
    path.write_text(",".join(str(int(value)) for value in predictions) + "\n")
    return path


def _write_valid_interpretability_judgment(tmp_path: Path, run_id: str) -> Path:
    judgment_path = tmp_path / f"{run_id}-judgment.json"
    judgment_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "rubric_version": RUBRIC_VERSION,
                "dimension_scores": {
                    "prediction": {
                        "score": 1.0,
                        "rationale": "The model string includes an equation.",
                    },
                    "feature_effects": {
                        "score": 0.8,
                        "rationale": "Top coefficients are listed.",
                    },
                    "sensitivity": {
                        "score": 0.6,
                        "rationale": "Coefficient magnitudes support local changes.",
                    },
                    "counterfactual": {
                        "score": 0.4,
                        "rationale": "Counterfactuals can be inferred from signs.",
                    },
                    "structure": {
                        "score": 0.7,
                        "rationale": "The model family is explicitly linear ridge.",
                    },
                },
                "interpretability_score": 0.7,
            }
        )
        + "\n"
    )
    return judgment_path


def test_run_experiment_evaluates_baseline_and_writes_artifacts(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(
        project=project,
        run_id="test-run",
    )

    assert result.status == "success"
    assert result.project_id == "synthetic_regression"
    assert result.candidate_module == DEFAULT_CANDIDATE_MODULE
    assert Path(result.fold_metrics_path).exists()
    diagnostics_path = result.artifact_paths["diagnostics_path"]
    submission_path = result.artifact_paths["submission_path"]
    assert Path(diagnostics_path).exists()
    assert Path(result.run_metadata_path).exists()
    assert Path(result.candidate_snapshot_path).exists()
    assert result.result_metrics["cv_rmse_mean"] > 0
    assert result.result_metrics["cv_rmse_std"] >= 0
    assert result.result_metrics["cv_mae_mean"] > 0
    assert result.result_metrics["cv_mae_std"] >= 0
    assert -1 <= result.result_metrics["cv_r2_mean"] <= 1
    assert result.result_metrics["cv_r2_std"] >= 0
    assert math.isnan(result.interpretability_score)

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.shape[0] == 1
    assert leaderboard.loc[0, "run_id"] == "test-run"
    assert leaderboard.loc[0, "status"] == "success"
    assert pd.isna(leaderboard.loc[0, "interpretability_score"])
    for column in (
        "project_id",
        "candidate_module",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_mae_mean",
        "cv_mae_std",
        "cv_r2_mean",
        "cv_r2_std",
        "spec_name",
        "primary_metric",
        "primary_metric_direction",
        "cv_strategy_name",
        "cv_n_splits",
        "cv_random_state",
        "fold_metrics_path",
        "run_metadata_path",
        "candidate_snapshot_path",
        "diagnostics_path",
        "error_traceback_path",
    ):
        assert column in leaderboard.columns
    assert leaderboard.loc[0, "project_id"] == "synthetic_regression"
    assert leaderboard.loc[0, "candidate_module"] == DEFAULT_CANDIDATE_MODULE
    assert leaderboard.loc[0, "spec_name"] == "default"
    assert leaderboard.loc[0, "primary_metric"] == "cv_rmse_mean"
    assert leaderboard.loc[0, "primary_metric_direction"] == "minimize"
    assert leaderboard.loc[0, "cv_rmse_mean"] > 0
    assert leaderboard.loc[0, "cv_rmse_std"] >= 0
    assert leaderboard.loc[0, "cv_strategy_name"] == "kfold_5_shuffle_seed42"
    assert leaderboard.loc[0, "cv_n_splits"] == 5
    assert leaderboard.loc[0, "cv_random_state"] == 42
    assert leaderboard.loc[0, "fold_metrics_path"] == result.fold_metrics_path
    assert leaderboard.loc[0, "diagnostics_path"] == diagnostics_path
    assert leaderboard.loc[0, "run_metadata_path"] == result.run_metadata_path
    assert leaderboard.loc[0, "candidate_snapshot_path"] == (
        result.candidate_snapshot_path
    )

    assert Path(submission_path).exists()
    assert Path(result.report_path).exists()
    report_text = Path(result.report_path).read_text()
    assert "Project: synthetic_regression" in report_text
    assert f"Candidate module: {DEFAULT_CANDIDATE_MODULE}" in report_text
    assert "Evaluation spec: default" in report_text
    assert "Primary metric: cv_rmse_mean" in report_text
    assert "Primary metric direction: minimize" in report_text
    assert "CV strategy: kfold_5_shuffle_seed42" in report_text
    assert "cv_rmse_mean (primary)" in report_text
    assert "Run artifacts" in report_text
    assert "Fold metrics:" in report_text
    assert "Diagnostics:" in report_text
    assert "Run metadata:" in report_text
    assert "Candidate snapshot:" in report_text
    assert "Interpretability status: pending_agent_judgment" in report_text
    assert "Interpretability score: NaN" in report_text
    assert "Model string" in report_text

    fold_metrics = json.loads(Path(result.fold_metrics_path).read_text())
    assert fold_metrics["run_id"] == "test-run"
    assert fold_metrics["project_id"] == "synthetic_regression"
    assert len(fold_metrics["folds"]) == 5
    assert fold_metrics["folds"][0]["fold_index"] == 0
    assert fold_metrics["folds"][0]["train_rows"] > 0
    assert fold_metrics["folds"][0]["valid_rows"] > 0
    assert {"rmse", "mae", "r2"} <= set(fold_metrics["folds"][0]["metrics"])

    run_metadata = json.loads(Path(result.run_metadata_path).read_text())
    assert run_metadata["run_id"] == "test-run"
    assert run_metadata["project_id"] == "synthetic_regression"
    assert run_metadata["candidate_module"] == DEFAULT_CANDIDATE_MODULE
    assert run_metadata["candidate_source_sha256"]
    assert run_metadata["candidate_source_sha256"] == sha256_file(
        result.candidate_snapshot_path
    )
    assert "git_commit" in run_metadata
    assert "git_parent_commit" in run_metadata
    assert isinstance(run_metadata["git_dirty"], bool)
    assert run_metadata["spec_name"] == "default"
    assert run_metadata["primary_metric"] == "cv_rmse_mean"
    assert run_metadata["primary_metric_direction"] == "minimize"
    assert run_metadata["spec"]["spec_name"] == "default"
    assert run_metadata["spec"]["primary_metric"] == "cv_rmse_mean"
    assert run_metadata["spec"]["primary_metric_direction"] == "minimize"
    assert run_metadata["spec"]["cv_strategy_name"] == "kfold_5_shuffle_seed42"
    assert run_metadata["artifacts"]["report_path"] == result.report_path
    assert run_metadata["artifacts"]["submission_path"] == submission_path
    assert run_metadata["artifacts"]["fold_metrics_path"] == (result.fold_metrics_path)
    assert run_metadata["artifacts"]["diagnostics_path"] == diagnostics_path
    assert run_metadata["artifacts"]["candidate_snapshot_path"] == (
        result.candidate_snapshot_path
    )

    residual_diagnostics = json.loads(Path(diagnostics_path).read_text())
    assert residual_diagnostics["run_id"] == "test-run"
    assert residual_diagnostics["project_id"] == "synthetic_regression"
    assert "oracle" in residual_diagnostics["diagnostic_note"]
    assert {
        "residual_summary",
        "top_features_by_residual_pattern",
        "feature_diagnostics",
    } <= set(residual_diagnostics)
    assert len(residual_diagnostics["feature_diagnostics"]) == len(FEATURE_COLUMNS)
    assert residual_diagnostics["top_features_by_residual_pattern"]
    first_feature = residual_diagnostics["feature_diagnostics"][0]
    assert {
        "feature",
        "residual_correlation",
        "absolute_residual_correlation",
        "max_abs_bin_residual_mean",
        "bins",
    } <= set(first_feature)

    candidate_snapshot = Path(result.candidate_snapshot_path).read_text()
    assert "class CandidateModel" in candidate_snapshot

    journal_path = tmp_path / "experiments" / "journal" / "test-run.md"
    assert journal_path.exists()
    journal_text = journal_path.read_text()
    assert "# Experiment Journal: test-run" in journal_text
    assert "Commit:" in journal_text
    candidate_hash_line = f"Candidate SHA256: {run_metadata['candidate_source_sha256']}"
    assert candidate_hash_line in journal_text
    assert "Spec: default" in journal_text
    assert "Primary metric: cv_rmse_mean" in journal_text
    assert "Comparable baseline: test-run" in journal_text
    assert "Interpretability status: pending_agent_judgment" in journal_text
    assert "Interpretability score: NaN" in journal_text
    assert "Next Action" in journal_text

    packet_path = (
        tmp_path / "results" / "runs" / "test-run" / ("interpretability_packet.json")
    )
    assert packet_path.exists()
    packet = json.loads(packet_path.read_text())
    assert packet["run_id"] == "test-run"
    assert packet["project_id"] == "synthetic_regression"
    assert packet["rubric_version"] == RUBRIC_VERSION
    assert packet["model_string"]
    assert packet["scoring_dimensions"] == [
        "prediction",
        "feature_effects",
        "sensitivity",
        "counterfactual",
        "structure",
    ]
    assert not (
        tmp_path / "results" / "runs" / "test-run" / "interpretability_judgment.json"
    ).exists()
    assert not (
        tmp_path
        / "results"
        / "runs"
        / "test-run"
        / "interpretability_judgment_audit.json"
    ).exists()
    assert "calibration_guide" in packet
    assert "high" in packet["calibration_guide"]["prediction"]
    assert "output_schema" in packet


def test_candidate_model_does_not_encode_synthetic_oracle_structure():
    candidate_source = Path(
        "projects/synthetic_regression/experiments/candidate_model.py"
    ).read_text()

    forbidden_fragments = [
        "_target_function",
        "known_structure",
        "hinge_x2_gt_0.25",
        "sin_pi_x5",
        "x3_times_x4",
        "regime_gate",
        "localized_wave",
        "sparse_bump",
        "max(0, x2 - 0.25)",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in candidate_source


def test_synthetic_metadata_does_not_publish_oracle_structure(tmp_path):
    data_dir = _copy_competition_data(tmp_path)
    metadata = json.loads((data_dir / "metadata.json").read_text())

    assert "known_structure" not in metadata
    assert "seed" not in metadata
    assert "n_samples" not in metadata


def test_public_test_data_does_not_include_targets(tmp_path):
    data_dir = _copy_competition_data(tmp_path)
    test = pd.read_csv(data_dir / "test.csv")

    assert "target" not in test.columns
    assert {"id", *FEATURE_COLUMNS} == set(test.columns)


def test_load_evaluation_data_requires_public_files(tmp_path):
    project = synthetic_regression_project(
        data_dir=tmp_path / "missing-data",
        results_dir=tmp_path / "results",
    )

    with pytest.raises(FileNotFoundError, match="Missing competition data files"):
        project.load_data(project.data_dir)


def test_candidate_visible_project_code_does_not_include_generator_oracle():
    datasets_source = Path("projects/synthetic_regression/datasets.py").read_text()

    forbidden_fragments = [
        "_target_function",
        "make_synthetic_frame",
        "generate_synthetic_competition",
        "regime_gate",
        "localized_wave",
        "sparse_bump",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in datasets_source


def test_candidate_model_runtime_does_not_perform_io(monkeypatch):
    evaluation_data = synthetic_regression_project().load_data(DEFAULT_DATA_DIR)
    x = evaluation_data.x_labeled.head(32)
    y = evaluation_data.y_labeled.head(32)

    def forbid_io(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Candidate runtime must not perform filesystem I/O")

    def forbid_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Candidate runtime must not perform network I/O")

    monkeypatch.setattr("builtins.open", forbid_io)
    monkeypatch.setattr(Path, "open", forbid_io)
    monkeypatch.setattr(Path, "read_text", forbid_io)
    monkeypatch.setattr(Path, "read_bytes", forbid_io)
    monkeypatch.setattr(Path, "write_text", forbid_io)
    monkeypatch.setattr(Path, "write_bytes", forbid_io)
    monkeypatch.setattr(Path, "iterdir", forbid_io)
    monkeypatch.setattr(Path, "glob", forbid_io)
    monkeypatch.setattr(Path, "rglob", forbid_io)
    monkeypatch.setattr(pd, "read_csv", forbid_io)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbid_io)

    import socket
    import urllib.request

    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)

    from projects.synthetic_regression.experiments.candidate_model import (
        CandidateModel,
    )

    candidate = CandidateModel()
    candidate.fit(x, y)
    predictions = candidate.predict(x.head(5))
    model_string = str(candidate)

    assert len(predictions) == 5
    assert model_string


def test_run_experiment_cv_metrics_are_deterministic(tmp_path):
    first_project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    first = run_experiment(
        project=first_project,
        run_id="deterministic-one",
    )
    second_project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    second = run_experiment(
        project=second_project,
        run_id="deterministic-two",
    )

    assert second.result_metrics == first.result_metrics


def test_default_evaluation_spec_records_default_cv_metadata():
    spec = DefaultEvaluationSpec()

    assert spec.name == "default"
    assert spec.primary_metric == "cv_rmse_mean"
    assert spec.primary_metric_direction == "minimize"
    assert spec.cv_strategy.name == "kfold_5_shuffle_seed42"
    assert spec.cv_strategy.n_splits == 5
    assert spec.cv_strategy.random_state == 42


def test_synthetic_regression_project_uses_project_scoped_defaults():
    project = synthetic_regression_project()

    assert project.project_id == "synthetic_regression"
    assert project.package_name == "projects.synthetic_regression"
    assert project.default_candidate_module == DEFAULT_CANDIDATE_MODULE
    assert project.data_dir == DEFAULT_DATA_DIR
    assert project.results_dir == DEFAULT_RESULTS_DIR


@pytest.mark.parametrize(
    "project_id",
    ["", "synthetic-regression", "SyntheticRegression", "synthetic regression", "../x"],
)
def test_project_rejects_invalid_project_id(project_id):
    with pytest.raises(ValueError, match="project_id"):
        Project(
            project_id=project_id,
            package_name=f"projects.{project_id}",
            spec=DefaultEvaluationSpec(),
            data_dir=Path("data"),
            results_dir=Path("results"),
            load_data=lambda path: pytest.fail("load_data should not be called"),
            protected_paths=(),
            forbidden_candidate_source_fragments=(),
            write_submission=_unused_submission_writer,
        )


def test_project_package_name_must_end_with_project_id():
    with pytest.raises(ValueError, match="package_name"):
        Project(
            project_id="synthetic_regression",
            package_name="projects.other",
            spec=DefaultEvaluationSpec(),
            data_dir=Path("data"),
            results_dir=Path("results"),
            load_data=lambda path: pytest.fail("load_data should not be called"),
            protected_paths=(),
            forbidden_candidate_source_fragments=(),
            write_submission=_unused_submission_writer,
        )


def test_evaluation_spec_rejects_unknown_primary_metric_direction():
    with pytest.raises(ValueError, match="primary_metric_direction"):
        EvaluationSpec(
            name="bad-direction",
            primary_metric="custom_score",
            primary_metric_direction="lower",  # type: ignore[arg-type]
            cv_strategy=CVStrategy(
                name="kfold_2_shuffle_seed7",
                n_splits=2,
                random_state=7,
                description="2-fold shuffled KFold for validation testing.",
                splitter=KFold(n_splits=2, shuffle=True, random_state=7),
            ),
        )


class CountingEvaluationSpec(DefaultEvaluationSpec):
    score_calls: int
    aggregate_calls: int

    def __init__(self):
        super().__init__(
            name="counting",
            primary_metric="custom_score",
            cv_strategy=CVStrategy(
                name="kfold_3_shuffle_seed11",
                n_splits=3,
                random_state=11,
                description="3-fold shuffled KFold for custom spec testing.",
                splitter=KFold(n_splits=3, shuffle=True, random_state=11),
            ),
        )
        object.__setattr__(self, "score_calls", 0)
        object.__setattr__(self, "aggregate_calls", 0)

    def score_predictions(self, y_true, y_pred):
        object.__setattr__(self, "score_calls", self.score_calls + 1)
        return super().score_predictions(y_true, y_pred)

    def aggregate_fold_scores(self, fold_scores):
        object.__setattr__(self, "aggregate_calls", self.aggregate_calls + 1)
        metrics = super().aggregate_fold_scores(fold_scores)
        metrics["custom_score"] = metrics["cv_rmse_mean"] + metrics["cv_rmse_std"]
        return metrics

    def report_metric_lines(self, metrics):
        return [
            f"- custom_score (primary): {metrics['custom_score']:.6f}",
            f"- cv_rmse_mean: {metrics['cv_rmse_mean']:.6f}",
        ]


def test_run_experiment_uses_custom_project_evaluation_spec(tmp_path):
    evaluation_spec = CountingEvaluationSpec()

    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
        spec=evaluation_spec,
    )
    result = run_experiment(
        project=project,
        run_id="custom-spec",
    )

    assert result.cv_strategy_name == "kfold_3_shuffle_seed11"
    assert result.cv_n_splits == 3
    assert result.cv_random_state == 11
    assert result.spec_name == "counting"
    assert result.primary_metric == "custom_score"
    assert result.primary_metric_direction == "minimize"
    assert "custom_score" in result.metrics
    assert evaluation_spec.score_calls == 3
    assert evaluation_spec.aggregate_calls == 1

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.loc[0, "spec_name"] == "counting"
    assert leaderboard.loc[0, "primary_metric"] == "custom_score"
    assert leaderboard.loc[0, "primary_metric_direction"] == "minimize"
    assert leaderboard.loc[0, "cv_strategy_name"] == "kfold_3_shuffle_seed11"
    assert leaderboard.loc[0, "cv_n_splits"] == 3
    assert leaderboard.loc[0, "cv_random_state"] == 11
    assert leaderboard.loc[0, "custom_score"] > 0

    report_text = Path(result.report_path).read_text()
    assert "Evaluation spec: counting" in report_text
    assert "Primary metric direction: minimize" in report_text
    assert "CV strategy: kfold_3_shuffle_seed11" in report_text
    assert "CV splits: 3" in report_text
    assert "custom_score (primary)" in report_text

    run_metadata = json.loads(Path(result.run_metadata_path).read_text())
    assert run_metadata["spec"]["primary_metric_direction"] == "minimize"


def test_run_experiment_supports_non_regression_project_policy(tmp_path):
    package_dir = tmp_path / "fake_project" / "experiments"
    package_dir.mkdir(parents=True)
    (tmp_path / "fake_project" / "__init__.py").write_text("")
    (package_dir / "__init__.py").write_text("")
    (package_dir / "candidate_model.py").write_text(
        textwrap.dedent(
            """
            from toy_imodels.core.candidate import BaseCandidateModel


            class CandidateModel(BaseCandidateModel):
                model_name = "threshold_classifier"
                notes = "Predicts class labels from x0 threshold."

                def fit(self, X, y):
                    return self

                def predict(self, X):
                    return (X["x0"] > 0.5).astype(int)

                def __str__(self):
                    return "if x0 > 0.5 then class 1 else class 0"
            """
        )
    )
    data = EvaluationData(
        x_labeled=pd.DataFrame({"x0": [0.0, 0.2, 0.7, 0.9, 0.1, 0.8]}),
        y_labeled=pd.Series([0, 0, 1, 1, 0, 1]),
        x_test=pd.DataFrame({"x0": [0.3, 0.6]}),
        feature_columns=["x0"],
    )
    project = Project(
        project_id="fake_project",
        package_name="fake_project",
        spec=AccuracyEvaluationSpec(),
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        load_data=lambda _path: data,
        protected_paths=("fake_project/spec.py",),
        forbidden_candidate_source_fragments=("NEVER_USE_THIS_FRAGMENT",),
        write_submission=_write_fake_submission,
        write_diagnostics=None,
    )

    result = run_experiment(
        project=project,
        run_id="accuracy-run",
        import_path=[tmp_path],
    )

    assert result.status == "success"
    assert result.primary_metric == "accuracy"
    assert result.primary_metric_direction == "maximize"
    assert result.metrics == {"accuracy": 1.0}
    assert result.result_metrics == {"accuracy": 1.0}
    assert Path(result.artifact_paths["submission_path"]).read_text() == "0,1\n"
    assert "diagnostics_path" not in result.artifact_paths

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.loc[0, "accuracy"] == 1.0
    assert "cv_rmse_mean" not in leaderboard.columns
    assert "diagnostics_path" not in leaderboard.columns

    report_text = Path(result.report_path).read_text()
    assert "accuracy (primary): 1.000000" in report_text
    assert "Diagnostics:" not in report_text


def test_interpretability_score_requires_agent_judgment(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(
        project=project,
        run_id="fixed-interp",
    )

    report_text = Path(result.report_path).read_text()
    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert math.isnan(result.interpretability_score)
    assert pd.isna(leaderboard.loc[0, "interpretability_score"])
    assert "Interpretability status: pending_agent_judgment" in report_text
    assert "Static interpretability score" not in report_text
    assert not hasattr(EvaluationSpec, "score_model_string")


def test_apply_interpretability_judgment_updates_leaderboard_and_report(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(
        project=project,
        run_id="judged-run",
    )
    judgment_path = _write_valid_interpretability_judgment(tmp_path, "judged-run")

    normalized_path = apply_interpretability_judgment(
        results_dir=tmp_path / "results",
        run_id="judged-run",
        judgment_path=judgment_path,
    )

    assert normalized_path == (
        tmp_path / "results" / "runs" / "judged-run" / "interpretability_judgment.json"
    )
    normalized = json.loads(normalized_path.read_text())
    assert normalized["interpretability_score"] == 0.7
    audit_path = (
        tmp_path
        / "results"
        / "runs"
        / "judged-run"
        / "interpretability_judgment_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    assert audit["audit_status"] in {"pass", "review"}
    assert audit["judged_score"] == 0.7
    assert "static_fallback_score" not in audit

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.shape[0] == 1
    assert leaderboard.loc[0, "run_id"] == "judged-run"
    assert leaderboard.loc[0, "interpretability_score"] == 0.7
    assert leaderboard.loc[0, "interpretability_rubric_version"] == RUBRIC_VERSION
    assert leaderboard.loc[0, "interpretability_judgment_path"] == str(normalized_path)
    assert leaderboard.loc[0, "interpretability_audit_status"] in {"pass", "review"}
    assert leaderboard.loc[0, "interpretability_audit_path"] == str(audit_path)

    report_text = Path(result.report_path).read_text()
    assert "Agent-judged interpretability score: 0.7000" in report_text
    assert "Interpretability judgment" in report_text
    assert "Audit artifact:" in report_text
    assert "Audit status:" in report_text
    assert "prediction: 1.0000" in report_text

    journal_text = (tmp_path / "experiments" / "journal" / "judged-run.md").read_text()
    assert "Interpretability status: agent_judged" in journal_text
    assert "Interpretability score: 0.7000" in journal_text
    assert "Applied Interpretability Judgment" in journal_text
    assert f"Judgment artifact: {normalized_path}" in journal_text
    assert f"Audit artifact: {audit_path}" in journal_text


def test_update_result_can_target_project_scoped_duplicate_run_ids(tmp_path):
    results_dir = tmp_path / "results"
    append_result(
        results_dir,
        {
            "run_id": "shared-run",
            "project_id": "first_project",
            "status": "success",
            "interpretability_score": 0.1,
        },
    )
    append_result(
        results_dir,
        {
            "run_id": "shared-run",
            "project_id": "second_project",
            "status": "success",
            "interpretability_score": 0.2,
        },
    )

    with pytest.raises(ValueError, match="Expected exactly one leaderboard row"):
        update_result(
            results_dir,
            run_id="shared-run",
            updates={"interpretability_score": 0.3},
        )

    update_result(
        results_dir,
        run_id="shared-run",
        project_id="second_project",
        updates={"interpretability_score": 0.9},
    )

    leaderboard = pd.read_csv(results_dir / "leaderboard.csv")
    first_score = leaderboard.loc[
        leaderboard["project_id"] == "first_project", "interpretability_score"
    ].iloc[0]
    second_score = leaderboard.loc[
        leaderboard["project_id"] == "second_project", "interpretability_score"
    ].iloc[0]
    assert first_score == 0.1
    assert second_score == 0.9


def test_verify_experiment_run_fails_before_agent_judgment(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    run_experiment(project=project, run_id="audit-pending-judgment")

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-pending-judgment",
        project=project,
    )

    judgment = next(
        finding for finding in findings if finding.check == "interpretability_judgment"
    )
    assert not judgment.ok
    assert "pending agent judgment" in judgment.detail


def test_verify_experiment_run_passes_for_agent_judged_run(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    run_experiment(project=project, run_id="audit-pass")
    apply_interpretability_judgment(
        results_dir=tmp_path / "results",
        run_id="audit-pass",
        judgment_path=_write_valid_interpretability_judgment(tmp_path, "audit-pass"),
    )

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-pass",
        project=project,
    )

    assert all(finding.ok for finding in findings)
    assert {finding.check for finding in findings} == {
        "metadata_present",
        "snapshot_hash",
        "active_candidate_hash",
        "active_spec",
        "protected_paths",
        "comparable_baseline",
        "journal",
        "interpretability_judgment",
    }


def test_verify_experiment_run_fails_when_snapshot_hash_differs(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(project=project, run_id="audit-bad-snapshot")
    Path(result.candidate_snapshot_path).write_text("# tampered\n")

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-bad-snapshot",
        project=project,
    )

    snapshot = next(finding for finding in findings if finding.check == "snapshot_hash")
    assert not snapshot.ok
    assert "does not match metadata" in snapshot.detail


def test_verify_experiment_run_fails_when_active_candidate_hash_differs(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(project=project, run_id="audit-bad-candidate")
    metadata_path = Path(result.run_metadata_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["candidate_source_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-bad-candidate",
        project=project,
    )

    active = next(
        finding for finding in findings if finding.check == "active_candidate_hash"
    )
    assert not active.ok
    assert "does not match run" in active.detail


def test_verify_experiment_run_fails_on_protected_path_drift(tmp_path, monkeypatch):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    result = run_experiment(project=project, run_id="audit-protected-drift")
    metadata_path = Path(result.run_metadata_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["git_dirty"] = False
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        "toy_imodels.audit.protected_paths_changed_since_commit",
        lambda git_commit, *, paths, root=".": list(paths),
    )

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-protected-drift",
        project=project,
    )

    protected = next(
        finding for finding in findings if finding.check == "protected_paths"
    )
    assert not protected.ok
    assert "toy_imodels/" in protected.detail


def test_verify_experiment_run_fails_without_comparable_baseline(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    run_experiment(project=project, run_id="audit-missing-baseline")
    leaderboard_path = tmp_path / "results" / "leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard.loc[0, "spec_name"] = "other-spec"
    leaderboard.to_csv(leaderboard_path, index=False)

    findings = verify_experiment_run(
        results_dir=tmp_path / "results",
        run_id="audit-missing-baseline",
        project=project,
    )

    baseline = next(
        finding for finding in findings if finding.check == "comparable_baseline"
    )
    assert not baseline.ok
    assert "no successful comparable baseline" in baseline.detail


def test_append_result_requires_project_id(tmp_path):
    with pytest.raises(ValueError, match="non-empty project_id"):
        append_result(
            tmp_path / "results",
            {"run_id": "missing-project", "status": "success"},
        )


def test_interpretability_judgment_audit_flags_scores_without_evidence_terms():
    packet = build_interpretability_packet(
        run_id="weak-run",
        model_string="Opaque predictor.",
    )
    judgment = validate_interpretability_judgment(
        {
            "run_id": "weak-run",
            "rubric_version": RUBRIC_VERSION,
            "dimension_scores": {
                "prediction": {"score": 1.0, "rationale": "Claims it predicts."},
                "feature_effects": {"score": 1.0, "rationale": "Claims features."},
                "sensitivity": {"score": 1.0, "rationale": "Claims changes."},
                "counterfactual": {"score": 1.0, "rationale": "Claims changes."},
                "structure": {"score": 1.0, "rationale": "Claims structure."},
            },
            "interpretability_score": 1.0,
        },
        expected_run_id="weak-run",
    )

    audit = audit_interpretability_judgment(packet=packet, judgment=judgment)

    assert audit["audit_status"] == "review"
    assert audit["judged_score"] == 1.0
    assert "static_fallback_score" not in audit
    assert any(
        "without obvious model-string evidence" in warning
        for warning in audit["warnings"]
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"run_id": "other-run"}, "run_id does not match"),
        ({"rubric_version": "wrong-version"}, "rubric_version is not supported"),
        (
            {
                "dimension_scores": {
                    "prediction": {"score": 2.0, "rationale": "Too high."},
                    "feature_effects": {"score": 0.8, "rationale": "ok"},
                    "sensitivity": {"score": 0.6, "rationale": "ok"},
                    "counterfactual": {"score": 0.4, "rationale": "ok"},
                    "structure": {"score": 0.7, "rationale": "ok"},
                }
            },
            "must be between 0 and 1",
        ),
        ({"interpretability_score": 0.9}, "must equal the mean"),
    ],
)
def test_apply_interpretability_judgment_rejects_invalid_artifacts(
    tmp_path, override, message
):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    run_experiment(
        project=project,
        run_id="invalid-judgment-run",
    )
    judgment = {
        "run_id": "invalid-judgment-run",
        "rubric_version": RUBRIC_VERSION,
        "dimension_scores": {
            "prediction": {"score": 1.0, "rationale": "ok"},
            "feature_effects": {"score": 0.8, "rationale": "ok"},
            "sensitivity": {"score": 0.6, "rationale": "ok"},
            "counterfactual": {"score": 0.4, "rationale": "ok"},
            "structure": {"score": 0.7, "rationale": "ok"},
        },
        "interpretability_score": 0.7,
    }
    judgment.update(override)
    judgment_path = tmp_path / "invalid_judgment.json"
    judgment_path.write_text(json.dumps(judgment) + "\n")

    with pytest.raises(ValueError, match=message):
        apply_interpretability_judgment(
            results_dir=tmp_path / "results",
            run_id="invalid-judgment-run",
            judgment_path=judgment_path,
        )


def test_project_data_dataclass_validates_feature_columns(tmp_path):
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    evaluation_data = project.load_data(project.data_dir)
    with pytest.raises(ValueError, match="missing feature columns"):
        evaluation_data.__class__(
            x_labeled=evaluation_data.x_labeled.drop(columns=["x0"]),
            y_labeled=evaluation_data.y_labeled,
            x_test=evaluation_data.x_test,
            feature_columns=evaluation_data.feature_columns,
        )


def test_run_experiment_records_runtime_failure_traceback(tmp_path):
    module_dir = tmp_path / "runtime_candidate_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    (module_dir / "broken_runtime.py").write_text(
        textwrap.dedent(
            """
            from toy_imodels.core.candidate import BaseCandidateModel


            class CandidateModel(BaseCandidateModel):
                def fit(self, X, y):
                    raise RuntimeError("planned fit failure")

                def predict(self, X):
                    return []

                def __str__(self):
                    return "broken"
            """
        )
    )
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )

    with pytest.raises(RuntimeError, match="planned fit failure"):
        run_experiment(
            project=project,
            candidate_module="runtime_candidate_pkg.broken_runtime",
            run_id="runtime-failure",
            import_path=[tmp_path],
        )

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.loc[0, "run_id"] == "runtime-failure"
    assert leaderboard.loc[0, "status"] == "error"
    assert "planned fit failure" in leaderboard.loc[0, "error"]
    traceback_path = Path(leaderboard.loc[0, "error_traceback_path"])
    assert traceback_path.exists()
    assert "planned fit failure" in traceback_path.read_text()


def test_run_experiment_records_contract_failure_traceback(tmp_path):
    module_dir = tmp_path / "contract_candidate_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    (module_dir / "missing_contract.py").write_text("BROKEN = True\n")
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )

    with pytest.raises(RuntimeError, match="must define CandidateModel"):
        run_experiment(
            project=project,
            candidate_module="contract_candidate_pkg.missing_contract",
            run_id="contract-failure",
            import_path=[tmp_path],
        )

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.loc[0, "run_id"] == "contract-failure"
    assert leaderboard.loc[0, "status"] == "contract_error"
    assert "must define CandidateModel" in leaderboard.loc[0, "error"]
    traceback_path = Path(leaderboard.loc[0, "error_traceback_path"])
    assert traceback_path.exists()
    assert "must define CandidateModel" in traceback_path.read_text()


def test_run_experiment_rejects_candidate_dataset_loader_references(tmp_path):
    module_dir = tmp_path / "leaky_candidate_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    (module_dir / "leaky.py").write_text(
        textwrap.dedent(
            """
            from projects.synthetic_regression.datasets import DEFAULT_DATA_DIR
            from toy_imodels.core.candidate import BaseCandidateModel


            class CandidateModel(BaseCandidateModel):
                def fit(self, X, y):
                    self.path = DEFAULT_DATA_DIR
                    return self

                def predict(self, X):
                    return [0.0] * len(X)

                def __str__(self):
                    return "leaky candidate"
            """
        )
    )
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )

    with pytest.raises(RuntimeError, match="must not reference competition data"):
        run_experiment(
            project=project,
            candidate_module="leaky_candidate_pkg.leaky",
            run_id="leaky-candidate",
            import_path=[tmp_path],
        )

    leaderboard = pd.read_csv(tmp_path / "results" / "leaderboard.csv")
    assert leaderboard.loc[0, "run_id"] == "leaky-candidate"
    assert leaderboard.loc[0, "status"] == "contract_error"
    assert "DEFAULT_DATA_DIR" in leaderboard.loc[0, "error"]


def test_run_experiment_uses_project_forbidden_source_fragments(tmp_path):
    module_dir = tmp_path / "project_policy_candidate_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("")
    (module_dir / "leaky.py").write_text(
        textwrap.dedent(
            """
            from toy_imodels.core.candidate import BaseCandidateModel

            PROJECT_SECRET_TOKEN = "do not use"


            class CandidateModel(BaseCandidateModel):
                def fit(self, X, y):
                    return self

                def predict(self, X):
                    return [0.0] * len(X)

                def __str__(self):
                    return PROJECT_SECRET_TOKEN
            """
        )
    )
    project = synthetic_regression_project(
        data_dir=_copy_competition_data(tmp_path),
        results_dir=tmp_path / "results",
    )
    project = Project(
        project_id=project.project_id,
        package_name=project.package_name,
        spec=project.spec,
        data_dir=project.data_dir,
        results_dir=project.results_dir,
        load_data=project.load_data,
        protected_paths=project.protected_paths,
        forbidden_candidate_source_fragments=("PROJECT_SECRET_TOKEN",),
        write_submission=project.write_submission,
        write_diagnostics=project.write_diagnostics,
    )

    with pytest.raises(RuntimeError, match="PROJECT_SECRET_TOKEN"):
        run_experiment(
            project=project,
            candidate_module="project_policy_candidate_pkg.leaky",
            run_id="project-policy-leak",
            import_path=[tmp_path],
        )


def test_fixed_harness_source_has_no_synthetic_project_policy():
    source_text = "\n".join(
        path.read_text() for path in Path("toy_imodels").rglob("*.py")
    )

    assert "synthetic_regression" not in source_text
    assert "projects/synthetic_regression" not in source_text
    assert "projects.synthetic_regression" not in source_text
    assert "cv_rmse" not in source_text


def test_toy_experiment_skill_mentions_pdca_run_artifacts():
    skill_text = Path(
        ".codex/skills/agentic-imodels-toy-experiment/SKILL.md"
    ).read_text()

    assert "PDCA Run Artifact Usage" in skill_text
    assert "Plan:" in skill_text
    assert "Do:" in skill_text
    assert "Check:" in skill_text
    assert "Act:" in skill_text
    assert "fold_metrics.json" in skill_text
    assert "run_metadata.json" in skill_text
    assert "candidate_snapshot.py" in skill_text
