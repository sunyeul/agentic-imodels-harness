from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold

from toy_imodels.metrics import mae, r2, rmse
from toy_imodels.spec import CVStrategy, EvaluationSpec


def default_cv_strategy() -> CVStrategy:
    return CVStrategy(
        name="kfold_5_shuffle_seed42",
        n_splits=5,
        random_state=42,
        description="5-fold shuffled KFold over all labeled rows.",
        splitter=KFold(n_splits=5, shuffle=True, random_state=42),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DefaultEvaluationSpec(EvaluationSpec):
    """Default synthetic regression project evaluation policy."""

    name: str = "default"
    primary_metric: str = "cv_rmse_mean"
    cv_strategy: CVStrategy = field(default_factory=default_cv_strategy)

    def score_predictions(self, y_true, y_pred):
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r2(y_true, y_pred),
        }

    def aggregate_fold_scores(self, fold_scores):
        return {
            "cv_rmse_mean": self._mean(fold_scores, "rmse"),
            "cv_rmse_std": self._std(fold_scores, "rmse"),
            "cv_mae_mean": self._mean(fold_scores, "mae"),
            "cv_mae_std": self._std(fold_scores, "mae"),
            "cv_r2_mean": self._mean(fold_scores, "r2"),
            "cv_r2_std": self._std(fold_scores, "r2"),
        }

    def result_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            "cv_rmse_mean": metrics.get("cv_rmse_mean", float("nan")),
            "cv_rmse_std": metrics.get("cv_rmse_std", float("nan")),
            "cv_mae_mean": metrics.get("cv_mae_mean", float("nan")),
            "cv_mae_std": metrics.get("cv_mae_std", float("nan")),
            "cv_r2_mean": metrics.get("cv_r2_mean", float("nan")),
            "cv_r2_std": metrics.get("cv_r2_std", float("nan")),
        }

    @staticmethod
    def _mean(fold_scores: list[dict[str, float]], metric_name: str) -> float:
        return float(np.mean([scores[metric_name] for scores in fold_scores]))

    @staticmethod
    def _std(fold_scores: list[dict[str, float]], metric_name: str) -> float:
        return float(np.std([scores[metric_name] for scores in fold_scores]))
