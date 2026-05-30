from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from toy_imodels.metrics import mae, r2, rmse

MetricDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CVStrategy:
    """Harness-owned cross-validation strategy metadata and splitter."""

    name: str
    n_splits: int
    random_state: int | None
    description: str
    splitter: Any

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CVStrategy.name must not be empty")
        if self.n_splits < 2:
            raise ValueError("CVStrategy.n_splits must be at least 2")
        if not hasattr(self.splitter, "split"):
            raise TypeError("CVStrategy.splitter must provide split(...)")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationSpec:
    """Project-owned contract for predictive evaluation policy."""

    name: str
    primary_metric: str
    cv_strategy: CVStrategy
    primary_metric_direction: MetricDirection = "minimize"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("EvaluationSpec.name must not be empty")
        if not self.primary_metric.strip():
            raise ValueError("EvaluationSpec.primary_metric must not be empty")
        if self.primary_metric_direction not in ("minimize", "maximize"):
            raise ValueError(
                "EvaluationSpec.primary_metric_direction must be "
                "'minimize' or 'maximize'"
            )

    def score_predictions(self, y_true: Any, y_pred: Any) -> dict[str, float]:
        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r2(y_true, y_pred),
        }

    def aggregate_fold_scores(
        self, fold_scores: list[dict[str, float]]
    ) -> dict[str, float]:
        return {
            "cv_rmse_mean": self._mean(fold_scores, "rmse"),
            "cv_rmse_std": self._std(fold_scores, "rmse"),
            "cv_mae_mean": self._mean(fold_scores, "mae"),
            "cv_mae_std": self._std(fold_scores, "mae"),
            "cv_r2_mean": self._mean(fold_scores, "r2"),
            "cv_r2_std": self._std(fold_scores, "r2"),
        }

    def report_metadata(self) -> dict[str, object]:
        return {
            "spec_name": self.name,
            "primary_metric": self.primary_metric,
            "primary_metric_direction": self.primary_metric_direction,
            "cv_strategy_name": self.cv_strategy.name,
            "cv_n_splits": self.cv_strategy.n_splits,
            "cv_random_state": self.cv_strategy.random_state,
            "cv_description": self.cv_strategy.description,
        }

    def leaderboard_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return metrics

    def report_metric_lines(self, metrics: dict[str, float]) -> list[str]:
        return [
            self._format_metric_line(metric_name, metric_value)
            for metric_name, metric_value in metrics.items()
        ]

    @staticmethod
    def _mean(fold_scores: list[dict[str, float]], metric_name: str) -> float:
        return float(np.mean([scores[metric_name] for scores in fold_scores]))

    @staticmethod
    def _std(fold_scores: list[dict[str, float]], metric_name: str) -> float:
        return float(np.std([scores[metric_name] for scores in fold_scores]))

    def _format_metric_line(self, metric_name: str, metric_value: float) -> str:
        primary_marker = " (primary)" if metric_name == self.primary_metric else ""
        return f"- {metric_name}{primary_marker}: {metric_value:.6f}"

    def result_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return {
            "cv_rmse_mean": metrics.get("cv_rmse_mean", float("nan")),
            "cv_rmse_std": metrics.get("cv_rmse_std", float("nan")),
            "cv_mae_mean": metrics.get("cv_mae_mean", float("nan")),
            "cv_mae_std": metrics.get("cv_mae_std", float("nan")),
            "cv_r2_mean": metrics.get("cv_r2_mean", float("nan")),
            "cv_r2_std": metrics.get("cv_r2_std", float("nan")),
        }
