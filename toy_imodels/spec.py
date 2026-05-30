from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
        raise NotImplementedError

    def aggregate_fold_scores(
        self, fold_scores: list[dict[str, float]]
    ) -> dict[str, float]:
        raise NotImplementedError

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

    def _format_metric_line(self, metric_name: str, metric_value: float) -> str:
        primary_marker = " (primary)" if metric_name == self.primary_metric else ""
        return f"- {metric_name}{primary_marker}: {metric_value:.6f}"

    def result_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return metrics
