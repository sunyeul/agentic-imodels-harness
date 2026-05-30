from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.model_selection import KFold

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
