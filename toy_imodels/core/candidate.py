from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin


class BaseCandidateModel(BaseEstimator, RegressorMixin, ABC):
    """Base contract for sklearn-compatible editable candidate regressors."""

    model_name = "candidate_model"
    notes = ""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> BaseCandidateModel:
        """Fit the candidate regressor and return self."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return predictions for X."""

    @abstractmethod
    def __str__(self) -> str:
        """Return an agent-readable textual representation of the model."""
