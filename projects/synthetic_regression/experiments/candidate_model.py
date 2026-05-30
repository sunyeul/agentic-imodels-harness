from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "baseline_ridge"
    notes = "Ridge regression baseline with standardized features."

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.pipeline = make_pipeline(
            StandardScaler(), Ridge(alpha=1.0, random_state=42)
        )

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def __str__(self) -> str:
        ridge = self.pipeline.named_steps["ridge"]
        scaler = self.pipeline.named_steps["standardscaler"]
        coefs = ridge.coef_ / scaler.scale_
        intercept = ridge.intercept_ - float(np.sum(coefs * scaler.mean_))
        terms = [
            f"{coef:.3f}*{name}"
            for name, coef in zip(self.feature_names, coefs, strict=True)
        ]
        top = sorted(
            zip(self.feature_names, coefs, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]
        top_features = ", ".join(f"{name} ({coef:.3f})" for name, coef in top)
        equation = " + ".join(terms)
        return (
            "Ridge linear regression. "
            f"Prediction equation: y = {intercept:.3f} + {equation}. "
            f"Top coefficient features: {top_features}. "
            "Coefficients are shown in original feature units."
        )
