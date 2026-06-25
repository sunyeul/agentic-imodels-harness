from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "poly2_ridgecv"
    notes = (
        "RidgeCV over standardized linear, squared, and pairwise interaction "
        "features to test whether smooth second-order structure lowers RMSE."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.alphas = np.logspace(-3, 3, 13)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("ridge", RidgeCV(alphas=self.alphas, cv=5)),
            ]
        )

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def __str__(self) -> str:
        ridge = self.pipeline.named_steps["ridge"]
        poly = self.pipeline.named_steps["poly"]
        basis_names = poly.get_feature_names_out(self.feature_names)
        coefs = ridge.coef_
        top = sorted(
            zip(basis_names, coefs, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]
        top_terms = ", ".join(f"{name} ({coef:.3f})" for name, coef in top)
        interaction_terms = [name for name in basis_names if " " in name]
        squared_terms = [name for name in basis_names if "^2" in name]
        return (
            "Polynomial RidgeCV regression. "
            "Inputs are z-scored, expanded to degree-2 polynomial features, "
            "then fit with ridge regularization selected by inner 5-fold CV. "
            f"Selected alpha: {ridge.alpha_:.4g}. "
            f"Intercept on the standardized polynomial basis: {ridge.intercept_:.3f}. "
            f"Top learned basis terms by absolute coefficient: {top_terms}. "
            f"Basis includes {len(self.feature_names)} linear terms, "
            f"{len(squared_terms)} squared terms, and "
            f"{len(interaction_terms)} pairwise interaction terms. "
            "Positive coefficients increase predictions as their standardized "
            "basis term rises; negative coefficients decrease predictions. "
            "Counterfactual sensitivity can be read from the dominant linear, "
            "squared, and interaction terms above, while smaller omitted terms "
            "remain ridge-shrunk rather than hard-pruned."
        )
