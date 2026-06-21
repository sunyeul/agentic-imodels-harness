from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "poly2_regularized_ridgecv"
    notes = (
        "RidgeCV over standardized degree-2 polynomial features with a denser "
        "regularization grid to reduce cubic-basis variance and lower RMSE."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.alphas = np.logspace(-1, 5, 25)
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

    def _basis_degree(self, name: str) -> int:
        degree = 0
        for part in name.split(" "):
            if "^" in part:
                degree += int(part.rsplit("^", maxsplit=1)[1])
            else:
                degree += 1
        return degree

    def __str__(self) -> str:
        ridge = self.pipeline.named_steps["ridge"]
        poly = self.pipeline.named_steps["poly"]
        basis_names = poly.get_feature_names_out(self.feature_names)
        coefs = ridge.coef_
        top = sorted(
            zip(basis_names, coefs, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:12]
        top_terms = ", ".join(f"{name} ({coef:.3f})" for name, coef in top)
        degree_counts = {
            degree: sum(1 for name in basis_names if self._basis_degree(name) == degree)
            for degree in (1, 2)
        }
        interaction_terms = [name for name in basis_names if " " in name]
        return (
            "Polynomial RidgeCV regression. "
            "Inputs are z-scored, expanded to degree-2 polynomial features "
            "(linear, squared, and pairwise interaction terms), then fit with "
            "ridge regularization selected by inner 5-fold CV over a denser "
            "alpha grid from 0.1 to 100000. "
            f"Selected alpha: {ridge.alpha_:.4g}. "
            f"Intercept on the standardized polynomial basis: {ridge.intercept_:.3f}. "
            f"Top learned basis terms by absolute coefficient: {top_terms}. "
            f"Basis includes {degree_counts[1]} linear terms, "
            f"{degree_counts[2]} quadratic terms, with "
            f"{len(interaction_terms)} cross-feature interaction terms. "
            "Positive coefficients increase predictions as their standardized "
            "basis term rises; negative coefficients decrease predictions. "
            "Counterfactual sensitivity can be read from the dominant linear, "
            "squared, and interaction terms above, while smaller omitted terms "
            "remain ridge-shrunk rather than hard-pruned. Cubic terms are "
            "excluded to control variance after the prior degree-3 expansion "
            "showed worse cross-validated RMSE."
        )
