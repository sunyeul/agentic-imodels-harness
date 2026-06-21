from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "quadratic_ridge_cv"
    notes = (
        "RidgeCV over standardized first- and second-order polynomial features "
        "to capture smooth curvature and pairwise interactions."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.term_names: list[str] = []
        self.pipeline = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        poly = self.pipeline.named_steps["polynomialfeatures"]
        self.pipeline.fit(X, y)
        self.term_names = list(poly.get_feature_names_out(self.feature_names))
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def __str__(self) -> str:
        ridge = self.pipeline.named_steps["ridgecv"]
        scaler = self.pipeline.named_steps["standardscaler"]
        coefs = ridge.coef_ / scaler.scale_
        intercept = ridge.intercept_ - float(np.sum(coefs * scaler.mean_))
        term_names = self.term_names or self.feature_names
        terms = [
            f"{coef:.3f}*{name}" for name, coef in zip(term_names, coefs, strict=True)
        ]
        top = sorted(
            zip(term_names, coefs, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:8]
        top_features = ", ".join(f"{name} ({coef:.3f})" for name, coef in top)
        equation = " + ".join(terms)
        return (
            "Quadratic RidgeCV regression over original features, squared "
            "terms, and pairwise interactions. "
            f"Prediction equation: y = {intercept:.3f} + {equation}. "
            f"Selected ridge alpha: {ridge.alpha_:.4g}. "
            f"Top absolute coefficient terms: {top_features}. "
            "Coefficients are shown after converting standardized polynomial "
            "terms back to their expanded-feature units. Positive terms raise "
            "the prediction as their expanded value increases; negative terms "
            "lower it. Counterfactual sensitivity is local and additive in this "
            "expanded basis, so changing an original feature can affect its "
            "linear term, squared term, and all pairwise interaction terms that "
            "include it."
        )
