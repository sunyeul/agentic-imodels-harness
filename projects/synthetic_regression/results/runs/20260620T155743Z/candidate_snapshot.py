from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "sparse_quadratic_elasticnet_cv"
    notes = (
        "ElasticNetCV over standardized first- and second-order polynomial "
        "features to retain strong curvature/interactions while pruning weak "
        "expanded terms."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.term_names: list[str] = []
        self.pipeline = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                alphas=np.logspace(-4, 1, 24),
                cv=5,
                max_iter=20000,
            ),
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
        enet = self.pipeline.named_steps["elasticnetcv"]
        scaler = self.pipeline.named_steps["standardscaler"]
        coefs = enet.coef_ / scaler.scale_
        intercept = enet.intercept_ - float(np.sum(coefs * scaler.mean_))
        term_names = self.term_names or self.feature_names
        active_terms = [
            f"{coef:.3f}*{name}"
            for name, coef in zip(term_names, coefs, strict=True)
            if abs(coef) > 1e-8
        ]
        top = sorted(
            [
                (name, coef)
                for name, coef in zip(term_names, coefs, strict=True)
                if abs(coef) > 1e-8
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]
        top_features = ", ".join(f"{name} ({coef:.3f})" for name, coef in top)
        equation = " + ".join(active_terms) if active_terms else "0.000"
        active_count = len(active_terms)
        zero_count = max(len(term_names) - active_count, 0)
        return (
            "Sparse quadratic ElasticNetCV regression over original features, "
            "squared terms, and pairwise interactions. "
            f"Prediction equation: y = {intercept:.3f} + {equation}. "
            f"Selected ElasticNet alpha: {enet.alpha_:.4g}; selected l1_ratio: "
            f"{enet.l1_ratio_:.2f}. Active expanded terms: {active_count}; "
            f"zeroed weak expanded terms: {zero_count}. Top absolute active "
            f"coefficient terms: {top_features}. Coefficients are shown after "
            "converting standardized polynomial terms back to their "
            "expanded-feature units; terms not shown have exact zero "
            "coefficients from the sparse penalty. Positive active terms raise "
            "the prediction as their expanded value increases; negative active "
            "terms lower it. Counterfactual sensitivity is local and additive "
            "in this retained expanded basis, so changing an original feature "
            "can affect its linear term, squared term, and retained pairwise "
            "interaction terms that include it."
        )
