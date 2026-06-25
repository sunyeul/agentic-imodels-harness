from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "sparse_cubic_elasticnet_cv"
    notes = (
        "ElasticNetCV over standardized polynomial features up to degree 3 "
        "to test whether sparse higher-order curvature/interactions reduce "
        "RMSE beyond the retained quadratic structure."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.term_names: list[str] = []
        self.pipeline = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.5, 0.75, 0.9, 0.97, 1.0],
                alphas=np.logspace(-2.5, 0.8, 30),
                cv=5,
                max_iter=40000,
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

    @staticmethod
    def _term_degree(name: str) -> int:
        degree = 0
        for factor in name.split():
            if "^" in factor:
                degree += int(factor.rsplit("^", 1)[1])
            else:
                degree += 1
        return degree

    def __str__(self) -> str:
        enet = self.pipeline.named_steps["elasticnetcv"]
        scaler = self.pipeline.named_steps["standardscaler"]
        coefs = enet.coef_ / scaler.scale_
        intercept = enet.intercept_ - float(np.sum(coefs * scaler.mean_))
        term_names = self.term_names or self.feature_names
        active = [
            (name, coef, self._term_degree(name))
            for name, coef in zip(term_names, coefs, strict=True)
            if abs(coef) > 1e-8
        ]
        active_terms = [f"{coef:.3f}*{name}" for name, coef, _ in active]
        top = sorted(active, key=lambda item: abs(item[1]), reverse=True)
        top_features = ", ".join(f"{name} ({coef:.3f})" for name, coef, _ in top[:15])
        degree_counts = {
            degree: sum(1 for _, _, term_degree in active if term_degree == degree)
            for degree in range(1, 4)
        }
        active_by_degree = (
            f"linear: {degree_counts[1]}, quadratic: {degree_counts[2]}, "
            f"cubic: {degree_counts[3]}"
        )
        cubic_top = [(name, coef) for name, coef, degree in top if degree == 3][:8]
        cubic_summary = ", ".join(f"{name} ({coef:.3f})" for name, coef in cubic_top)
        equation = " + ".join(active_terms) if active_terms else "0.000"
        if len(active_terms) > 80:
            preview_terms = [f"{coef:.3f}*{name}" for name, coef, _ in top[:60]]
            equation = " + ".join(preview_terms) + " + ..."
        active_count = len(active_terms)
        zero_count = max(len(term_names) - active_count, 0)
        return (
            "Sparse cubic ElasticNetCV regression over original features, "
            "squared terms, pairwise interactions, and third-order polynomial "
            "terms. "
            f"Prediction equation in retained expanded terms: y = "
            f"{intercept:.3f} + {equation}. "
            f"Selected ElasticNet alpha: {enet.alpha_:.4g}; selected l1_ratio: "
            f"{enet.l1_ratio_:.2f}. Active expanded terms: {active_count}; "
            f"zeroed weak expanded terms: {zero_count}; active terms by degree: "
            f"{active_by_degree}. Top absolute active coefficient terms: "
            f"{top_features}. "
            f"Top retained cubic terms: {cubic_summary or 'none'}. "
            "Coefficients are shown after converting standardized polynomial "
            "terms back to their expanded-feature units. If the equation is "
            "truncated, omitted active terms are smaller than the displayed "
            "dominant terms but are still used by predict(); terms not shown "
            "because they were zeroed have exact zero coefficients from the "
            "sparse penalty. Positive active terms raise the prediction as "
            "their expanded value increases; negative active terms lower it. "
            "Counterfactual sensitivity is local and additive in this retained "
            "expanded basis, so changing an original feature can affect its "
            "linear term, squared/cubic powers, and retained interactions "
            "that include it."
        )
