from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "poly2_sparse_elasticnetcv"
    notes = (
        "ElasticNetCV over standardized degree-2 polynomial features to prune "
        "weak quadratic terms while retaining ridge-like shrinkage when useful."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.alphas = np.logspace(-4, 1.5, 50)
        self.l1_ratios = [0.01, 0.03, 0.07, 0.15, 0.3, 0.6, 1.0]
        self.pipeline = Pipeline(
            [
                ("input_scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("basis_scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        alphas=self.alphas,
                        l1_ratio=self.l1_ratios,
                        cv=5,
                        max_iter=50000,
                        random_state=42,
                        selection="cyclic",
                        tol=1e-5,
                    ),
                ),
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
        elasticnet = self.pipeline.named_steps["elasticnet"]
        poly = self.pipeline.named_steps["poly"]
        basis_names = poly.get_feature_names_out(self.feature_names)
        coefs = elasticnet.coef_
        nonzero_count = int(np.count_nonzero(np.abs(coefs) > 1e-8))
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
            "Sparse polynomial ElasticNetCV regression. "
            "Inputs are z-scored, expanded to degree-2 polynomial features "
            "(linear, squared, and pairwise interaction terms), then each "
            "generated basis column is standardized before elastic-net fitting. "
            "The second standardization makes the shrinkage penalty comparable "
            "across linear, squared, and interaction columns instead of letting "
            "higher-variance basis terms receive different effective shrinkage. "
            "Elastic-net alpha and L1 ratio are selected by inner 5-fold CV; "
            "small L1 ratios allow nearly ridge-like behavior, while larger "
            "ratios can prune weak polynomial terms. "
            f"Selected alpha: {elasticnet.alpha_:.4g}. "
            f"Selected L1 ratio: {elasticnet.l1_ratio_:.2f}. "
            "Intercept after input and basis standardization: "
            f"{elasticnet.intercept_:.3f}. "
            f"Top learned basis terms by absolute coefficient: {top_terms}. "
            f"Nonzero basis terms: {nonzero_count} of {len(basis_names)}. "
            f"Basis includes {degree_counts[1]} linear terms, "
            f"{degree_counts[2]} quadratic terms, with "
            f"{len(interaction_terms)} cross-feature interaction terms. "
            "Positive coefficients increase predictions as their standardized "
            "basis term rises; negative coefficients decrease predictions. "
            "Counterfactual sensitivity can be read from the dominant linear, "
            "squared, and interaction terms above, while smaller omitted terms "
            "are either shrunk or set to zero by the elastic-net penalty. The "
            "basis stays at degree 2 because the prior degree-3 expansion "
            "showed worse cross-validated RMSE."
        )
