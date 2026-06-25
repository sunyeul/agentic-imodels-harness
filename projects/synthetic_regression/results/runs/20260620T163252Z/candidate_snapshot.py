from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "direct_sparse_cubic_elasticnet_cv"
    notes = (
        "ElasticNetCV directly fits a sparse cubic polynomial basis, retaining "
        "the higher-order terms that improved RMSE while preserving shrinkage "
        "that the post-selection Ridge refit appeared to remove."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.term_names: list[str] = []
        self.selected_mask: np.ndarray | None = None
        self.poly = PolynomialFeatures(degree=3, include_bias=False)
        self.scaler = StandardScaler()
        self.model = ElasticNetCV(
            l1_ratio=[0.9, 0.97, 1.0],
            alphas=np.logspace(-2.7, -0.2, 45),
            cv=5,
            max_iter=40000,
        )

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        expanded = self.poly.fit_transform(X)
        scaled = self.scaler.fit_transform(expanded)
        self.model.fit(scaled, y)
        selected = np.abs(self.model.coef_) > 1e-8
        if not np.any(selected):
            top_count = min(20, selected.size)
            top_idx = np.argsort(np.abs(self.model.coef_))[-top_count:]
            selected = np.zeros_like(self.model.coef_, dtype=bool)
            selected[top_idx] = True
        self.selected_mask = selected
        self.term_names = list(self.poly.get_feature_names_out(self.feature_names))
        return self

    def predict(self, X):
        expanded = self.poly.transform(X)
        scaled = self.scaler.transform(expanded)
        if self.selected_mask is None:
            raise RuntimeError("CandidateModel must be fitted before predict().")
        return self.model.predict(scaled)

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
        if self.selected_mask is None:
            return (
                "Direct sparse cubic ElasticNetCV regression, not yet fitted. "
                "The model "
                "will expand original features through degree 3, use L1 "
                "dominant shrinkage to retain predictive expanded terms, and "
                "predict directly from the regularized cubic basis."
            )

        selected_names = [
            name
            for name, keep in zip(self.term_names, self.selected_mask, strict=True)
            if keep
        ]
        selected_scale = self.scaler.scale_[self.selected_mask]
        selected_mean = self.scaler.mean_[self.selected_mask]
        selected_scaled_coefs = self.model.coef_[self.selected_mask]
        coefs = selected_scaled_coefs / selected_scale
        intercept = self.model.intercept_ - float(np.sum(coefs * selected_mean))
        term_names = self.term_names or self.feature_names
        active = [
            (name, coef, self._term_degree(name))
            for name, coef in zip(selected_names, coefs, strict=True)
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
            "Direct sparse cubic ElasticNetCV regression over original features, "
            "squared terms, pairwise interactions, and third-order polynomial "
            "terms. ElasticNetCV predicts directly from the regularized cubic "
            "basis, correcting the prior post-selection Ridge refit by keeping "
            "coefficient shrinkage on retained high-order terms. "
            f"Prediction equation for nonzero retained expanded terms: y = "
            f"{intercept:.3f} + {equation}. "
            f"ElasticNet alpha: {self.model.alpha_:.4g}; l1_ratio: "
            f"{self.model.l1_ratio_:.3g}. Active expanded terms after direct "
            f"shrinkage: {active_count}; zeroed weak expanded terms: "
            f"{zero_count}; active terms by degree: "
            f"{active_by_degree}. Top absolute active coefficient terms: "
            f"{top_features}. "
            f"Top retained cubic terms: {cubic_summary or 'none'}. "
            "Coefficients are shown after converting standardized polynomial "
            "terms back to their expanded-feature units. If the equation is "
            "truncated, omitted active terms are smaller than the displayed "
            "dominant terms but are still used by predict(); terms not shown "
            "were removed by the sparse selection step. Positive active terms "
            "raise the prediction as their expanded value increases; negative "
            "active terms lower it. "
            "Counterfactual sensitivity is local and additive in this retained "
            "expanded basis, so changing an original feature can affect its "
            "linear term, squared/cubic powers, and retained interactions "
            "that include it."
        )
