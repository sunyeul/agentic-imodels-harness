from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toy_imodels.core.candidate import BaseCandidateModel


class CandidateModel(BaseCandidateModel):
    model_name = "post_lasso_cubic_ridge_cv"
    notes = (
        "L1-select a sparse cubic polynomial basis, then refit RidgeCV on the "
        "retained terms to reduce shrinkage bias while preserving the higher-"
        "order structure that improved RMSE."
    )

    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.term_names: list[str] = []
        self.selected_mask: np.ndarray | None = None
        self.poly = PolynomialFeatures(degree=3, include_bias=False)
        self.scaler = StandardScaler()
        self.selector = ElasticNetCV(
            l1_ratio=[1.0],
            alphas=np.logspace(-2.5, 0.8, 30),
            cv=5,
            max_iter=40000,
        )
        self.refit = RidgeCV(
            alphas=np.logspace(-4, 3, 50),
            cv=5,
            scoring="neg_mean_squared_error",
        )

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        expanded = self.poly.fit_transform(X)
        scaled = self.scaler.fit_transform(expanded)
        self.selector.fit(scaled, y)
        selected = np.abs(self.selector.coef_) > 1e-8
        if not np.any(selected):
            top_count = min(20, selected.size)
            top_idx = np.argsort(np.abs(self.selector.coef_))[-top_count:]
            selected = np.zeros_like(self.selector.coef_, dtype=bool)
            selected[top_idx] = True
        self.selected_mask = selected
        self.refit.fit(scaled[:, selected], y)
        self.term_names = list(self.poly.get_feature_names_out(self.feature_names))
        return self

    def predict(self, X):
        expanded = self.poly.transform(X)
        scaled = self.scaler.transform(expanded)
        selected_mask = self.selected_mask
        if selected_mask is None:
            raise RuntimeError("CandidateModel must be fitted before predict().")
        return self.refit.predict(np.asarray(scaled)[:, selected_mask])

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
                "Post-Lasso cubic RidgeCV regression, not yet fitted. The model "
                "will expand original features through degree 3, use L1 "
                "selection to retain predictive expanded terms, and refit a "
                "ridge model on that selected basis."
            )

        selected_names = [
            name
            for name, keep in zip(self.term_names, self.selected_mask, strict=True)
            if keep
        ]
        selected_scale = self.scaler.scale_[self.selected_mask]
        selected_mean = self.scaler.mean_[self.selected_mask]
        coefs = self.refit.coef_ / selected_scale
        intercept = self.refit.intercept_ - float(np.sum(coefs * selected_mean))
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
            "Post-Lasso cubic RidgeCV regression over original features, "
            "squared terms, pairwise interactions, and third-order polynomial "
            "terms. A LassoCV selector first retains the sparse cubic basis; "
            "RidgeCV then refits the prediction equation on only those retained "
            "expanded terms to reduce L1 shrinkage bias. "
            f"Prediction equation in retained expanded terms: y = "
            f"{intercept:.3f} + {equation}. "
            f"Selection Lasso alpha: {self.selector.alpha_:.4g}; refit Ridge "
            f"alpha: {self.refit.alpha_:.4g}. Active expanded terms after "
            f"selection/refit: {active_count}; zeroed weak expanded terms: "
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
